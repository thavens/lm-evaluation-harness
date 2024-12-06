from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model


@register_model("torchllms")
class torchllms(TemplateLM):
    def __init__(
        self,
        base_path,
        lora_path=None,
        template_config=None,
        temperature=0.0,
        device="cuda",
        precision="bfloat16",
        batch_size=1,
        add_bos_token=False,
    ) -> None:
        super().__init__()

        from torchllms.models import utils

        ckpt_paths = [base_path]
        if lora_path:
            ckpt_paths.append(lora_path)

        model, tokenizer, template_config = utils.setup_model_and_tokenizer(
            ckpt_paths,
            template_config,
            device,
            precision,
        )
        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        self.template_config = template_config
        self.temperature = float(temperature)
        self.device = device

        self.add_bos_token = add_bos_token
        self.logprobs_cache = {}

        if template_config:
            self.eot_ids = self.template_config.assistant_end
        else:
            self.eot_ids = [self.tokenizer.eos_token_id]

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.eot_token_id

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )

        return chat_templated

    # XXX: copied from HFLM huggingface model class, untested
    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_bos_token is set
        if add_special_tokens is None:
            special_tokens_kwargs = {"add_special_tokens": False or self.add_bos_token}
        # otherwise the method explicitly defines the value
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def _loglikelihood_tokens_single(
        self,
        context_tokens: List[int],
        generation_tokens: List[int],
    ) -> List[Tuple[float, bool]]:
        """Calculate log likelihood of generation tokens given context tokens.

        Caches single-token generations which is useful in MMLU or similar multiple choice tasks.

        Args:
            context_tokens (List[int]): List of token IDs for the context.
            generation_tokens (List[int]): List of token IDs for the generation.

        Returns:
            logprob (float): Log probability of the generation tokens.
            is_greedy (bool): Whether the generation tokens are the greedy choice.
        """
        context_tokens_t = tuple(context_tokens)
        if context_tokens_t in self.logprobs_cache and len(generation_tokens) == 1:
            logprobs, argmax = self.logprobs_cache[context_tokens_t]
            is_greedy = argmax == generation_tokens[0]
            logprob = logprobs[generation_tokens[0]]
            return logprob, is_greedy

        mask = [0] * len(context_tokens) + [1] * len(generation_tokens)
        mask = mask[1:]
        mask = torch.tensor(mask, dtype=torch.bool, device=self.device)

        full_tokens = context_tokens + generation_tokens
        sequences = torch.tensor(
            full_tokens[:-1], dtype=torch.int64, device=self.device
        )
        labels = torch.tensor(full_tokens[1:], dtype=torch.int64, device=self.device)

        with torch.inference_mode():
            logits, _ = self.model.forward(sequences.view(1, -1))
            logits = logits.squeeze(0)  # shape: [seq_len, vocab_size]
            full_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

        traj_logprobs = torch.gather(full_logprobs, 1, labels.unsqueeze(1)).squeeze(1)
        logprob = torch.sum(traj_logprobs * mask).item()

        if len(generation_tokens) == 1:
            logprobs = full_logprobs[-1]
            argmax = logprobs.argmax().item()
            self.logprobs_cache[context_tokens_t] = (logprobs.tolist(), argmax)

        # ignore masked tokens when checking for greedy decoding
        traj_argmax = full_logprobs.argmax(dim=-1)
        is_greedy = ((traj_argmax == labels) * mask).all().item()

        return logprob, is_greedy

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        results = []
        for (context, generation), context_tokens, generation_tokens in tqdm(
            requests, disable=disable_tqdm
        ):
            logprob, is_greedy = self._loglikelihood_tokens_single(
                context_tokens, generation_tokens
            )
            results.append((logprob, is_greedy))
        return results

    # XXX: should we be adding bos/eos tokens here?
    def loglikelihood_rolling(
        self, requests, disable_tqdm: bool = False
    ) -> List[float]:
        generations = [request.args[0] for request in requests]
        generations = self.tokenizer(generations, add_special_tokens=False)
        generations = [generation + [self.eot_token_id] for generation in generations]

        context_strings = [self.tokenizer.bos_token] * len(generations)
        context_tokens = [self.tokenizer.bos_token_id] * len(generations)

        new_requests = []
        for context_string, context_token, generation in zip(
            context_strings, context_tokens, generations
        ):
            new_requests.append(
                ((context_string, generation), context_token, generation)
            )
        return (self._loglikelihood_tokens(new_requests, disable_tqdm)[0],)

    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        raise NotImplementedError()
