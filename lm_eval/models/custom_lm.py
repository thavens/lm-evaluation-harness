import json
from importlib import resources
from pathlib import Path
from typing import List, Tuple

import torch
import yaml
from torch_llms import inference, models
from torch_llms.messages import configs, tokenization
from tqdm import tqdm
from transformers import AutoTokenizer

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model


@register_model("torch_llms")
class TorchLLM(TemplateLM):
    def __init__(self, base_path, lora_path=None, tokenizer_config=None, temperature=0.0, device="cuda", precision="bfloat16", batch_size=1, add_bos_token=False) -> None:
        super().__init__()
        
        self.base_path = Path(base_path)
        self.lora_path = Path(lora_path) if lora_path else None
        self.temperature = float(temperature)
        self.device = device
        self.precision = precision
        self.batch_size = int(batch_size)
        self.add_bos_token = add_bos_token
        self.ll_cache = {}
        
        # load model
        main_ckpt_dir = self.lora_path.parent if self.lora_path else self.base_path.parent
        self.tokenizer = AutoTokenizer.from_pretrained(main_ckpt_dir, trust_remote_code=True)
        
        if tokenizer_config is not None:
            with resources.files(configs).joinpath(tokenizer_config).open() as file:
                self.tokenizer_config = tokenization.TokenizerConfig(**yaml.safe_load(file))
            self.eot_ids = self.tokenizer_config.assistant_end
        else:
            self.tokenizer_config = None
            self.eot_ids = [self.tokenizer.eos_token_id]
        
        with open(main_ckpt_dir / "params.json", "r") as f:
            params = json.load(f)
        
        
        lora_args = None
        if (main_ckpt_dir / "lora_args.json").exists():
            with open(main_ckpt_dir / "lora_args.json", "r") as f:
                lora_args = json.load(f)
        
        if device == "cpu":
            params["use_flash_attn"] = False
        
        model_params = models.llama.ModelParams(**params)
        
        with torch.device("meta"):
            self.model = models.llama.Transformer(model_params)
        
        if lora_args is not None:
            replace_linear_with_lora = models.lora.make_replace_linear_with_lora(
                rank=lora_args["lora_rank"],
                alpha=lora_args["lora_alpha"],
                dropout=lora_args["lora_dropout"],
            )
            self.model.apply(replace_linear_with_lora)
        
            if lora_args.get("lora_embedding", False):
                replace_embedding_with_lora = models.lora.make_replace_embedding_with_lora(
                    rank=lora_args["lora_rank"],
                    alpha=lora_args["lora_alpha"],
                    dropout=lora_args["lora_dropout"],
                )
                self.model.apply(replace_embedding_with_lora)
        
        if self.lora_path:
            self.model = models.load_model_weights(
                [self.base_path, self.lora_path],
                model=self.model,
                precision=precision,
                device=device,
            )
        else:
            self.model = models.load_model_weights(
                [self.base_path],
                model=self.model,
                precision=precision,
                device=device,
            )
        
        with torch.device(device):
            models.utils.init_meta_params(self.model)
        
        if lora_args is not None:
            # merge LoRA weights into the original model weights for efficient inference
            self.model.apply(models.lora.replace_lora_with_linear)
            if lora_args.get("lora_embedding", False):
                self.model.apply(models.lora.replace_lora_with_embedding)
        
        self.model.eval()
        self.kv_cache = self.model.init_kv_cache(1, device)
            
    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.eot_token_id

    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_bos_token is set
        if add_special_tokens is None:
            special_tokens_kwargs = {
                "add_special_tokens": False or self.add_bos_token
            }
        # otherwise the method explicitly defines the value
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def _loglikelihood_tokens_unbatched(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        """Log likelihood given context and generation string as well as context and generation tokens

        Args:
            requests (List[Tuple[Tuple[str, str], List[int], List[int]]]): batch[((context string, generation string), context tokens, generation tokens)]
            disable_tqdm (bool, optional): Defaults to False.

        Returns:
            List[Tuple[float, bool]]: batch list of (log likelihood, )
        """
        # generate our sequences and masks
        assert len(requests) == 1
        
        if requests[0][0][0] in self.ll_cache and len(requests[0][2]) == 1:
            traj_probs = self.ll_cache[requests[0][0][0]]
            greedy = traj_probs.argmax() == requests[0][2][0]
            return [(traj_probs[requests[0][2][0]].item(), bool(greedy))]
        
        masks = [[0] * len(context_tokens) + [1] * len(generation_tokens) for _, context_tokens, generation_tokens in requests]
        masks = torch.tensor(masks, dtype=torch.bool, device=self.device)
        
        sequences = [context_tokens + generation_tokens for _, context_tokens, generation_tokens in requests]
        
        sequences = torch.tensor(sequences, dtype=torch.int64, device=self.device)
        sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.eot_token_id)
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)[:, 1:]
        
        labels = sequences[:, 1:]
        sequences = sequences[:, :-1]
        
        with torch.inference_mode():
            logits, _ = self.model.forward(sequences)
            logits = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # gather logits for the actual sequence
        expanded_log_probs = torch.gather(logits, 2, labels.unsqueeze(2)).squeeze(2)
        traj_log_probs = torch.sum(expanded_log_probs * masks, dim=1)
        
        # insane hack to cache the log likelihoods
        if len(requests[0][2]) == 1:
            self.ll_cache[requests[0][0][0]] = traj_log_probs.item() - expanded_log_probs[0, -1].item() + logits[0, -1]
        
        # get argmax of the logits to check greedy decoding
        # set the masked tokens to 0 for both argmax and sequences
        # this is to ensure that the masked tokens are not considered in the greedy decoding check
        argmax = logits.max(dim=-1)[1]
        greedy_decoding = ((argmax == labels) * masks).all(dim=1)
        
        return list(zip(traj_log_probs.tolist(), greedy_decoding.tolist()))

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        """Log likelihood given context and generation string as well as context and generation tokens

        Args:
            requests (List[Tuple[Tuple[str, str], List[int], List[int]]]): batch[((context string, generation string), context tokens, generation tokens)]
            disable_tqdm (bool, optional): Defaults to False.

        Returns:
            List[Tuple[float, bool]]: batch list of (log likelihood, )
        """
        results = []
        for i in tqdm(range(0, len(requests)), disable=disable_tqdm):
            results.extend(self._loglikelihood_tokens_unbatched([requests[i]], disable_tqdm))
        return results
        
    
    def loglikelihood_rolling(
        self, requests, disable_tqdm: bool = False
    ) -> List[float]:
        generations = [request.args[0] for request in requests]
        generations = self.tokenizer(generations, add_special_tokens=False)
        generations = [generation + [self.eot_token_id] for generation in generations]
        
        context_strings = [self.tokenizer.bos_token] * len(generations)
        context_tokens = [self.tokenizer.bos_token_id] * len(generations)
        
        new_requests = []
        for context_string, context_token, generation in zip(context_strings, context_tokens, generations):
            new_requests.append(((context_string, generation), context_token, generation))
        return (self._loglikelihood_tokens(new_requests, disable_tqdm)[0],)

    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False) -> List[str]:
        conversations = []
        max_lengths = []
        stop_seqs = []
        for request in requests:
            conversations.append([{"role": "user", "content": request.args[0]}])
            max_lengths.append(request.args[1]["until"])
            stop_seqs.append(request.args[1]["max_gen_toks"])
        
        return self.generate(conversations, max_lengths, stop_seqs)
        
    def generate(self, conversations: list[list[dict[str, str]]], max_lengths: list[int], stop_seqs: list[list[str]]) -> list[str]:
        prompts = self.tokenizer.apply_chat_template(conversations, add_generation_prompt=True)
        
        responses = []
        for prompt, max_length, stop_seq in zip(prompts, max_lengths, stop_seqs):
            self.kv_cache.reset()
            
            new_tokens = []
            encoded = torch.tensor(prompt, dtype=torch.int, device=self.device)

            logits, self.kv_cache = self.model(encoded.view(1, -1), kv_cache=self.kv_cache)
            cur_token, _ = inference.utils.sample(logits)
            new_tokens.append(cur_token.item())

            if new_tokens[-len(self.eot_ids) :] == self.eot_ids:
                return [], self.kv_cache

            for i in range(1, max_length):
                logits, self.kv_cache = self.model(cur_token.view(1, -1), kv_cache=self.kv_cache)
                cur_token, _ = inference.utils.sample(logits)
                new_tokens.append(cur_token.item())

                if new_tokens[-len(self.eot_ids) :] == self.eot_ids:
                    break

            if i == max_length - 1:
                print("[warning] max_new_tokens reached")
                new_tokens.extend(self.eot_ids)
            
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            for stop_seq in stop_seqs:
                if len(stop_seq) > 0:
                    response = response.split(stop_seq)[0]
            
            responses.append(response)

        return responses
    