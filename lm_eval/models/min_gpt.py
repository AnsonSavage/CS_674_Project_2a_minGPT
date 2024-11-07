import random
from typing import List, Tuple
import torch

import sys
sys.path.append('/home/ansonsav/cs_674/project_2a_minGPT/minGPT/mingpt')

from tqdm import tqdm

from lm_eval.api.model import LM, TemplateLM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance
from transformers import GPT2Tokenizer
from model import GPT


@register_model("min_gpt", "mingpt", "min-gpt")
class MinGPTEval(LM):
    def __init__(self, model: GPT, tokenizer: GPT2Tokenizer) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = model.tokenizer
    
    def generate_until(self, requests: list[Instance]) -> List[str]:
        prompts = [r.args[0] for r in requests]
        params = [r.args[1] for r in requests]
        total_output = []

        for i, prompt in enumerate(prompts):
            param = params[i]
            until = param['until'] # A list of tokens indicating when to stop generating
            max_gen_toks = param['max_gen_toks'] # The maximum number of tokens to generate otherwise

            tokens = torch.tensor(self.tokenizer(prompt)['input_ids'], dtype=torch.long).unsqueeze(0)
            for _ in range(max_gen_toks):
                next_token = self.model.generate_next_token(tokens)
                tokens = torch.cat((tokens, next_token), dim=1)
                if self.tokenizer.decode(next_token[0]) in until:
                    break
            
            total_output.append(self.tokenizer.decode(tokens[0]))

        return total_output
    
    def loglikelihood(self, requests) -> List[Tuple[float | bool]]:
        return super().loglikelihood(requests)