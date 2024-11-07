import random
from typing import List, Tuple
import torch

import sys
sys.path.append('/home/ansonsav/cs_674/project_2a_minGPT/minGPT/mingpt')

from tqdm import tqdm

from lm_eval.api.model import LM, TemplateLM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
# from lm_eval.api.instance import Instance
# from transformers import GPT2Tokenizer
# from model import GPT


@register_model("min-gpt")
class MinGPTEval(LM):
    # def __init__(self, model: GPT, tokenizer: GPT2Tokenizer) -> None:
    def __init__(self, model, tokenizer) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_until(self, requests: list) -> List[str]:
        print(requests)
        total_output = []

        for request in requests:
            prompt, param = request.args
            until = param['until'] # A list of tokens indicating when to stop generating
            max_gen_toks = param['max_gen_toks'] # The maximum number of tokens to generate otherwise

            tokens = self.tokenize(prompt)
            for _ in range(max_gen_toks):
                next_token = self.model.generate_next_token(tokens)
                tokens = torch.cat((tokens, next_token), dim=1)
                if self.tokenizer.decode(next_token[0]) in until:
                    break
            
            total_output.append(self.tokenizer.decode(tokens[0]))

        return total_output

    def tokenize(self, string):
        tokens = torch.tensor(self.tokenizer(string)['input_ids'], dtype=torch.long).unsqueeze(0)
        return tokens
    
    def loglikelihood(self, requests) -> List[Tuple[float | bool]]:
        for request in requests:
            context, expected_output = request.args
            context_tokens = self.tokenize(context)
            expected_output_tokens = self.tokenize(expected_output)
            generated_tokens = []
            while len(generated_tokens) < len(expected_output_tokens):
                next_token = self.model.generate_next_token(context_tokens)
                generated_tokens.append(next_token)
                context_tokens = torch.cat((context_tokens, next_token), dim=1)

    def loglikelihood_rolling(self, requests) -> List[float]:
        pass