import random
from typing import List, Tuple
import torch
import torch.nn.functional as F

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
        # Each item in requests has a .args which is : Tuple[str, dict]
        # The first string is the prompt and the second dict is the parameters
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
        # Each item in requests has a .args which is : Tuple[str, str]
        # The first string is the context and the second string is the expected output
        
        # Pretty print the requests
        response = []

        for request in requests:
            with torch.no_grad():
                context, expected_output = request.args
                context_tokens = self.tokenize(context)
                expected_output_tokens = self.tokenize(expected_output)
                total_context = torch.cat((context_tokens, expected_output_tokens), dim=1)

                total_input = total_context[:, :-1] # Remove the last token

                generated_output_logits, _ = self.model(total_input)
                print(generated_output_logits.shape)
                generated_output_logits = generated_output_logits[:, -len(expected_output_tokens[0]):] # Only take the logits corresponding to the expected output
                print(generated_output_logits.shape)
                assert generated_output_logits.shape[2] == self.tokenizer.vocab_size, "The number of logits should be equal to the vocab size"

                # We now have to figure out which of these logits correspond to the expected output

                log_likelihoods_of_generated_output_logits = F.log_softmax(generated_output_logits, dim=2)

                # Now we need to index into this with the expected_output_tokens
                # We need to get the log likelihood of the expected_output_tokens and then sum them up
                # We can do this by using the gather method
                # The gather method takes the indices of the elements to gather from the tensor
                # The first argument is the dimension to gather from
                # The second argument is the indices to gather
                # The third argument is the dimension to gather into
                print(log_likelihoods_of_generated_output_logits.shape)
                print(expected_output_tokens.shape)
                log_likelihoods_of_expected_output = torch.gather(log_likelihoods_of_generated_output_logits, 2, expected_output_tokens.unsqueeze(2)).squeeze(2) # We unsqueeze the expected_output_tokens so that the number of dimensions between the input and the index remains the same
                # If it was the greedy response, then the argmax along the last dimension of the log_likelihoods_of_generated_output_logits would be the expected_output_tokens
                is_greedy = torch.equal(log_likelihoods_of_generated_output_logits.argmax(dim=2), expected_output_tokens)
                # Because these are log_likelihoods, then the sum of them is the log likelihood of the expected output (if they weren't logged, it'd be the product)
                response.append((log_likelihoods_of_expected_output.sum().item(), is_greedy))
        
        return response




    def loglikelihood_rolling(self, requests) -> List[float]:
        # Each item in requests has a .args which is : Tuple[str,]
        # The string is the input string to the model whose entire loglikelihood, conditioned purely on the EOT token, will be calculated
        response = []
        eot_token = torch.tensor(self.tokenizer.encode("<|endoftext|>"), dtype=torch.long).unsqueeze(0)
        for request in requests:
            with torch.no_grad():
                input_string = request.args[0]
                expected_tokens = self.tokenize(input_string)
                input_tokens = torch.cat((eot_token, expected_tokens), dim=1) # Prepend the EOT token
                input_tokens = input_tokens[:, :-1] # Remove the last token so that our model can predict it
                logits = self.model(input_tokens)
                assert logits.shape[2] == self.tokenizer.vocab_size, "The number of logits should be equal to the vocab size"
                assert logits.shape[1] == len(expected_tokens[0]), "The number of logit sets should be equal to the number of tokens in the expected output"

                log_likelihoods_of_logits = F.log_softmax(logits, dim=2)
                log_likelihoods_of_expected_output = torch.gather(log_likelihoods_of_logits, 2, expected_tokens.unsqueeze(2)).squeeze(2)
                response.append(log_likelihoods_of_expected_output.sum().item())
        return response