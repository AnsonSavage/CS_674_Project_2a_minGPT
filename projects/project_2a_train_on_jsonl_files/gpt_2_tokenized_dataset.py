from transformers import GPT2Tokenizer
from torch.utils.data import Dataset
import torch

import sys
sys.path.append('/home/ansonsav/cs_674/project_2a_minGPT/minGPT')
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

class GPT2TokenizedDataset(Dataset):

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 1024
        return C

    def __init__(self, text_based_dataset):
        self.text_based_dataset = text_based_dataset
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.vocab_size = self.tokenizer.vocab_size
        self.block_size = GPT2TokenizedDataset.get_default_config().block_size
    
    def __len__(self):
        return len(self.text_based_dataset)
    
    def __getitem__(self, idx):
        tokens = torch.tensor(self.tokenizer(self.text_based_dataset[idx], truncation=True, max_length=self.block_size)['input_ids'])

        # TODO: we can consider randomly sampling a subsequence of tokens here
        # if len(tokens) > self.block_size:
        #     # Randomly sample 
        #     tokens = tokens[:self.block_size]
        return tokens[:-1], tokens[1:]
    
    def get_vocab_size(self):
        return self.vocab_size
    
    @staticmethod
    def get_block_size():
        return GPT2TokenizedDataset.get_default_config().block_size