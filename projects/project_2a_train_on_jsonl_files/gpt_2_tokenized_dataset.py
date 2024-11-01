from transformers import GPT2Tokenizer
from torch.utils.data import Dataset

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
    
    def __len__(self):
        return len(self.text_based_dataset)
    
    def __getitem__(self, idx):
        return self.tokenizer(self.text_based_dataset[idx])['input_ids']
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_block_size():
        return GPT2TokenizedDataset.get_default_config().block_size