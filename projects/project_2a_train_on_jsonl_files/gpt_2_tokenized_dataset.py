from transformers import GPT2Tokenizer
from torch.utils.data import Dataset

class GPT2TokenizedDataset(Dataset):
    def __init__(self, text_based_dataset):
        self.text_based_dataset = text_based_dataset
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    def __len__(self):
        return len(self.text_based_dataset)
    
    def __getitem__(self, idx):
        return self.tokenizer(self.text_based_dataset[idx])['input_ids']