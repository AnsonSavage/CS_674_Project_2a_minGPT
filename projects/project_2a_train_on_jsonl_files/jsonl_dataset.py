from torch.utils.data import Dataset
import json

class JSONLDataset(Dataset):
    def __init__(self, path_to_jsonl, length=None):
        self.path_to_jsonl = path_to_jsonl
        # Count number of lines in the file
        if length is None:
            with open(path_to_jsonl, 'r') as f:
                self.length = sum(1 for line in f)
        else:
            self.length = length
        
        self.line_cache = {}
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx in self.line_cache:
            return self.line_cache[idx]

        with open(self.path_to_jsonl, 'r') as f:
            for i, line in enumerate(f):
                if i == idx:
                    json_obj = json.loads(line.strip())
                    value = json_obj['text']
                    self.line_cache[idx] = value
                    return value
                    