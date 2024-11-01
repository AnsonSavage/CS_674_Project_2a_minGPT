from torch.utils.data import Dataset
import json
import os
import pickle
from tqdm import tqdm

class JSONLDataset(Dataset):
    def __init__(self, path_to_jsonl, length=10000):
        self.path_to_jsonl = path_to_jsonl
        self.pkl_path = path_to_jsonl.replace('.jsonl', '.pkl')

        if length is not None:
            self.pkl_path = self.pkl_path.replace('.pkl', f'size_{length}.pkl')

        # Check if a pickle file exists for faster loading
        if os.path.exists(self.pkl_path):
            # Load the dataset from the pickle file
            with open(self.pkl_path, 'rb') as f:
                self.items = pickle.load(f)
        else:
            # Load the dataset from the jsonl file and cache it in a pickle file
            self.items = []
            with open(path_to_jsonl, 'r') as f:
                for line in tqdm(f):
                    self.items.append(json.loads(line)['text'])
                    if length is not None and len(self.items) >= length:
                        break
            # Save the loaded data to a pickle file for faster future loading
            with open(self.pkl_path, 'wb') as f:
                pickle.dump(self.items, f)

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]
