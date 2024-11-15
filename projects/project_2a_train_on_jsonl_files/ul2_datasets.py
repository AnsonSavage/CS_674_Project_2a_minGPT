import torch
from torch.utils.data import Dataset
from gpt_2_tokenized_dataset import GPT2TokenizedDataset
from abc import ABC, abstractmethod

class DenoisingDataset(Dataset, ABC):
    """ Abstract class containing common functionality across denoising datasets.

    Each denoising dataset will have a GPT2TokenizedDataset (probably the same instance of it for efficiency)

    A mixture of denoisers (50% PrefixLM [Sequential], 25% Extreme Denoising, and 25% Regular Span Corruption) will be used
    """
    def __init__(self, tokenized_dataset: GPT2TokenizedDataset, mode_token_string):
        self.tokenized_dataset = tokenized_dataset
        self.mode_token = mode_token_string
    
    def __len__(self):
        return len(self.tokenized_dataset)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_dataset[idx]
        # TODO: prepend the mode token to the tokens
        return self._noise(tokens)

    @abstractmethod
    def _noise(self, tokens):
        pass
    

class RegularDenoisingDataset(DenoisingDataset):
    """ Prepends the [NLU] token to the tokens
    Has a standard span corruption noise model
    It corrupts spans with a uniform mean of 3 and a corruption rate of 15%
    """
    pass

class ExtremeDenoisingDataset(DenoisingDataset):
    """ Prepends the [NLG] token to the tokens
    """
    pass

class SequentialDenoising(DenoisingDataset):
    """ Prepends the [S2S] token to the tokens
    """
    pass