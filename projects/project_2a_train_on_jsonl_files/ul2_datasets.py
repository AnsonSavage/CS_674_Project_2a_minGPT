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
        self.tokenizer = tokenized_dataset.tokenizer
        
        # Get the mode token
        self.mode_token = self.tokenizer.encode(mode_token_string)
        assert len(self.mode_token) == 1
        self.mode_token = torch.tensor(self.mode_token)

        # Set the suffix and end tokens
        self.suffix_token = torch.tensor(self.tokenizer.encode('[SUFFIX]'))
        self.end_token = torch.tensor(self.tokenizer.encode('<|endoftext|>'))

    def __len__(self):
        return len(self.tokenized_dataset)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_dataset[idx]
        tokens = self._prepend_mode_token(tokens)
        
        # TODO: what is the prefix and what is the suffix?
        return self._noise(tokens)

    def create_prefix_and_suffix(self, corrupted_tokens, corrupted_spans, removed_tokens):
        """ Creates the prefix and suffix for the denoising task
        Args:
        Returns:
        """
        
        # Create the prefix
        train_tensor = None
        assert len(corrupted_spans) == len(removed_tokens)
        for i in range(len(corrupted_spans)):
            extra_id_token = torch.tensor(self.tokenizer.encode(f'<extra_id_{i}>'))
            corrupted_span_start, corrupted_span_end = corrupted_spans[i]

            # If we're on the first iteration, we want the train tensor to go from the start of the tokens to the start of the corrupted span
            if train_tensor is None:
                assert i == 0
                train_tensor = corrupted_tokens[:corrupted_span_start].clone() # TODO: is cloning necessary?
            else:
                previous_corrupted_span_start, previous_corrupted_span_end = corrupted_spans[i - 1]
                train_tensor = torch.cat((train_tensor, corrupted_tokens[previous_corrupted_span_end:corrupted_span_start].clone()))
            train_tensor = torch.cat((train_tensor, extra_id_token))
        
        # Create the suffix
        train_tensor = torch.cat((train_tensor, self.suffix_token)) # add the suffix token
        for i in range(len(removed_tokens)):
            removed_token = removed_tokens[i]
            extra_id_token = torch.tensor(self.tokenizer.encode(f'<extra_id_{i}>'))
            train_tensor = torch.cat((train_tensor, extra_id_token, removed_token))

        # Add the end token
        train_tensor = torch.cat((train_tensor, self.end_token))
        



    def corrupt(self, tokens, min_span_size: int, max_span_size: int, percentage_to_corrupt: float) -> tuple:
        """ Corrupts a percentage of the tokens with spans of random size between min_span_size and max_span_size. Returns the corrupted tokens, the corresponding spans, and tokens that were removed
        Args:
            tokens (torch.Tensor): The tokens to corrupt
            min_span_size (int): The minimum size of the span to corrupt
            max_span_size (int): The maximum size of the span to corrupt
            percentage_to_corrupt (float): The percentage of the tokens to corrupt
        Returns:
            tuple: A tuple containing the corrupted tokens, the corrupted spans (a list of tuples representing start and end indices of corruption) and a list of the tokens that were removed
        """
        removed_tokens = []
        corrupted_spans = []
        percentage_corrupted = 0
        corruption_token = -1 # A placeholder that represents a corrupted token
        while percentage_corrupted < percentage_to_corrupt:
            span_size = torch.randint(min_span_size, max_span_size, (1,)).item()
            start_index = torch.randint(0, len(tokens) - span_size, (1,)).item()
            end_index = start_index + span_size
            if tokens[start_index:end_index].eq(corruption_token).any(): # If the span already contains a corruption token, perform rejection sampling
                continue
            removed_tokens.append(tokens[start_index:end_index].clone())
            corrupted_spans.append((start_index, end_index))

            tokens[start_index:end_index] = corruption_token
            percentage_corrupted += span_size / len(tokens)
        return tokens, corrupted_spans, removed_tokens

    
    def _prepend_mode_token(self, tokens):
        return torch.cat((self.mode_token, tokens))

    @abstractmethod
    def _noise(self, tokens):
        pass
    

class RegularDenoisingDataset(DenoisingDataset):
    """ Prepends the [NLU] token to the tokens
    Has a standard span corruption noise model
    It corrupts spans with a uniform mean of 3 and a corruption rate of 15%
    """
    def __init__(self, tokenized_dataset: GPT2TokenizedDataset):
        super().__init__(tokenized_dataset, '[NLU]')
    
    def _noise(self):
        return self.corrupt(tokens, 1, 5, 0.15)

class ExtremeDenoisingDataset(DenoisingDataset):
    """ Prepends the [NLG] token to the tokens
    """
    def __init__(self, tokenized_dataset: GPT2TokenizedDataset):
        super().__init__(tokenized_dataset, '[NLG]')

class SequentialDenoising(DenoisingDataset):
    """ Prepends the [S2S] token to the tokens
    """
    def __init__(self, tokenized_dataset: GPT2TokenizedDataset):
        super().__init__(tokenized_dataset, '[S2S]')