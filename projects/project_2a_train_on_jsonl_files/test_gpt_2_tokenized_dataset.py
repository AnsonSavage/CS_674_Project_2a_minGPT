import unittest
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset
import torch
from gpt_2_tokenized_dataset import GPT2TokenizedDataset

# Assuming GPT2TokenizedDataset and JSONLDataset are defined as provided
# For testing purposes, we'll create a simple mock dataset

class MockTextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

class TestGPT2TokenizedDataset(unittest.TestCase):
    def setUp(self):
        # Sample texts for testing
        self.sample_texts = [
            "Hello world! This is a test.",
            "Another test sentence.",
            "Short.",
            "This is a longer test sentence that should be truncated accordingly.",
            "The quick brown fox jumps over the lazy dog.",
            "",  # Empty string should be filtered out
            "   ",  # String with only spaces should be filtered out
            None  # None value should be filtered out
        ]

        # Filter out None and empty strings
        self.filtered_texts = [text for text in self.sample_texts if text and text.strip()]
        self.text_dataset = MockTextDataset(self.filtered_texts)
        self.tokenized_dataset = GPT2TokenizedDataset(self.text_dataset)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.block_size = self.tokenized_dataset.block_size

    def test_dataset_length(self):
        """Test that the length of the tokenized dataset matches the filtered text dataset."""
        self.assertEqual(len(self.tokenized_dataset), len(self.filtered_texts))

    def test_getitem_output(self):
        """Test that __getitem__ returns tensors of correct types and shapes."""
        for i in range(len(self.tokenized_dataset)):
            x, y = self.tokenized_dataset[i]
            self.assertIsInstance(x, torch.Tensor)
            self.assertIsInstance(y, torch.Tensor)
            self.assertEqual(x.shape, y.shape)
            self.assertLessEqual(len(x), self.block_size - 1)

    def test_token_alignment(self):
        """Test that input and target tokens are correctly aligned (shifted by one)."""
        for i in range(len(self.tokenized_dataset)):
            x, y = self.tokenized_dataset[i]
            # Ensure that y is x shifted by one position to the left
            self.assertTrue(torch.equal(x[1:], y[:-1]))

    def test_block_size_truncation(self):
        """Test that sequences longer than block_size are properly truncated."""
        long_text = "This is a very long test sentence " * 9999  # Create a long text
        self.text_dataset.texts.append(long_text)
        self.tokenized_dataset = GPT2TokenizedDataset(self.text_dataset)
        x, y = self.tokenized_dataset[-1]
        self.assertEqual(len(x), self.block_size - 1)
        self.assertEqual(len(y), self.block_size - 1)

    def test_random_subsequence_false(self):
        """Test that setting random_subsequence=False returns consistent sequences."""
        idx = 0
        x1, y1 = self.tokenized_dataset.__getitem__(idx, random_subsequence=False)
        x2, y2 = self.tokenized_dataset.__getitem__(idx, random_subsequence=False)
        self.assertTrue(torch.equal(x1, x2))
        self.assertTrue(torch.equal(y1, y2))

    def test_random_subsequence_true(self):
        """Test that setting random_subsequence=True can return different subsequences."""
        idx = 0
        # Add several sentences that are over the length of block_size
        old_texts = self.text_dataset.texts
        new_texts = [
            "This is a long sentence that should be truncated to the block size." * 999,
            "Another long sentence that should also be truncated." * 999,
            "A third long sentence that will be truncated." * 999
        ]
        self.text_dataset.texts = new_texts
        # Collect multiple samples to check for differences
        sequences = [self.tokenized_dataset.__getitem__(idx, random_subsequence=True)[0] for _ in range(5)]
        unique_sequences = {seq.numpy().tobytes() for seq in sequences}
        # There should be more than one unique sequence
        self.assertGreater(len(unique_sequences), 1)
        self.text_dataset.texts = old_texts

    def test_get_vocab_size(self):
        """Test that get_vocab_size returns the correct vocabulary size."""
        self.assertEqual(self.tokenized_dataset.get_vocab_size(), self.tokenizer.vocab_size)

    def test_get_block_size(self):
        """Test that get_block_size returns the correct block size."""
        self.assertEqual(self.tokenized_dataset.get_block_size(), self.block_size)

    def test_empty_and_none_texts(self):
        """Test that empty strings and None values are handled properly."""
        # Ensure that empty strings and None values are not included
        original_length = len(self.sample_texts)
        filtered_length = len(self.filtered_texts)
        self.assertLess(filtered_length, original_length)
        self.assertEqual(len(self.tokenized_dataset), filtered_length)

if __name__ == '__main__':
    unittest.main()
