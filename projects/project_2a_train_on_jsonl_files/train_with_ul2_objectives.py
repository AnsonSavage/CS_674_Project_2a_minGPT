from jsonl_dataset import JSONLDataset
from gpt_2_tokenized_dataset import GPT2TokenizedDataset
import os
import matplotlib.pyplot as plt
from ul2_datasets import RegularDenoisingDataset, ExtremeDenoisingDataset, SequentialDenoising
from transformers import GPT2Tokenizer


import sys
sys.path.append('/home/ansonsav/cs_674/project_2a_minGPT/minGPT')

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
import argparse

def get_config():
    parser = argparse.ArgumentParser(description='Train GPT model on JSONL files')
    parser.add_argument('--work_dir', type=str, required=True, help='Working directory for output')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the JSONL data file')
    args = parser.parse_args()

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = args.work_dir

    # data
    C.data = GPT2TokenizedDataset.get_default_config()
    C.data_path = args.data_path

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt2'
    # C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    # Can overwrite things like learning rate here
    C.trainer.learning_rate = 5e-4
    C.trainer.batch_size = 1
    C.trainer.num_workers = 1

    return C

if __name__ == '__main__':
    # get the config
    config = get_config()

    setup_logging(config)
    set_seed(config.system.seed)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Add the mode switching tokens, [NLU], [NLG], and [S2S] for regular denoising, extreme denoising, and sequential denoising respectively
    tokenizer.add_tokens(['[NLU]', '[NLG]', '[S2S]'], special_tokens=True) # TODO: use add_special_tokens if this doesn't

    # Add the suffix token
    tokenizer.add_tokens(['[SUFFIX]'], special_tokens=True)

    # The end token will just use '<|endoftext|>'

    # Add sentinel tokens, <extra_id_0> through <extra_id_99>
    tokenizer.add_tokens([f'<extra_id_{i}>' for i in range(100)]) # TODO: do they need to be special tokens?
    
    tokenized_dataset = GPT2TokenizedDataset(JSONLDataset(config.data_path, length=None), tokenizer) # length=None means use all the data
    # TODO: Create the denoising datasets
    regular_denoising_dataset = RegularDenoisingDataset(tokenized_dataset)

    # TODO: is the attention mask configured in the tokenizer?

    config.model.vocab_size = tokenized_dataset.get_vocab_size()
    config.model.block_size = tokenized_dataset.get_block_size()

    # create the model
    model = GPT(config.model)

    # create the trainer
    trainer = Trainer(config.trainer, model, tokenized_dataset)

    def batch_end_callback(trainer: Trainer):
        print_every_n = 100
        evaluate_every_n = 10000
        num_ul2_objective_iterations = evaluate_every_n // 100 # 1% of the iterations
        if trainer.iter_num % print_every_n == 0:
            print(f'iteration: {trainer.iter_num}', flush=True)
        
        if trainer.iter_num % evaluate_every_n == 0:
            # Checkpoint the model trained using causal language modeling
            trainer.checkpoint(os.path.join(config.system.work_dir, f'checkpoint_{trainer.iter_num}.pth'))

            # Evaluate the model using the UL2 objectives
            for _ in range(num_ul2_objective_iterations):
                # TODO: train the model using the denoising datasets
                # Train with sequential 50% of the time, extreme 25% of the time, and regular 25% of the time
                pass
            
            # TODO: Save the model checkpoint after training with the UL2 objectives

            # Revert back to training with causal language modeling
            # Load the model checkpoint saved with causal language modeling
            # continue

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()