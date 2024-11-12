from jsonl_dataset import JSONLDataset
from gpt_2_tokenized_dataset import GPT2TokenizedDataset
import os
import matplotlib.pyplot as plt


import sys
sys.path.append('/home/ansonsav/cs_674/project_2a_minGPT/minGPT')

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

def get_config():
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/project_2a_train_on_jsonl_files/iteration_4_full'

    # data
    C.data = GPT2TokenizedDataset.get_default_config()
    C.data_path = '/home/ansonsav/nobackup/autodelete/pile_data_10.jsonl'

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

    # create the data
    dataset = GPT2TokenizedDataset(JSONLDataset(config.data_path, length=None)) # length=None means use all the data

    config.model.vocab_size = dataset.get_vocab_size()
    config.model.block_size = dataset.get_block_size()

    # create the model
    model = GPT(config.model)

    # create the trainer
    trainer = Trainer(config.trainer, model, dataset)

    def batch_end_callback(trainer: Trainer):
        print_every_n = 100
        save_checkpoint_every_n = 10000
        evaluate_every_n = 10000
        if trainer.iter_num % print_every_n == 0:
            print(f'iteration: {trainer.iter_num}', flush=True)
        
        if trainer.iter_num % save_checkpoint_every_n == 0:
            trainer.checkpoint(os.path.join(config.system.work_dir, f'checkpoint_{trainer.iter_num}.pth'))
        
        if trainer.iter_num % evaluate_every_n == 0:
            # Evaluate the model
            print(trainer.loss_history[-1], flush=True)
            
            try:
                # Create a graph of the loss history
                plt.plot(trainer.loss_history)
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Loss history')
                plt.savefig(os.path.join(config.system.work_dir, f'loss_history_iteration_{trainer.iter_num}.png'))
                plt.clf()
            except Exception as e:
                print(f'Error creating loss history graph: {e}')



    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()