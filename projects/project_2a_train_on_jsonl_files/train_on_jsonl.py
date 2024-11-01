from jsonl_dataset import JSONLDataset
from gpt_2_tokenized_dataset import GPT2TokenizedDataset

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
    C.system.work_dir = './out/project_2a_train_on_jsonl_files'

    # data
    C.data = GPT2TokenizedDataset.get_default_config()
    C.data_path = '/home/ansonsav/nobackup/autodelete/pile_data_10.jsonl'

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-nano' # This is really small just for initial testing purposes :)

    # trainer
    C.trainer = Trainer.get_default_config()
    # Can overwrite things like learning rate here
    C.trainer.learning_rate = 5e-4

    return C

if __name__ == '__main__':
    # get the config
    config = get_config()

    setup_logging(config)
    set_seed(3407)




    # create the data
    dataset = GPT2TokenizedDataset(JSONLDataset(config.data))

    config.model.vocab_size = dataset.get_vocab_size()
    config.model.block_size = dataset.get_block_size()

    # create the model
    model = GPT(config.model)

    # create the trainer
    trainer = Trainer(config.trainer, model, dataset)
    trainer.add_callback_every_n_batches(100, lambda trainer: print(trainer.iter_num))
    # Save the model every 1000 iterations
    trainer.add_callback_every_n_batches(1000, trainer.checkpoint('model.pth'))


    # train
    trainer.train()