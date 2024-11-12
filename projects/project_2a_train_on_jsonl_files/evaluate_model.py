import argparse
import os
import glob
from lm_eval import evaluator
import torch
import sys
sys.path.append('/home/ansonsav/cs_674/project_2a_minGPT/minGPT')
sys.path.append('/home/ansonsav/cs_674/project_2a_minGPT/minGPT/lm_eval/models')
from min_gpt import MinGPTEval
from jsonl_dataset import JSONLDataset
from gpt_2_tokenized_dataset import GPT2TokenizedDataset
from mingpt.model import GPT
# from min_gpt import MinGPTEval
from transformers import GPT2Tokenizer


from mingpt.utils import set_seed, setup_logging, CfgNode as CN

class DummyInstance:
    def __init__(self, args) -> None:
        self.args = args

def get_config():
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/project_2a_train_on_jsonl_files/mini'

    # data
    C.data = GPT2TokenizedDataset.get_default_config()
    C.data_path = '/home/ansonsav/nobackup/autodelete/pile_data_10.jsonl'

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'



def evaluate_mingpt(model_path, tasks):
    config = get_config()

    setup_logging(config)
    set_seed(config.system.seed)

    # create the data
    dataset = GPT2TokenizedDataset(JSONLDataset(config.data_path, length=10000)) # length=None means use all the data

    config.model.vocab_size = dataset.get_vocab_size()
    config.model.block_size = dataset.get_block_size()

    # create the model
    lm = GPT(config.model)
    checkpoint = torch.load(model_path)
    lm.load_state_dict(checkpoint['model'])
    results = evaluator.simple_evaluate(
        model=MinGPTEval(lm, GPT2Tokenizer.from_pretrained('gpt2')),
        tasks=tasks,
        num_fewshot=0,
        batch_size=32,
        limit=10,
    )
    return results['results']

def evaluate_model(model_path, tasks):
    results = evaluate_mingpt(model_path, tasks)
    print(results)

    for task in tasks:
        print(f"Task: {task}")
        if 'acc,none' in results[task]:
            print(f"Accuracy: {results[task]['acc,none']}")
        else:
            print(f"Accuracy key 'acc,none' not found for task: {task}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate GPT models on specified tasks.')
    parser.add_argument('model_path', type=str, help='Path to the checkpoint file or directory containing checkpoints.')
    args = parser.parse_args()

    tasks = ["boolq", "hellaswag", "anli", "arc_easy", "copa", "rte", "cb"]

    if os.path.isfile(args.model_path):
        # Single file
        print(f"Evaluating model: {args.model_path}")
        evaluate_model(args.model_path, tasks)
    elif os.path.isdir(args.model_path):
        # Directory containing checkpoint files
        checkpoint_files = glob.glob(os.path.join(args.model_path, '*.pth'))
        checkpoint_files.sort()
        for ckpt_file in checkpoint_files:
            print(f"Evaluating model: {ckpt_file}")
            evaluate_model(ckpt_file, tasks)
    else:
        print(f"Error: {args.model_path} is not a valid file or directory.")
