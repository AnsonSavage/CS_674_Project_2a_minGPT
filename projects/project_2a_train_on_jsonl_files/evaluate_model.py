import argparse
import pprint
import os
import re
import glob
import torch
from lm_eval import evaluator
import sys
sys.path.append('/home/ansonsav/cs_674/project_2a_minGPT/minGPT')
sys.path.append('/home/ansonsav/cs_674/project_2a_minGPT/minGPT/lm_eval/models')
from min_gpt import MinGPTEval
from jsonl_dataset import JSONLDataset
from gpt_2_tokenized_dataset import GPT2TokenizedDataset
from mingpt.model import GPT
from transformers import GPT2Tokenizer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

def get_config(model_type):
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
    C.model.model_type = model_type
    return C

def evaluate_mingpt(lm, model_path, tasks):
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

def extract_iteration_from_filename(filename):
    """
    Extracts the iteration number from a single filename if it matches the pattern 'checkpoint_xxxxx.pth'.
    
    Args:
        filename (str): The filename to process.
    
    Returns:
        int: The iteration number if matched, otherwise None.
    """
    # Regex to match 'checkpoint_' followed by digits and ending with '.pth'
    pattern = re.compile(r'checkpoint_(\d+)\.pth')
    match = pattern.match(filename)
    if match:
        # Convert the matched iteration part to an integer
        return int(match.group(1))
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate GPT models on specified tasks.')
    parser.add_argument('model_path', type=str, help='Path to the checkpoint file or directory containing checkpoints.')
    parser.add_argument('--model_type', type=str, default='gpt2', help='Type of the model to evaluate (default: gpt2)')
    args = parser.parse_args()

    tasks = ["boolq", "hellaswag", "anli", "arc_easy", "copa", "rte", "cb"]

    model_type = args.model_type
    config = get_config(model_type)
    setup_logging(config)
    set_seed(config.system.seed)

    # create the data
    dataset = GPT2TokenizedDataset(JSONLDataset(config.data_path, length=10000))

    config.model.vocab_size = dataset.get_vocab_size()
    config.model.block_size = dataset.get_block_size()

    # create the model
    lm = GPT(config.model)

    # For each iteration checkpoint file, we want to evaluate the model and construct a data structure that maps:
    """
    {
        iteration_number: {
            task: {
                'acc,none': accuracy,
                ...
            },
            ...
        },
        ...
    }
    """

    results_per_iteration = {}

    paths = []
    if os.path.isfile(args.model_path):
        # Single file
        paths.append(args.model_path)
    elif os.path.isdir(args.model_path):
        # Directory containing checkpoint files
        checkpoint_files = glob.glob(os.path.join(args.model_path, '*.pth'))
        checkpoint_files.sort()
        paths = checkpoint_files
    else:
        print(f"Error: {args.model_path} is not a valid file or directory.", flush=True)
        sys.exit(1)

    for ckpt_file in paths:
        iteration_number = extract_iteration_from_filename(ckpt_file)
        results_per_iteration[iteration_number] = {}
        print(f"Evaluating model: {ckpt_file}", flush=True)
        results = evaluate_mingpt(lm, ckpt_file, tasks)

        for task in tasks:
            print(f"Task: {task}")
            try:
                print(f"Accuracy: {results[task]['acc,none']}", flush=True)
                results_per_iteration[iteration_number][task] = results[task]
            except KeyError as e:
                print(e, flush=True)

pprint.pprint(results_per_iteration)