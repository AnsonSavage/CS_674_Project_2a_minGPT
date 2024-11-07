from lm_eval import evaluator
import torch
import sys
sys.path.append('/home/ansonsav/cs_674/project_2a_minGPT/minGPT')
sys.path.append('/home/ansonsav/cs_674/project_2a_minGPT/minGPT/lm_eval/models')
from min_gpt import MinGPTEval
from jsonl_dataset import JSONLDataset
from gpt_2_tokenized_dataset import GPT2TokenizedDataset
from mingpt.model import GPT
from mingpt.trainer import Trainer
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
    # C.model.model_type = 'gpt2'
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    # Can overwrite things like learning rate here
    C.trainer.learning_rate = 5e-4
    C.trainer.batch_size = 1
    C.trainer.num_workers = 1

    return C

# if __name__ == '__main__':
#     # get the config
#     config = get_config()

#     setup_logging(config)
#     set_seed(config.system.seed)

#     # create the data
#     dataset = GPT2TokenizedDataset(JSONLDataset(config.data_path, length=10000)) # length=None means use all the data

#     config.model.vocab_size = dataset.get_vocab_size()
#     config.model.block_size = dataset.get_block_size()

#     # create the model
#     model = GPT(config.model)
#     checkpoint = torch.load("/home/ansonsav/cs_674/project_2a_minGPT/minGPT/projects/project_2a_train_on_jsonl_files/out/project_2a_train_on_jsonl_files/iteration_3/checkpoint_1960000.pth")
#     model.load_state_dict(checkpoint['model'])
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     # input_tokens = torch.tensor(tokenizer("The quick brown fox", truncation=True, max_length=1024)['input_ids'], dtype=torch.long).unsqueeze(0)
#     # output_tokens = model.generate(input_tokens, 100)
#     # output_text = tokenizer.decode(output_tokens[0])
#     # print(output_text)
#     tester = MinGPTEval(model, tokenizer)
#     results = tester.generate_until([DummyInstance(("The quick brown fox", {"until": [".", "\n\n"], "max_gen_toks": 200}))])
#     print(results)



def evaluate_mingpt_hellaswag(model_path):
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
        tasks=["hellaswag"],
        num_fewshot=0,
        batch_size=32,
        limit=10,
    )
    return results['results']['hellaswag']['acc,none']

score = evaluate_mingpt_hellaswag("/home/ansonsav/cs_674/project_2a_minGPT/minGPT/projects/project_2a_train_on_jsonl_files/out/project_2a_train_on_jsonl_files/iteration_3/checkpoint_1960000.pth")
print(score)

