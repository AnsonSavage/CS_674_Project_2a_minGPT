
from lm_eval import evaluator
from mingpt.model import GPT

def evaluate_mingpt_hellaswag(model_path):
    lm = GPT(model_path)
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=["hellaswag"],
        num_fewshot=0,
        batch_size=32,
        limit=10,
    )
    return results['results']['hellaswag']['acc,none']

score = evaluate_mingpt_hellaswag("output/checkpoint_1000.pth")
print(score)