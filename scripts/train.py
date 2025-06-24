from datasets import load_dataset
from datasets import Dataset
import json
from math_verify import parse, verify
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM 
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model')
parser.add_argument('--save_path', type=str, required=True, help='Path to save the trained model')
# learning rate parameter
parser.add_argument('--lr', type=float, default=3.0e-6, help='learning rate')
# num_generations
parser.add_argument('--num_generations', type=int, default=6, help='G in grpo')
args = parser.parse_args()

ckpt_path = os.path.join(args.save_path, "ckpt")
save_path = os.path.join(args.save_path, "model")
os.makedirs(ckpt_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

dataset_path = '../dataset'
LC_extractor_path = '../extractor_model'
if not os.path.exists(dataset_path):
    raise "Dataset is not exists."

train_dataset = load_dataset(dataset_path, split='train')

if not os.path.exists(LC_extractor_path):
    raise "LC-extractor is not exists."

train_dataset = train_dataset.rename_column("problem", "prompt")
train_dataset = train_dataset.map(lambda x: {"prompt": [{'role': 'user', 'content': x["prompt"] + " Please reason step by step, and put your final answer within \\boxed{}."}]})

def extract(answer:str):
    cnt = 0
    for i in range(len(answer)):
        if answer[i] == '{': cnt += 1
        if answer[i] == '}': cnt -= 1
        if cnt == 0:
            answer = answer[:i+1]
            return answer

def accuracy_reward(prompts:str, completions:str, **kwargs):
    solutions = kwargs['solution']
    results = []
    for i in range(len(prompts)):
        answer = extract(completions[i][0]['content'].rsplit('\\boxed', 1)[-1])
        if answer == solutions[i]:
            results.append(1)
            continue
        try:
            is_true = verify(parse('\\boxed' + answer), parse('\\boxed{' + solutions[i] + '}'))
            results.append(1 if is_true else 0) 
        except: results.append(0)
    return results

def format_reward(prompts:str, completions:str, **kwargs):
    results = []
    for completion in completions:
        if completion[0]['content'].count('</think>') == 1:
            results.append(1)
        else:
            results.append(0)
    return results

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir=ckpt_path,
    learning_rate=args.lr,
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    bf16=True,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.6,
    max_completion_length=8192, 
    num_generations=args.num_generations, 
    report_to=["tensorboard"],
    logging_steps=3,
    per_device_train_batch_size=4,
    gradient_checkpointing=True, 
    save_steps = 0.2,
    save_strategy='steps',
    scale_rewards=False,
    max_grad_norm=0.1,
    shuffle_dataset=False,
    small_model_path = LC_extractor_path
)

model = AutoModelForCausalLM.from_pretrained(args.model_path)

trainer = GRPOTrainer(
    model, 
    [format_reward, accuracy_reward], 
    args = training_args, 
    train_dataset=train_dataset,
    len_penalty_coef= 1.0,
    has_small_model = True,
    add_adv = 1)

trainer.train()

trainer.save_model(save_path)
    