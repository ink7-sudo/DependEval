from run import main
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse

parser = argparse.ArgumentParser(description="Run model evaluation with specified parameters.")
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--language", type=str, required=True)
parser.add_argument("--task", type=str, default="task1")
parser.add_argument("--dataset_path", type=str, default="./data")
parser.add_argument("--res_dir", type=str, default="./results")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--max_token_nums", type=int, default=40000)
args = parser.parse_args()
cache_dir = "./huggingface"


model = AutoModelForCausalLM.from_pretrained(
args.model_name,
torch_dtype="auto",
device_map="auto",
cache_dir=cache_dir
)
tokenizer = AutoTokenizer.from_pretrained(args.model_name,cache_dir=cache_dir)

def inference_func(prompt,top_p,temperature):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=8094,
        top_p = top_p,
        temperature=temperature
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response



if __name__ == "__main__":
    eval_path = main(
        model_name=args.model_name,
        language=args.language,
        task=args.task,
        dataset_path=args.dataset_path,
        res_dir=args.res_dir,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_token_nums=args.max_token_nums,
        inference_func=inference_func
    )
    #evaluate(eval_path)

