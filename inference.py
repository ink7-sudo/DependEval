import os
import fire
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import DatasetDict, Dataset
import pandas as pd
from data.utils import construct_prompt
from openai import OpenAI
from huggingface_hub import InferenceClient
from typing import Callable  # 注意这里要导入 typing.Callable
from vllm import LLM, SamplingParams
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM


def main(
    model_name: str,
    language: str,
    task: str = "task1",
    dataset_path: str = "./data",
    res_dir: str = "./results",
    batch_size: int = 1,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_token_nums: int = 40000,
    inference_func: Callable[[str, float, float], str] = None
):


    # Load the dataset
    if task == "task1":
        path = os.path.join(dataset_path, language, f"{task}_{language}.json")
    elif task == "task2":
        path = os.path.join(dataset_path, language, f"{task}_{language}_final.json")
    elif task == "task4":
        path = os.path.join(dataset_path, language, f"{task}_{language}_new.json")
    else:
        raise ValueError(f"Unknown task: {task}")

    with open(path, 'r') as f:
        dataset = json.load(f)

    #dataset = load_dataset("json", data_files=path,split="train")
    # Create the save directory
    save_dir = f"{res_dir}/{task}/{model_name}-{language}"
    os.makedirs(save_dir, exist_ok=True)
    evalpath = ''
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Processing data in batches"):
        batch_data = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        try:
            batch_prompts = [construct_prompt(d, max_token_nums=max_token_nums, language=language, task=task) for d in batch_data]
        except Exception as e:
            print(f"Error constructing prompts: {e}")
            continue
        for j, prompt in enumerate(batch_prompts):
            try:
            # print(prompt)
                result = ""
                result = inference_func(prompt,top_p,temperature)
                name = model_name.split('/')[-1]
                evalpath = f"{save_dir}/{name}.jsonl"
                with open(evalpath, "a") as f_out:
                    if task == "task1":
                        f_out.write(json.dumps({"idx": i + j,  "pred": result, "gt": batch_data[j]["modified_complete_code"]}) + "\n")
                    else:
                        f_out.write(json.dumps({"idx": i + j,  "pred": result, "gt": batch_data[j]["gt"]}) + "\n")
            except Exception as e:
                print(f"Error: {e}")
    return evalpath


if __name__ == "__main__":
    fire.Fire(main)
#     main(
#         model_name="hunyuan-large",
#         top_p=0.85,
#         temperature=0.6,
#         inference_func=get_tencent_response  # 函数作为参数传入
#     )