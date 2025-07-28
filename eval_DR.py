from utils.metric import exact_match_score, f1_score
import json
import re
import os
import pandas as pd
import argparse

def task2_match(text: str):
    match = re.search(r"\[.*?\]", str(text))
    if match:
        array_str = match.group()
        # print(array_str)  # 输出: ['Coeus/scanner/dynamic.py', 'Coeus/utils/manager.py', 'Coeus/coeus.py']
        return array_str
    else:
        # print("未找到数组部分")
        return ""

def process_task2_with_idx(jsonl_path: str):
    """
    从 JSONL 文件中解析预测值、真实值和索引。

    Parameters:
        jsonl_path (str): JSONL 文件路径。

    Returns:
        tuple: 包含预测值列表、真实值列表和索引列表。
    """
    datas = []
    preds = []
    gts = []
    indices = []  # 用于存储 idx
    
    # 读取 JSONL 文件
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            datas.append(data)
            preds.append(data["pred"])
            gts.append(data["gt"])
            indices.append(data["idx"])  # 提取并存储 idx
    
    new_preds = []
    new_gts = []
    new_indices = []  # 用于存储有效的 idx
    
    # 对 pred 进行匹配和过滤
    for idx, pred, gt in zip(indices, preds, gts):
        new_pred = None
        new_pred = task2_match(pred)
        if new_pred != '':  # 如果匹配结果非空
            try:
                new_pred = eval(new_pred)  # 将字符串解析为 Python 对象
                new_preds.append(new_pred)
                new_gts.append(gt)
                new_indices.append(idx)  # 保留有效的 idx
            except Exception as e:
                print(f"Error processing idx {idx}: {e}")
                new_preds.append(None)
        else:
            new_pred = None
                
    
    return new_preds, new_gts, new_indices


def evaluate(jsonl_path):
    # 解析 JSONL 文件，获取预测值和真实值
    preds, gts, indices = process_task2_with_idx(jsonl_path=jsonl_path)
    
    EM_avg = []
    total = 0
    count = 0
    # for idx, pred, gt in zip(indices, preds, gts):
    #     if pred != None:
    #         print(pred)
    #         count+=1
    # return count / len(gts)


    for idx, pred, gt in zip(indices, preds, gts):
        em_total = 0
        try:
            for p, g in zip(pred, gt):
                em_total += exact_match_score(p, g)
            if len(pred) != 0:
                em_avg = em_total / len(pred)
            else:
                em_avg = 0.0
            if em_avg == 1:
                total += em_avg
                EM_avg.append(em_avg)
        except Exception as e:
            print(f"Error processing idx {idx}: {e}")
            pass    
        # 打印当前 idx 和 em_avg
        print(f"idx: {idx}, em_avg: {em_avg:.4f}")
    # 计算总的平均值
    total_avg = total / len(preds)
    print(total)
    print(f"{jsonl_path} total average EM score:", total_avg)
    
    return total_avg

    
#score = evaluate("/root/autodl-tmp/Codebench/results/task2/Qwen/Qwen2.5-Coder-32B-Instruct-c++/Qwen2.5-Coder-32B-Instruct.jsonl")


def process_all_files_in_directory(directory_path, output_csv_path):
    # 获取目录中的所有 JSON 文件（递归）
    json_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:

            if file.endswith('.jsonl'):
                json_files.append(os.path.join(root, file))
    
    # 提取模型名称和语言
    model_language_scores = {}

    for jsonl_path in json_files:
        # 提取模型名称和语言
        print(jsonl_path)
        parts = jsonl_path.split('/')
        parent_dir = parts[-2]
        print(parent_dir)
        name = parent_dir.split('-')
        language = name[-1] # 假设模型名称在文件路径的倒数第二部分
        model_name = '-'.join(name[:-1])  # 假设语言在倒数第三部分
        
        # 计算 EM 平均分
        score = evaluate(jsonl_path)
        
        # 如果模型没有记录过，初始化一个空的字典
        if model_name not in model_language_scores:
            model_language_scores[model_name] = {}
        
        # 保存模型与语言对应的分数
        model_language_scores[model_name][language] = score
    
    # 创建一个 DataFrame 来存储结果
    print("model_language_scores:", model_language_scores)
    df = pd.DataFrame.from_dict(model_language_scores, orient='index')
    
    # 将 DataFrame 存储为 CSV 文件
    df.to_csv(output_csv_path)
    print(f"Results saved to {output_csv_path}")

# 使用示例

def main():
    parser = argparse.ArgumentParser(description="Process JSONL files and compute scores.")
    parser.add_argument("--input_dir", required=True, help="Directory containing JSONL files.")
    args = parser.parse_args()

    directory_path = args.input_dir
    name = directory_path.rstrip('/').split('/')[-1]
    output_csv_path = directory_path + "_result.csv"
    process_all_files_in_directory(directory_path, output_csv_path)

if __name__ == "__main__":
    main()
