import json
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
import re
import argparse

def find_common_prefix(paths):
    """找到路径数组中目录部分的最长公共前缀"""
    if not paths:
        return ""
    dirs = [os.path.dirname(path) for path in paths]
    common_dir_prefix = os.path.commonprefix(dirs)
    return common_dir_prefix

def remove_common_prefix(paths):
    """去除路径中公共的目录部分，只保留文件名部分"""
    common_dir_prefix = find_common_prefix(paths)
    if common_dir_prefix: common_dir_prefix += '/'
    result = []
    for path in paths:
        relative_path = path[len(common_dir_prefix):] if path.startswith(common_dir_prefix) else path
        result.append(relative_path)
    
    return result

def find_common_prefix2(paths):
    """找到路径数组中的最长公共前缀"""
    if not paths:
        return ""
    return os.path.commonprefix(paths)

def remove_common_prefix_across_groups(groups):
    """将所有组的路径作为整体去掉公共前缀，保留路径层级关系"""
    all_paths = [path for group in groups for path in group]
    common_prefix = find_common_prefix2(all_paths)
    updated_paths = [path[len(common_prefix):] for path in all_paths]
    result = []
    idx = 0
    for group in groups:
        group_size = len(group)
        result.append(updated_paths[idx:idx + group_size])
        idx += group_size
    
    return result

# Function to build a directed graph from inverted call chains
def build_graph_inverted(call_chains):
    """
    将调用链（倒序）构建为有向图。支持多个不同排列的调用链结构。

    :param call_chains: 调用链列表，每个调用链是一个列表，表示倒序的文件之间的调用关系
    :return: 返回构建的有向图
    """
    try:
        old_call_chains = call_chains
        pattern = r"\[\[.*?\]\]"
        match = re.search(pattern, call_chains.replace('\n', '').replace(' ',''), re.DOTALL)
        if match:
            call_chains = match[0]
        call_chains = ast.literal_eval(call_chains)
        if isinstance(call_chains, list) and len(call_chains) == 1 and isinstance(call_chains[0], list) and isinstance(call_chains[0][0], list):
            call_chains = call_chains[0]  # 去掉外层一层括号
        
    except Exception as e:
        print("Error: Failed to parse 'call_chains' as a list. Please check the format.")
        print("Exception message:", e)
    
        call_chains = []  # 或者设置为默认值，视需求而
 
      
    G = nx.DiGraph()
    for chain in call_chains:
        # try:
        #   chain = remove_common_prefix(chain) #解决根目录信息不统一的问题
        # except:
            
        #     chain = remove_common_prefix_across_groups(chain)
        # 从链中的最后一个文件往前建立有向边
        for i in range(1, len(chain)):
          try:
              G.add_edge(chain[i], chain[i - 1])
          except:
              print("error",chain)  # c调用了b, b调用了a
    return G


# Function to calculate F1 score
def calculate_f1(precision, recall):
  """
  计算 F1 分数
  :param precision: 精度
  :param recall: 召回率
  :return: F1 分数
  """
  if precision + recall == 0:
    return 0
  return 2 * (precision * recall) / (precision + recall)


# Function to evaluate graph similarity (nodes and edges)
def evaluate_graph_similarity(pred_graph, gt_graph):
  """
  评估两个有向图（预测图和真实图）的相似度

  :param pred_graph: 预测的有向图
  :param gt_graph: 真实的有向图
  :return: 返回节点和边的精确度、召回率和F1分数
  """
  # 获取节点和边的集合
  pred_nodes = set(pred_graph.nodes())
  gt_nodes = set(gt_graph.nodes())

  pred_edges = set(pred_graph.edges())
  gt_edges = set(gt_graph.edges())

  # 计算节点匹配
  true_positive_nodes = len(pred_nodes.intersection(gt_nodes))
  precision_nodes = true_positive_nodes / len(pred_nodes) if pred_nodes else 0
  recall_nodes = true_positive_nodes / len(gt_nodes) if gt_nodes else 0
  f1_nodes = calculate_f1(precision_nodes, recall_nodes)

  # 计算边匹配
  true_positive_edges = len(pred_edges.intersection(gt_edges))
  precision_edges = true_positive_edges / len(pred_edges) if pred_edges else 0
  recall_edges = true_positive_edges / len(gt_edges) if gt_edges else 0
  f1_edges = calculate_f1(precision_edges, recall_edges)

  return {
    "node_precision": precision_nodes,
    "node_recall": recall_nodes,
    "node_f1_score": f1_nodes,
    "edge_precision": precision_edges,
    "edge_recall": recall_edges,
    "edge_f1_score": f1_edges
  }


# Function to process JSONL call_chains
def process_jsonl_data(file_path):
  """
  处理给定的jsonl文件，读取每一行并计算预测图和真实图的相似性

  :param file_path: jsonl文件路径
  :return: DataFrame with evaluation results
  """
  results = []

  with open(file_path, 'r') as file:
    for line in file:
      line = line.strip()  # Remove extra spaces or new lines
      if line:  # Skip empty lines
        try:
            call_chains = json.loads(line)
            pred_chains = call_chains.get('pred', [])
            gt_chains = call_chains.get('gt', [])
            # Build the directed graphs
            pred_graph = build_graph_inverted(pred_chains)
            gt_graph = build_graph_inverted(gt_chains)
         
            # # Evaluate similarity
            # if pred_graph == None:
            #     similarity_results = None
            # else:
            similarity_results = evaluate_graph_similarity(pred_graph, gt_graph)

            results.append({
                "idx": call_chains.get("idx", "N/A"),
                "evaluation": similarity_results
            })
        except json.JSONDecodeError:
          print(f"Skipping invalid JSON entry: {line}")

  return pd.DataFrame(results)




def evaluate(jsonl_file_path, output_csv_path=None):
    """
    Evaluate JSONL data to compute average and combined F1 scores.

    Parameters:
        jsonl_file_path (str): Path to the JSONL file containing evaluation data.
        output_csv_path (str, optional): Path to save the DataFrame as a CSV file. Default is None.

    Returns:
        dict: A dictionary containing the average and combined F1 scores.
    """
    # Process JSONL data
    df_jsonl_results = process_jsonl_data(jsonl_file_path)
    
    # Convert to DataFrame
    df = pd.DataFrame(df_jsonl_results)
    
    # Initialize total scores
    total_node_f1_score = 0
    total_edge_f1_score = 0
    num_entries = len(df)
    count = 0
    # for evaluation in df['evaluation']:
    #     if evaluation != None:
    #         count += 1
    # return count / num_entries
    #Accumulate F1 scores
    for evaluation in df['evaluation']:
        total_node_f1_score += evaluation["node_f1_score"]
        total_edge_f1_score += evaluation["edge_f1_score"]
    
    # Calculate average F1 scores
    average_f1_scores = {
        "node_f1_score": total_node_f1_score / num_entries,
        "edge_f1_score": total_edge_f1_score / num_entries
    }
    
    # Calculate combined F1 score
    combined_f1_score = 0.15*average_f1_scores["node_f1_score"] + 0.85*average_f1_scores["edge_f1_score"]
    
    # Print results
    print("Average F1 Scores:", average_f1_scores)
    print("Combined F1 Score:", combined_f1_score)
    
    # Optionally save results to CSV
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
        print(f"Results saved to {output_csv_path}")
    return combined_f1_score
 
# Example usage
# jsonl_file_path = '/root/autodl-tmp/Codebench/results/task4/Qwen/Qwen2.5-Coder-3B-c++/Qwen2.5-Coder-3B.jsonl'
# output_csv_path = 'jsonl_evaluation_results_v2.csv'

# evaluate(jsonl_file_path)


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
        print(jsonl_path)
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

# directory_path = '/home/test/test/rag/Codebench_exp/results/task4'
# name = directory_path.split('/')[-1]
# output_csv_path = directory_path  + "_result.csv"
# process_all_files_in_directory(directory_path, output_csv_path)

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
