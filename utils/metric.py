import re
from collections import Counter

# 清理答案字符串的函数（去除标点符号并转换为小写）
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punctuation(text):
        return re.sub(r'[^\w\s]', '', text)

    def lower(text):
        return text.lower()

    return lower(remove_articles(remove_punctuation(s))).strip()

# 计算EM（Exact Match）的函数
def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

# 计算F1 Score的函数
def f1_score(prediction, ground_truth):
    pred_tokens = set(prediction)
    gt_tokens = set(ground_truth)
    
    common = pred_tokens & gt_tokens  # 找到共同的路径
    num_same = len(common)  # 共同路径的数量

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)  # 精确率
    recall = num_same / len(gt_tokens)  # 召回率
    f1 = 2 * (precision * recall) / (precision + recall)  # F1分数

    return f1

