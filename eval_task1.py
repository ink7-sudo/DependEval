import json
import demjson3
from openai import OpenAI
import re
import argparse
import os
llm_judge_prompt = '''

Gt: {gt}
Pred: {pred}

Using Gt as the correct answer, compare the content of Pred with Gt and evaluate Pred based on the following aspects. Each aspect contains tailored evaluation criteria to handle the complexities of multi-file interactions and feature integration. The output must follow the JSON format described in Point 6.

Evaluation Aspects

1. Correctness of Function Calls 
Objective: Evaluate the accuracy of all function calls between segments and across files.
	•	Ensure:
	•	Each invoking_code_segment correctly calls its corresponding called_code_segment as per Gt.
	•	Calls include appropriate parameter matching, order, and context alignment.
	•	Evaluation Criteria:
	•	Does the function signature match, including parameter names, types, and order?
	•	Are correct arguments passed, meeting expectations in feature_description and detailed_feature_description?
	•	Is the pre- or post-logic necessary for context included?
	•	Are cross-file dependencies invoked correctly, as shown in modified_complete_code?

Scoring Rules:
	•	5 points: All function calls are completely correct and match Gt, including parameters, order, and logical dependencies.
	•	4 points: Mostly correct with minor parameter or comment issues but no major gaps.
	•	3 points: Partially correct; missing key parameters, logic, or dependencies.
	•	2 points: Significant issues in invocation logic, causing likely runtime errors.
	•	0-1 points: Calls are incorrect, incomplete, or not implemented.

2. Alignment with Feature Requirements
Objective: Check if the code in Pred aligns with the intended feature and modification goals.
	•	Ensure:
	•	Every call reflects requirements in feature_description and detailed_feature_description.
	•	The new or modified logic directly implements the required functionality.
	•	Evaluation Criteria:
	•	Does the logic adhere to the functional goals described?
	•	Does it integrate with multi-file dependencies correctly (if applicable)?
	•	Are the new components in new_file_code_segment aligned with expectations?

Scoring Rules:
	•	5 points: Perfectly aligned with feature requirements; implementation is logically complete.
	•	4 points: Correctly aligned but with potential optimizations or minor improvements.
	•	3 points: Partially fulfills requirements with clear gaps in alignment.
	•	2 points: Loosely aligned with significant logic missing.
	•	0-1 points: Not aligned or entirely unrelated to the described requirements.

3. Accuracy of Functionality Implementation
Objective: Verify the correctness of the implementation, focusing on functional outcomes.
	•	Evaluation Criteria:
	•	Does the functionality fully satisfy the requirements in feature_description?
	•	Are components correctly loaded, initialized, or referenced?
	•	Are all dependencies resolved for seamless multi-file integration?

Scoring Rules:
	•	5 points: Fully accurate implementation without functional defects.
	•	4 points: Mostly accurate with minor issues or deviations.
	•	3 points: Partially correct but lacking essential steps or logic.
	•	2 points: Basic framework present but largely incomplete.
	•	0-1 points: Non-functional due to missing or incorrect logic.

4. Completeness of Implementation
Objective: Ensure that all functional components, including new and modified ones, are fully implemented.
	•	Evaluation Criteria:
	•	Are all required segments across files defined and updated per Gt?
	•	Does the implementation cover all subparts described in detailed_feature_description?
	•	Are all new dependencies (#New segments) and modifications (#Modify segments) accounted for?

Scoring Rules:
	•	5 points: Complete implementation with no omissions.
	•	4 points: Nearly complete, with only minor omissions.
	•	3 points: Significant missing functionality, but partially meets requirements.
	•	2 points: Too many missing components, achieving minimal functionality.
	•	0-1 points: Nearly all components are missing or incorrect.

5. Code Quality
Objective: Assess the overall quality, maintainability, and readability of the code.
	•	Evaluation Criteria:
	•	Readability: Clear naming, concise comments, and consistent style.
	•	Maintainability: Modular structure, minimal duplication, and extensibility.
	•	Efficiency: Appropriate algorithms, data structures, and resource use.

Scoring Rules:
	•	5 points: Excellent quality with clean, efficient, and maintainable code.
	•	4 points: Good quality, but minor readability or efficiency issues.
	•	3 points: Average quality; readable but not optimized or modular.
	•	2 points: Poor quality; lacks structure or suffers from inefficiencies.
	•	0-1 points: Unreadable, unstructured, or inefficient code.

'''

json_format= '''
6. JSON Output Format

The output JSON should include the following fields:
	•	correctness_score: Score for correctness of function calls.
	•	purpose_alignment_score: Score for alignment with the purpose of the call.
	•	functionality_accuracy_score: Score for accuracy of functionality implementation.
	•	functionality_completeness_score: Score for completeness of functionality implementation.
	•	code_quality_score: Score for overall code quality.
Example Output:
{
  "correctness_score": 5,
  "purpose_alignment_score": 4,
  "functionality_accuracy_score": 5,
  "functionality_completeness_score": 4,
  "code_quality_score": 5
}
'''

MAX_RETRY = 3

def get_deepseek_response(model_name,prompt,top_p,temperature):
    # client = OpenAI(api_key="sk-0dd626b7739c44dbb3ddf7cc5fd358d6", base_url="https://api.deepseek.com/v1")
    # response = client.chat.completions.create(
    #     model=model_name,
    #     messages=[
    #         {"role": "system", "content": prompt},
    #     ],
    #     stream=False,
    #     top_p = top_p,
    #     temperature=temperature
    # )
    # return response.choices[0].message.content
     # Initialize client
    client = OpenAI(api_key="sk-b9e51e93f33640909bd7b6b1106d2032", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    # Call  API
    response = client.chat.completions.create(
        model="qwen-plus",
        temperature = temperature,
        messages=[
            {"role": "system", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content

def get_tencent_response(model_name,prompt,top_p,temperature):
    client = OpenAI(api_key="sk-SrxGmOVTzA2zt7gAGhuALxmsHpKKPLCzSbwnP24lBVun4Wqp", base_url="https://api.hunyuan.cloud.tencent.com/v1")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=False,
        top_p = top_p,
        temperature=temperature
    ) 
    return response.choices[0].message.content

def LLM_judge(pred, gt,llm_function,model_name,top_p,temperature):
    json_pattern = r"\{[\s\S]*?\}"
    for attempt in range(MAX_RETRY):
        try:
            # 使用模型生成初始响应
            prompt = llm_judge_prompt.format(pred=pred, gt=gt) + json_format
            result = llm_function(model_name, prompt=prompt, top_p=0.95, temperature=0.2)
            match = re.search(json_pattern, result)
# 提取 JSON 部分
            if match:
                response = match.group(0)
            # 尝试解析响应为 JSON
            parsed_response = json.loads(response)
            return parsed_response  # 返回解析后的 JSON 对象

        except json.JSONDecodeError:
            # 尝试使用修复工具修复
            print(f"Attempt {attempt + 1}: Response is not valid JSON. Trying to fix...")
            try:
                fixed_response = demjson3.decode(response)  # 自定义函数，尝试修复 JSON
                parsed_response = json.loads(fixed_response)
                return parsed_response
            except Exception as e:
                print(f"Fix attempt failed: {e}")
    # 如果无法修复，则返回错误提示或继续其他逻辑
    return None

weights = {
    "correctness_score": 0.25,
    "purpose_alignment_score": 0.25,
    "functionality_accuracy_score": 0.20,
    "functionality_completeness_score": 0.20,
    "code_quality_score": 0.10
}




def process_jsonl_file(filepath, weights):
    total_score = 0
    updated_lines = []

    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                if not line.strip():  # 检查是否为空行
                    continue

                data = json.loads(line.strip())

                # 提取并解析 pred
                match = re.search(r'\{.*\}', data.get("pred", "{}"), re.DOTALL)
                if match:
                    pred = match.group(0)
                    pred = json.loads(pred)  # 解析为 Python 字典
                else:
                    pred = {}
                gt = data.get("gt", {})
                # 调用评分函数
                result = LLM_judge(pred, gt, get_deepseek_response, "deepseek-chat", 0.2, 0.7)
                print(result)
                if result is not None:
                    total_weighted_score = sum((result[key] / 5) * weight for key, weight in weights.items())
                    scaled_score = total_weighted_score * 100
                    total_score += scaled_score
                    data["score"] = scaled_score
                else:
                    data["score"] = 0

            except json.JSONDecodeError as e:
                print(f"JSON 解析错误: {e} | 文件: {filepath}")
                data = {"score": 0.01}  # 标记是否格式遵循
            except Exception as e:
                print(f"处理数据时发生错误: {e} | 文件: {filepath}")
                data = {"score": 0}  # 确保 data 被初始化

            # 记录更新后的数据
            updated_lines.append(json.dumps(data, ensure_ascii=False))

    # 将更新后的内容写回文件
    output_filepath = filepath.replace(".jsonl", "_scored.jsonl")
    with open(output_filepath, 'w', encoding='utf-8') as file:
        file.write("\n".join(updated_lines) + "\n")

    return total_score


# 主程序入口
def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Process JSONL files and compute scores.")
    parser.add_argument("--input", required=True, help="Path to the input JSONL file.")
    
    args = parser.parse_args()

    # 加载权重
    weights = {
        "correctness_score": 0.25,
        "purpose_alignment_score": 0.25,
        "functionality_accuracy_score": 0.20,
        "functionality_completeness_score": 0.20,
        "code_quality_score": 0.10
    }


    # 调用处理函数
    total_score = process_jsonl_file(args.input, weights)
    print(f"处理完成: 文件 {args.input} 的总得分为 {total_score:.2f}")


if __name__ == "__main__":
    main()