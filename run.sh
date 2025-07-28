#!/bin/bash

# 定义模型名称、语言选项和任务选项
models=("codellama/CodeLlama-13b-Instruct-hf") # 可自行添加更多模型
languages=("c#" "java" "javascript" "php" "python" "typescript" "c" "c++")
tasks=("task1" "task2" "task4")

# 创建结果存储目录
log_dir="./logs"
mkdir -p "$log_dir"

# 遍历所有任务、模型和语言的组合
for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        for language in "${languages[@]}"; do
            safe_model_name=$(echo "$model" | sed 's|/|_|g')
            log_file="$log_dir/${safe_model_name}_${language}_${task}.txt"
            echo "Running model: $model with language: $language and task: $task" | tee -a "$log_file"

            # 执行推理
            CUDA_VISIBLE_DEVICES=0,1 HF_ENDPOINT=https://hf-mirror.com nohup python -u run_llama.py --model_name "$model" --language "$language" --task "$task" > "$log_file" 2>&1
            if [ $? -ne 0 ]; then
                echo "Error encountered while running model: $model with language: $language and task: $task" | tee -a "$log_file"
                echo "Check log file for details: $log_file" | tee -a "$log_file"
                exit 1
            fi

            # 结果文件路径
            result_jsonl="./results/$task/${model//\//-}-$language/${model##*/}.jsonl"

            # 运行评测脚本
            if [ "$task" == "task1" ]; then
                python3 eval_ME.py --input "$result_jsonl" 2>&1 | tee -a "$log_file"
            elif [ "$task" == "task2" ]; then
                python3 eval_DR.py --input "$result_jsonl" 2>&1 | tee -a "$log_file"
            elif [ "$task" == "task4" ]; then
                python3 eval_RC.py --input "$result_jsonl" 2>&1 | tee -a "$log_file"
            fi

            echo "Completed task: $model with language: $language and task: $task (log: $log_file)" | tee -a "$log_file"
        done
    done
done

echo "All tasks and evaluations completed successfully!"