#!/bin/bash

# 定义模型名称、语言选项和任务选项
models=("codellama/CodeLlama-13b-Instruct-hf")
languages=("c#" "java" "javascript" "php" "python" "typescript")
tasks=("task1")

# 创建结果存储目录
log_dir="./logs"
mkdir -p "$log_dir"

# 遍历所有任务、模型和语言的组合
for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        for language in "${languages[@]}"; do
            safe_model_name=$(echo "$model" | sed 's|/|_|g')
            log_file="$log_dir/${safe_model_name}_${language}_${task}.txt"
            echo "Running model: $model with language: $language and task: $task"
            
            # 执行命令并实时输出到日志和控制台
            CUDA_VISIBLE_DEVICES=0,1 HF_ENDPOINT=https://hf-mirror.com nohup python -u run_llama.py --model_name "$model" --language "$language" --task "$task" > "$log_file" 2>&1
            
            if [ $? -ne 0 ]; then
                echo "Error encountered while running model: $model with language: $language and task: $task"
                echo "Check log file for details: $log_file"
                exit 1
            fi

            echo "Completed task: $model with language: $language and task: $task (log: $log_file)"
        done
    done
done

echo "All tasks completed successfully!"