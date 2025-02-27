#!/bin/bash
# 评估模型脚本

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
cd "$PROJECT_ROOT" || exit 1

# 检查CUDA可用性
echo "===== 检查GPU可用性 ====="
if [ -z "$(nvidia-smi 2>/dev/null)" ]; then
    echo "警告: 未检测到NVIDIA GPU，将使用CPU进行评估"
    DEVICE="cpu"
else
    echo "使用设备: cuda"
    DEVICE="cuda"
fi

# 查找最新的模型目录
LATEST_MODEL_DIR=$(find outputs -maxdepth 1 -type d -name "train_*" | sort -r | head -n 1)

if [ -z "$LATEST_MODEL_DIR" ]; then
    echo "错误: 未找到训练好的模型目录"
    exit 1
fi

echo "使用最新的模型目录: $LATEST_MODEL_DIR"

# 查找模型文件
MODEL_FILE=$(find "$LATEST_MODEL_DIR" -name "best_model.pth")

if [ -z "$MODEL_FILE" ]; then
    echo "错误: 在 $LATEST_MODEL_DIR 中未找到模型文件"
    exit 1
fi

echo "使用模型文件: $MODEL_FILE"

# 创建评估输出目录
MODEL_TYPE=$(basename "$LATEST_MODEL_DIR" | cut -d'_' -f2)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/eval_${MODEL_TYPE}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo "评估结果将保存到: $OUTPUT_DIR"

# 创建日志文件
LOG_FILE="${OUTPUT_DIR}/evaluation.log"
touch "$LOG_FILE"

# 运行评估脚本
echo "===== 开始评估模型 ====="
python src/evaluation/evaluate.py \
    --config configs/training_config.yaml \
    --model_path "$MODEL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" 2>&1 | tee "$LOG_FILE"

# 检查评估是否成功
if [ $? -eq 0 ]; then
    echo "===== 评估完成 ====="
    echo "评估结果保存在: $OUTPUT_DIR"
    echo "日志文件: $LOG_FILE"
else
    echo "===== 评估失败 ====="
    echo "查看日志文件了解详情: $LOG_FILE"
fi
