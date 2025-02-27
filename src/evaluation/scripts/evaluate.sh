#!/bin/bash
# 评估模型脚本

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
cd "$PROJECT_ROOT" || exit 1

# 检查CUDA可用性
echo "===== 检查GPU可用性 ====="
if [ -z "$(nvidia-smi 2>/dev/null)" ]; then
    echo "警告: 未检测到NVIDIA GPU，将使用CPU进行评估"
else
    echo "使用设备: GPU"
fi

# 查找最新的模型输出目录
LATEST_OUTPUT=$(ls -td outputs/2025*/ | head -1)
if [ -z "$LATEST_OUTPUT" ]; then
    echo "错误: 未找到模型输出目录"
    exit 1
fi

echo "找到最新输出目录: $LATEST_OUTPUT"

# 查找最新的训练运行目录
LATEST_RUN=$(find "$LATEST_OUTPUT" -name "run_*" -type d | head -1)
if [ -z "$LATEST_RUN" ]; then
    echo "错误: 未找到训练运行目录"
    exit 1
fi

echo "找到最新训练运行目录: $LATEST_RUN"

# 查找模型文件
MODEL_PATH="${LATEST_RUN}/best_model.pth"
if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 未找到模型文件 $MODEL_PATH"
    exit 1
fi

# 查找配置文件
CONFIG_PATH="${LATEST_RUN}/config.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 未找到配置文件 $CONFIG_PATH"
    exit 1
fi

# 创建评估输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_OUTPUT="outputs/evaluation_${TIMESTAMP}"
mkdir -p "$EVAL_OUTPUT"

echo "===== 开始评估模型 ====="
echo "模型路径: $MODEL_PATH"
echo "配置文件: $CONFIG_PATH"
echo "输出目录: $EVAL_OUTPUT"

# 运行评估脚本
python src/evaluation/evaluate.py \
    --model_path "$MODEL_PATH" \
    --config_path "$CONFIG_PATH" \
    --output_dir "$EVAL_OUTPUT" \
    --threshold 0.5

# 检查评估是否成功
if [ $? -eq 0 ]; then
    echo "===== 评估完成 ====="
    echo "评估结果保存在: $EVAL_OUTPUT"
else
    echo "===== 评估失败 ====="
fi
