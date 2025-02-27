#!/bin/bash
# 训练模型脚本

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
cd "$PROJECT_ROOT" || exit 1

# 检查CUDA可用性
echo "===== 检查GPU可用性 ====="
if [ -z "$(nvidia-smi 2>/dev/null)" ]; then
    echo "警告: 未检测到NVIDIA GPU，将使用CPU进行训练"
    DEVICE="cpu"
else
    echo "使用设备: cuda"
    DEVICE="cuda"
fi

# 创建输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo "输出将保存到: $OUTPUT_DIR"

# 运行训练脚本
echo "===== 开始训练模型 ====="
python src/training/train.py \
    --config configs/skill_matching_config.yaml \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo "===== 训练完成 ====="
    echo "模型和日志保存在: $OUTPUT_DIR"
else
    echo "===== 训练失败 ====="
fi
