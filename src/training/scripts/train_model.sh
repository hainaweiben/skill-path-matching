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

# 创建更有描述性的输出目录
MODEL_TYPE="gat_focal"  # 模型类型描述
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/train_${MODEL_TYPE}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo "输出将保存到: $OUTPUT_DIR"

# 创建日志文件
LOG_FILE="${OUTPUT_DIR}/training.log"
touch "$LOG_FILE"

# 运行训练脚本并同时输出到终端和日志文件
echo "===== 开始训练模型 ====="
python src/training/train.py \
    --config configs/training_config.yaml \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" 2>&1 | tee "$LOG_FILE"

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo "===== 训练完成 ====="
    echo "模型和日志保存在: $OUTPUT_DIR"
    echo "日志文件: $LOG_FILE"
else
    echo "===== 训练失败 ====="
    echo "查看日志文件了解详情: $LOG_FILE"
fi
