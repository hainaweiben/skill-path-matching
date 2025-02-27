#!/bin/bash
# 训练技能匹配模型的脚本

# 设置工作目录
cd /home/u2021201733/test/skill_path_matching

# 检查GPU可用性
echo "===== 检查GPU可用性 ====="
python -c "import torch; print('使用设备:', 'cuda' if torch.cuda.is_available() else 'cpu')"

# 创建输出目录
OUTPUT_DIR="outputs/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR
echo "输出将保存到: $OUTPUT_DIR"

# 开始训练
echo "===== 开始训练模型 ====="
python src/training/train.py --config configs/training_config.yaml --output_dir $OUTPUT_DIR

# 训练完成
echo "===== 训练完成 ====="
echo "模型和日志保存在: $OUTPUT_DIR"
