#!/bin/bash
# 项目清理脚本

# 设置项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT" || exit 1

echo "===== 开始清理项目 ====="

# 清理Python缓存文件
echo "清理Python缓存文件..."
find "$PROJECT_ROOT" -name "__pycache__" -type d -exec rm -rf {} +
find "$PROJECT_ROOT" -name "*.pyc" -o -name "*.pyo" -delete

# 清理临时文件
echo "清理临时文件..."
find "$PROJECT_ROOT" -name "*.bak" -o -name "*.tmp" -o -name "*.old" -delete

# 清理旧的输出目录
echo "清理旧的输出目录..."
# 保留最新的训练和评估结果
LATEST_TRAIN_DIR=$(find "$PROJECT_ROOT/outputs" -maxdepth 1 -type d -name "train_*" | sort -r | head -n 1)
LATEST_EVAL_DIR=$(find "$PROJECT_ROOT/outputs" -maxdepth 1 -type d -name "eval_*" | sort -r | head -n 1)

if [ -n "$LATEST_TRAIN_DIR" ]; then
    echo "保留最新的训练目录: $(basename "$LATEST_TRAIN_DIR")"
fi

if [ -n "$LATEST_EVAL_DIR" ]; then
    echo "保留最新的评估目录: $(basename "$LATEST_EVAL_DIR")"
fi

# 删除其他旧的训练和评估目录
find "$PROJECT_ROOT/outputs" -maxdepth 1 -type d -name "train_*" | grep -v "$LATEST_TRAIN_DIR" | xargs rm -rf
find "$PROJECT_ROOT/outputs" -maxdepth 1 -type d -name "eval_*" | grep -v "$LATEST_EVAL_DIR" | xargs rm -rf

# 删除数值命名的目录
find "$PROJECT_ROOT/outputs" -maxdepth 1 -type d -name "[0-9]*" -exec rm -rf {} \;

echo "===== 项目清理完成 ====="
