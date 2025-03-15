#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
构建数据集脚本
使用处理后的数据构建技能匹配数据集和技能图
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到系统路径
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.data_processing.dataset import SkillMatchingDataset


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='构建技能匹配数据集和技能图')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='处理后的数据目录路径')
    parser.add_argument('--word2vec_path', type=str, default='word2vec-models/word2vec-google-news-300.bin',
                        help='预训练词向量模型路径')
    
    args = parser.parse_args()
    
    # 确保路径是相对于项目根目录的
    data_dir = os.path.join(project_root, args.data_dir)
    word2vec_path = os.path.join(project_root, args.word2vec_path)
    
    print(f"数据目录: {data_dir}")
    print(f"词向量模型路径: {word2vec_path}")
    
    # 构建训练集
    print("构建训练集...")
    train_dataset = SkillMatchingDataset(
        data_dir=data_dir,
        split='train',
        word2vec_path=word2vec_path
    )
    
    # 构建验证集
    print("构建验证集...")
    val_dataset = SkillMatchingDataset(
        data_dir=data_dir,
        split='val',
        word2vec_path=word2vec_path
    )
    
    # 构建测试集
    print("构建测试集...")
    test_dataset = SkillMatchingDataset(
        data_dir=data_dir,
        split='test',
        word2vec_path=word2vec_path
    )
    
    # 打印数据集信息
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 打印技能图信息
    skill_graph = train_dataset.skill_graph
    print(f"技能图节点数: {skill_graph.num_nodes}")
    print(f"技能图边数: {skill_graph.num_edges}")
    
    print("数据集构建完成!")


if __name__ == "__main__":
    main()
