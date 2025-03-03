#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据预处理脚本
用于处理原始O*NET数据并生成处理后的数据集
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到系统路径
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.data_processing.onet_processor import OnetProcessor


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='处理O*NET数据并生成技能匹配数据集')
    parser.add_argument('--raw_dir', type=str, default='data/raw/db_27_0_excel',
                        help='原始数据目录路径')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='输出目录路径')
    
    args = parser.parse_args()
    
    # 确保路径是相对于项目根目录的
    raw_dir = os.path.join(project_root, args.raw_dir)
    output_dir = os.path.join(project_root, args.output_dir)
    
    print(f"处理原始数据: {raw_dir}")
    print(f"输出目录: {output_dir}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化O*NET处理器
    processor = OnetProcessor(data_dir=raw_dir)
    
    # 加载数据
    print("加载O*NET数据...")
    processor.load_data()
    
    # 处理数据并保存
    print("处理数据并保存...")
    processor.save_processed_data(output_dir)
    
    # 创建职位-技能匹配数据集
    print("创建职位-技能匹配数据集...")
    processor.generate_job_skill_dataset(output_dir)
    
    print("数据处理完成!")


if __name__ == "__main__":
    main()
