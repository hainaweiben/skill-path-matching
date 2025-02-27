#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型评估脚本
用于评估训练好的技能匹配模型
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.dataset import SkillMatchingDataset
from models.skill_matching_model import SkillMatchingModel
from utils.data_utils import load_model
from evaluation.metrics import (
    calculate_classification_metrics,
    calculate_ranking_metrics,
    calculate_ndcg
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估技能匹配模型')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--config_path', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='评估结果输出目录')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='分类阈值')
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_output_dir(output_dir=None):
    """创建输出目录"""
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('outputs', f'evaluation_{timestamp}')
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def evaluate_model(model, test_loader, threshold=0.5, device='cuda'):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估中"):
            # 解包数据
            occupation_features, skill_idx, labels, importance, level = batch
            
            # 转移到设备
            occupation_features = occupation_features.to(device)
            skill_idx = skill_idx.to(device)
            
            # 前向传播
            outputs, _ = model(
                occupation_features=occupation_features,
                skill_idx=skill_idx
            )
            
            # 收集预测和标签
            preds = outputs.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算指标
    classification_metrics = calculate_classification_metrics(all_labels, all_preds, threshold)
    ranking_metrics = calculate_ranking_metrics(all_labels, all_preds)
    ndcg_metrics = calculate_ndcg(all_labels, all_preds)
    
    # 合并所有指标
    metrics = {
        **classification_metrics,
        **ranking_metrics,
        **ndcg_metrics
    }
    
    return metrics, all_preds, all_labels


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config_path)
    
    # 创建输出目录
    output_dir = create_output_dir(args.output_dir)
    logger.info(f"评估结果将保存到: {output_dir}")
    
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载测试数据集
    logger.info("加载测试数据集...")
    test_dataset = SkillMatchingDataset(
        data_dir=config['data']['data_dir'],
        split='test'
    )
    logger.info(f"测试数据集大小: {len(test_dataset)}")
    
    # 创建数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # 初始化模型
    logger.info("初始化模型...")
    model = SkillMatchingModel(
        skill_input_dim=config['model']['skill_input_dim'],
        occupation_input_dim=config['model']['occupation_input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        embedding_dim=config['model']['embedding_dim'],
        num_gnn_layers=config['model']['num_gnn_layers'],
        num_mlp_layers=config['model']['num_mlp_layers'],
        dropout=config['model']['dropout'],
        gnn_type=config['model']['gnn_type']
    )
    
    # 加载模型权重
    logger.info(f"加载模型权重: {args.model_path}")
    model, metadata = load_model(model, args.model_path)
    model.to(device)
    
    # 设置技能图
    model.skill_graph = test_dataset.skill_graph.to(device)
    logger.info(f"设置技能图: {model.skill_graph}")
    
    # 评估模型
    logger.info("开始评估...")
    metrics, all_preds, all_labels = evaluate_model(
        model=model,
        test_loader=test_loader,
        threshold=args.threshold,
        device=device
    )
    
    # 打印评估结果
    logger.info("评估结果:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    # 保存评估结果
    results = {
        'metrics': {k: float(v) for k, v in metrics.items()},  # 确保所有值都是Python原生类型
        'config': config,
        'args': vars(args),
        'threshold': args.threshold
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存预测结果
    predictions_df = pd.DataFrame({
        'true_label': all_labels,
        'prediction': all_preds
    })
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    logger.info(f"评估完成，结果保存在: {output_dir}")


if __name__ == '__main__':
    main()
