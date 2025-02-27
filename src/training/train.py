#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练脚本，用于训练技能匹配模型
"""

import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import logging
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入自定义模块
from src.models.skill_matching_model import SkillMatchingModel, get_device
from src.training.trainer import Trainer
from src.data_processing.dataset import SkillMatchingDataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练技能匹配模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--device', type=str, default=None, help='计算设备 (cpu 或 cuda)')
    return parser.parse_args()

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    
    # 获取设备
    device = get_device(args.device)
    logger.info(f"使用设备: {device}")
    
    # 加载数据集
    logger.info("加载数据集...")
    train_dataset = SkillMatchingDataset(
        data_dir=config['data']['data_dir'],
        split='train',
        transform=None
    )
    
    val_dataset = SkillMatchingDataset(
        data_dir=config['data']['data_dir'],
        split='val',
        transform=None
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 创建模型
    logger.info("初始化模型...")
    model = SkillMatchingModel(**config['model'])
    
    # 设置技能图
    model.skill_graph = train_dataset.skill_graph
    logger.info(f"设置技能图: {model.skill_graph}")
    
    # 创建优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0)
    )
    
    # 创建学习率调度器
    scheduler = None
    if config['training'].get('lr_scheduler', None):
        scheduler_config = config['training']['lr_scheduler']
        if scheduler_config['type'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        elif scheduler_config['type'] == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_config['factor'],
                patience=scheduler_config['patience'],
                verbose=True
            )
    
    # 创建训练器
    logger.info("初始化训练器...")
    model_save_path = os.path.join(output_dir, "best_model.pth")
    log_file_path = os.path.join(output_dir, "training_log.jsonl")
    trainer = Trainer(
        model=model,
        device=device,
        model_save_path=model_save_path,
        log_file_path=log_file_path,
        **config['trainer']
    )
    
    # 恢复训练
    if args.resume:
        logger.info(f"从检查点恢复训练: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        early_stopping=config['trainer'].get('early_stopping', True),
        patience=config['trainer'].get('early_stopping_patience', 5)
    )
    
    logger.info(f"训练完成，模型保存在: {output_dir}")

if __name__ == '__main__':
    main()
