"""
测试技能匹配模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from src.models.skill_matching_model import SkillMatchingModel
from src.data_processing.dataset import SkillMatchingDataset, create_dataloader


def test_model_forward():
    """
    测试模型前向传播
    """
    print("加载数据集...")
    data_dir = "/home/u2021201733/test/skill_path_matching/data"
    train_loader, train_dataset = create_dataloader(
        data_dir=data_dir, batch_size=4, split='train'
    )
    
    # 获取第一个批次
    for batch in train_loader:
        occupation_features, skill_idx, match, importance, level = batch
        break
    
    # 获取技能图
    skill_graph = train_dataset.skill_graph
    
    # 创建模型
    print("创建模型...")
    model = SkillMatchingModel(
        skill_input_dim=skill_graph.x.size(1),
        occupation_input_dim=occupation_features.size(1),
        hidden_dim=64,
        embedding_dim=32,
        num_gnn_layers=2,
        num_mlp_layers=2,
        dropout=0.1,
        gnn_type='gcn'
    )
    
    # 前向传播
    print("执行前向传播...")
    match_prob = model(skill_graph, occupation_features)
    
    print(f"匹配概率形状: {match_prob.shape}")
    print(f"匹配概率: {match_prob}")
    
    return model, skill_graph, occupation_features, match_prob


def compute_loss(model, skill_graph, occupation_features, target_skill_idx, target_match):
    """
    计算损失
    
    参数:
        model: 模型
        skill_graph: 技能图
        occupation_features: 职业特征
        target_skill_idx: 目标技能索引
        target_match: 目标匹配标签
        
    返回:
        loss: 损失
    """
    # 前向传播
    match_prob = model(skill_graph, occupation_features)
    
    # 提取目标技能的匹配概率
    batch_size = occupation_features.size(0)
    selected_probs = torch.zeros(batch_size, dtype=torch.float)
    
    for i in range(batch_size):
        selected_probs[i] = match_prob[i, target_skill_idx[i]]
    
    # 计算二元交叉熵损失
    loss = F.binary_cross_entropy(selected_probs, target_match)
    
    return loss, selected_probs


def test_model_training():
    """
    测试模型训练
    """
    print("\n测试模型训练...")
    
    # 加载数据
    data_dir = "/home/u2021201733/test/skill_path_matching/data"
    train_loader, train_dataset = create_dataloader(
        data_dir=data_dir, batch_size=16, split='train'
    )
    
    # 获取技能图
    skill_graph = train_dataset.skill_graph
    
    # 获取第一个批次
    for batch in train_loader:
        occupation_features, skill_idx, match, importance, level = batch
        break
    
    # 创建模型
    model = SkillMatchingModel(
        skill_input_dim=skill_graph.x.size(1),
        occupation_input_dim=occupation_features.size(1),
        hidden_dim=64,
        embedding_dim=32,
        num_gnn_layers=2,
        num_mlp_layers=2,
        dropout=0.1,
        gnn_type='gcn'
    )
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练一个小批次
    model.train()
    optimizer.zero_grad()
    
    # 计算损失
    loss, selected_probs = compute_loss(
        model, skill_graph, occupation_features, skill_idx, match
    )
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    print(f"训练损失: {loss.item()}")
    print(f"预测概率: {selected_probs[:5]}")
    print(f"目标标签: {match[:5]}")
    
    # 再次前向传播，检查损失是否下降
    model.eval()
    with torch.no_grad():
        new_loss, new_selected_probs = compute_loss(
            model, skill_graph, occupation_features, skill_idx, match
        )
    
    print(f"更新后的损失: {new_loss.item()}")
    print(f"更新后的预测概率: {new_selected_probs[:5]}")


if __name__ == "__main__":
    # 测试模型前向传播
    model, skill_graph, occupation_features, match_prob = test_model_forward()
    
    # 测试模型训练
    test_model_training()
