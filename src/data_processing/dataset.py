"""
数据集处理模块

用于将处理后的数据转换为PyTorch数据集，以便训练模型。
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import json
import networkx as nx
import random


class SkillMatchingDataset(Dataset):
    """
    技能匹配数据集
    
    用于训练技能匹配模型的数据集。
    """
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        初始化数据集
        
        参数:
            data_dir (str): 数据目录
            split (str): 数据集划分，可选 'train', 'val', 'test'
            transform (callable, optional): 数据转换函数
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # 加载数据
        self.load_data()
    
    def load_data(self):
        """
        加载数据
        """
        # 加载技能匹配数据集
        dataset_path = os.path.join(self.data_dir, 'processed', 'skill_matching_dataset.csv')
        self.data = pd.read_csv(dataset_path)
        
        # 加载职业元数据
        occupation_path = os.path.join(self.data_dir, 'processed', 'occupations.csv')
        self.occupations = pd.read_csv(occupation_path)
        
        # 加载技能元数据
        skill_path = os.path.join(self.data_dir, 'processed', 'skills.csv')
        self.skills = pd.read_csv(skill_path)
        
        # 加载技能图
        graph_path = os.path.join(self.data_dir, 'processed', 'skill_graph.json')
        with open(graph_path, 'r') as f:
            self.skill_graph_data = json.load(f)
        
        # 构建职业代码到索引的映射
        self.occupation_code_to_idx = {code: i for i, code in enumerate(self.occupations['occupation_code'])}
        
        # 划分数据集
        self.split_dataset()
        
        # 构建技能图
        self.build_skill_graph()
    
    def split_dataset(self):
        """
        划分数据集为训练集、验证集和测试集
        """
        # 获取唯一的职业代码
        unique_occupations = self.data['occupation_code'].unique()
        
        # 使用固定的随机种子以确保可重复性
        np.random.seed(42)
        np.random.shuffle(unique_occupations)
        
        # 划分职业
        n_occupations = len(unique_occupations)
        train_idx = int(n_occupations * 0.7)
        val_idx = int(n_occupations * 0.85)
        
        train_occupations = unique_occupations[:train_idx]
        val_occupations = unique_occupations[train_idx:val_idx]
        test_occupations = unique_occupations[val_idx:]
        
        # 根据划分选择数据
        if self.split == 'train':
            self.data = self.data[self.data['occupation_code'].isin(train_occupations)]
            
            # 检查正负样本比例
            positive_samples = self.data[self.data['match'] == 1]
            negative_samples = self.data[self.data['match'] == 0]
            pos_count = len(positive_samples)
            neg_count = len(negative_samples)
            
            print(f"训练集正样本数: {pos_count}, 负样本数: {neg_count}, 比例: {pos_count/neg_count:.2f}")
            
            # 如果正负样本比例不平衡，进行重采样
            if abs(pos_count / neg_count - 1.0) > 0.1:  # 如果比例偏差超过10%
                # 确定目标样本数
                target_count = max(pos_count, neg_count)
                
                # 对少数类进行过采样
                if pos_count < neg_count:
                    # 过采样正样本
                    oversampled_positive = positive_samples.sample(target_count, replace=True, random_state=42)
                    self.data = pd.concat([oversampled_positive, negative_samples]).sample(frac=1, random_state=42).reset_index(drop=True)
                    print(f"对正样本进行过采样，新的样本数: {len(self.data)}")
                else:
                    # 过采样负样本
                    oversampled_negative = negative_samples.sample(target_count, replace=True, random_state=42)
                    self.data = pd.concat([positive_samples, oversampled_negative]).sample(frac=1, random_state=42).reset_index(drop=True)
                    print(f"对负样本进行过采样，新的样本数: {len(self.data)}")
                
        elif self.split == 'val':
            self.data = self.data[self.data['occupation_code'].isin(val_occupations)]
        elif self.split == 'test':
            self.data = self.data[self.data['occupation_code'].isin(test_occupations)]
        else:
            raise ValueError(f"不支持的数据集划分: {self.split}")
        
        print(f"{self.split} 数据集大小: {len(self.data)}")
    
    def build_skill_graph(self):
        """
        构建技能图
        """
        # 从JSON文件加载技能图数据
        nodes = self.skill_graph_data['nodes']
        edges = self.skill_graph_data['edges']
        
        # 创建节点ID到索引的映射
        node_id_to_idx = {node['id']: i for i, node in enumerate(nodes)}
        
        # 创建节点特征
        num_nodes = len(nodes)
        
        # 使用更有意义的初始特征，而不是随机特征
        # 1. 使用节点度数作为特征的一部分
        node_degrees = {node['id']: 0 for node in nodes}
        for edge in edges:
            source_id = edge['source']
            target_id = edge['target']
            if source_id in node_degrees:
                node_degrees[source_id] += 1
            if target_id in node_degrees:
                node_degrees[target_id] += 1
        
        # 2. 使用节点名称的嵌入作为特征
        node_features = torch.zeros(num_nodes, 128)
        
        for i, node in enumerate(nodes):
            # 使用节点度数作为特征的一部分
            node_id = node['id']
            degree = node_degrees[node_id] / max(max(node_degrees.values()), 1)  # 归一化
            
            # 使用节点名称的哈希值来初始化一部分特征
            name = node.get('name', '')
            name_hash = hash(name) % 1000000
            
            # 将度数和哈希值转换为特征向量
            for j in range(64):  # 前64维使用哈希特征
                node_features[i, j] = ((name_hash >> j) & 1) * 0.1
            
            # 后64维使用度数和随机特征
            node_features[i, 64] = degree
            
            # 其余位置使用小的随机值
            for j in range(65, 128):
                node_features[i, j] = torch.randn(1).item() * 0.01
        
        # 创建边索引
        edge_index = []
        edge_attr = []
        
        for edge in edges:
            source_id = edge['source']
            target_id = edge['target']
            weight = edge['weight']
            
            # 获取节点索引
            if source_id in node_id_to_idx and target_id in node_id_to_idx:
                source_idx = node_id_to_idx[source_id]
                target_idx = node_id_to_idx[target_id]
                
                edge_index.append([source_idx, target_idx])
                edge_attr.append([float(weight)])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # 创建PyTorch Geometric数据对象
        self.skill_graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        
        print(f"技能图构建完成: {num_nodes}个节点, {len(edge_index[0])}条边")
    
    def get_occupation_features(self, occupation_code):
        """
        获取职业特征
        
        参数:
            occupation_code (str): 职业代码
            
        返回:
            torch.Tensor: 职业特征
        """
        # 从职业元数据中获取特征
        occupation_idx = self.occupation_code_to_idx[occupation_code]
        
        # 这里我们使用职业索引的one-hot编码作为特征
        # 然后将其映射到128维空间
        one_hot = torch.zeros(len(self.occupation_code_to_idx))
        one_hot[occupation_idx] = 1.0
        
        # 创建一个随机投影矩阵（在实际应用中，这应该是一个学习的映射）
        if not hasattr(self, 'occupation_projection'):
            # 设置随机种子以确保一致性
            torch.manual_seed(42)
            self.occupation_projection = torch.randn(len(self.occupation_code_to_idx), 128)
            
        # 将one-hot向量投影到128维空间
        features = torch.matmul(one_hot, self.occupation_projection)
        
        return features
    
    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取数据项
        
        参数:
            idx (int): 索引
            
        返回:
            tuple: (职业特征, 技能ID, 匹配标签, 重要性, 水平)
        """
        # 获取数据项
        item = self.data.iloc[idx]
        
        # 获取职业特征
        occupation_features = self.get_occupation_features(item['occupation_code'])
        
        # 获取技能ID和索引
        skill_id = item['skill_id']
        
        # 在节点ID到索引的映射中查找
        node_id_to_idx = {node['id']: i for i, node in enumerate(self.skill_graph_data['nodes'])}
        if skill_id in node_id_to_idx:
            skill_idx = node_id_to_idx[skill_id]
        else:
            # 如果找不到，使用随机索引
            skill_idx = random.randint(0, len(node_id_to_idx) - 1)
            print(f"警告: 找不到技能ID {skill_id}，使用随机索引 {skill_idx}")
        
        skill_idx = torch.tensor(skill_idx, dtype=torch.long)
        
        # 获取匹配标签
        match = torch.tensor(item['match'], dtype=torch.float)
        
        # 获取重要性和水平
        importance = torch.tensor(item['importance'], dtype=torch.float)
        level = torch.tensor(item['level'], dtype=torch.float)
        
        # 应用转换
        if self.transform:
            occupation_features, skill_idx, match, importance, level = self.transform(
                occupation_features, skill_idx, match, importance, level
            )
        
        return occupation_features, skill_idx, match, importance, level


def create_dataloader(data_dir, batch_size=32, split='train', num_workers=4):
    """
    创建数据加载器
    
    参数:
        data_dir (str): 数据目录
        batch_size (int): 批次大小
        split (str): 数据集划分，可选 'train', 'val', 'test'
        num_workers (int): 数据加载的工作线程数
        
    返回:
        DataLoader: 数据加载器
    """
    dataset = SkillMatchingDataset(data_dir=data_dir, split=split)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset


if __name__ == "__main__":
    # 测试数据集
    data_dir = "/home/u2021201733/test/skill_path_matching/data"
    
    # 创建数据集
    train_dataset = SkillMatchingDataset(data_dir=data_dir, split='train')
    
    # 获取第一个数据项
    occupation_features, skill_idx, match, importance, level = train_dataset[0]
    
    print(f"职业特征形状: {occupation_features.shape}")
    print(f"技能索引: {skill_idx}")
    print(f"匹配标签: {match}")
    print(f"重要性: {importance}")
    print(f"水平: {level}")
    
    # 获取技能图
    skill_graph = train_dataset.skill_graph
    
    print(f"技能图节点特征形状: {skill_graph.x.shape}")
    print(f"技能图边索引形状: {skill_graph.edge_index.shape}")
    
    # 创建数据加载器
    train_loader, _ = create_dataloader(data_dir=data_dir, batch_size=32, split='train')
    
    # 获取第一个批次
    for batch in train_loader:
        occupation_features, skill_idx, match, importance, level = batch
        
        print(f"批次大小: {occupation_features.shape[0]}")
        print(f"职业特征形状: {occupation_features.shape}")
        print(f"技能索引形状: {skill_idx.shape}")
        print(f"匹配标签形状: {match.shape}")
        print(f"重要性形状: {importance.shape}")
        print(f"水平形状: {level.shape}")
        
        break
