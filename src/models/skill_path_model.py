"""
能力点路径模型
用于构建和分析技能之间的关系图，识别能力点发展路径
"""

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Set, Optional
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data

class SkillGraph:
    """技能关系图，用于构建和分析技能之间的关系"""
    
    def __init__(self):
        """初始化技能关系图"""
        self.graph = nx.DiGraph()
        self.skill_to_idx = {}  # 技能到索引的映射
        self.idx_to_skill = {}  # 索引到技能的映射
        self.skill_embeddings = {}  # 技能嵌入向量
    
    def add_skill(self, skill: str, attributes: Dict = None):
        """
        添加技能节点
        
        Args:
            skill: 技能名称
            attributes: 技能属性
        """
        if skill not in self.skill_to_idx:
            idx = len(self.skill_to_idx)
            self.skill_to_idx[skill] = idx
            self.idx_to_skill[idx] = skill
            
        if attributes:
            self.graph.add_node(self.skill_to_idx[skill], **attributes)
        else:
            self.graph.add_node(self.skill_to_idx[skill])
    
    def add_relationship(self, source_skill: str, target_skill: str, relationship_type: str, weight: float = 1.0):
        """
        添加技能之间的关系
        
        Args:
            source_skill: 源技能
            target_skill: 目标技能
            relationship_type: 关系类型，如"prerequisite"（前置）、"leads_to"（导向）等
            weight: 关系权重
        """
        # 确保两个技能都已添加到图中
        if source_skill not in self.skill_to_idx:
            self.add_skill(source_skill)
        
        if target_skill not in self.skill_to_idx:
            self.add_skill(target_skill)
        
        source_idx = self.skill_to_idx[source_skill]
        target_idx = self.skill_to_idx[target_skill]
        
        self.graph.add_edge(source_idx, target_idx, type=relationship_type, weight=weight)
    
    def get_skill_path(self, source_skill: str, target_skill: str) -> List[str]:
        """
        获取从源技能到目标技能的路径
        
        Args:
            source_skill: 源技能
            target_skill: 目标技能
            
        Returns:
            技能路径列表
        """
        if source_skill not in self.skill_to_idx or target_skill not in self.skill_to_idx:
            return []
        
        source_idx = self.skill_to_idx[source_skill]
        target_idx = self.skill_to_idx[target_skill]
        
        try:
            path_indices = nx.shortest_path(self.graph, source=source_idx, target=target_idx, weight='weight')
            return [self.idx_to_skill[idx] for idx in path_indices]
        except nx.NetworkXNoPath:
            return []
    
    def get_prerequisite_skills(self, skill: str) -> List[str]:
        """
        获取指定技能的前置技能
        
        Args:
            skill: 技能名称
            
        Returns:
            前置技能列表
        """
        if skill not in self.skill_to_idx:
            return []
        
        skill_idx = self.skill_to_idx[skill]
        predecessors = list(self.graph.predecessors(skill_idx))
        return [self.idx_to_skill[idx] for idx in predecessors]
    
    def get_next_level_skills(self, skill: str) -> List[str]:
        """
        获取指定技能的下一级技能
        
        Args:
            skill: 技能名称
            
        Returns:
            下一级技能列表
        """
        if skill not in self.skill_to_idx:
            return []
        
        skill_idx = self.skill_to_idx[skill]
        successors = list(self.graph.successors(skill_idx))
        return [self.idx_to_skill[idx] for idx in successors]
    
    def get_skill_similarity(self, skill1: str, skill2: str) -> float:
        """
        计算两个技能之间的相似度
        
        Args:
            skill1: 第一个技能
            skill2: 第二个技能
            
        Returns:
            相似度分数
        """
        if skill1 not in self.skill_embeddings or skill2 not in self.skill_embeddings:
            return 0.0
        
        # 使用余弦相似度计算技能嵌入向量的相似度
        vec1 = self.skill_embeddings[skill1]
        vec2 = self.skill_embeddings[skill2]
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def to_pytorch_geometric_data(self) -> Data:
        """
        将技能图转换为PyTorch Geometric数据格式
        
        Returns:
            PyTorch Geometric Data对象
        """
        # 节点特征矩阵
        num_nodes = len(self.skill_to_idx)
        # 这里假设我们有一个节点特征矩阵，实际应用中需要根据具体情况构建
        node_features = torch.eye(num_nodes)  # 简单起见，使用单位矩阵作为节点特征
        
        # 边索引和边特征
        edge_index = []
        edge_attr = []
        
        for source, target, data in self.graph.edges(data=True):
            edge_index.append([source, target])
            # 使用边的权重作为边特征
            edge_attr.append([data.get('weight', 1.0)])
        
        if not edge_index:  # 如果没有边，返回空图
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    
    def save(self, path: str):
        """
        保存技能图到文件
        
        Args:
            path: 文件路径
        """
        data = {
            'nodes': list(self.graph.nodes(data=True)),
            'edges': list(self.graph.edges(data=True)),
            'skill_to_idx': self.skill_to_idx,
            'idx_to_skill': {str(k): v for k, v in self.idx_to_skill.items()},  # 将整数键转换为字符串
            'skill_embeddings': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                for k, v in self.skill_embeddings.items()}
        }
        
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SkillGraph':
        """
        从文件加载技能图
        
        Args:
            path: 文件路径
            
        Returns:
            SkillGraph对象
        """
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        skill_graph = cls()
        
        # 恢复映射
        skill_graph.skill_to_idx = data['skill_to_idx']
        skill_graph.idx_to_skill = {int(k): v for k, v in data['idx_to_skill'].items()}  # 将字符串键转换回整数
        
        # 恢复图
        skill_graph.graph = nx.DiGraph()
        
        for node, attrs in data['nodes']:
            skill_graph.graph.add_node(node, **attrs)
        
        for source, target, attrs in data['edges']:
            skill_graph.graph.add_edge(source, target, **attrs)
        
        # 恢复嵌入向量
        skill_graph.skill_embeddings = {k: np.array(v) if isinstance(v, list) else v 
                                       for k, v in data['skill_embeddings'].items()}
        
        return skill_graph


class SkillPathGNN(nn.Module):
    """基于图神经网络的能力点路径模型"""
    
    def __init__(self, num_node_features: int, hidden_channels: int = 64, num_classes: int = 1):
        """
        初始化模型
        
        Args:
            num_node_features: 节点特征维度
            hidden_channels: 隐藏层维度
            num_classes: 输出类别数（对于回归任务，为1）
        """
        super(SkillPathGNN, self).__init__()
        
        # 图卷积层
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # 图注意力层
        self.gat = GATConv(hidden_channels, hidden_channels, heads=4, dropout=0.6)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_channels * 4, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_classes)
    
    def forward(self, data):
        """
        前向传播
        
        Args:
            data: PyTorch Geometric Data对象
            
        Returns:
            模型输出
        """
        x, edge_index = data.x, data.edge_index
        
        # 图卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # 图注意力
        x = self.gat(x, edge_index)
        
        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.fc2(x)
        
        return x
