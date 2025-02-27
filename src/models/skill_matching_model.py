"""
技能匹配模型

这个模块实现了基于能力路径建模的人才和技能匹配深度学习模型。
模型使用图神经网络来学习技能之间的关系，并预测职业和技能之间的匹配程度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data, Batch


class SkillPathEncoder(nn.Module):
    """
    技能路径编码器
    
    使用图神经网络对技能图进行编码，学习技能之间的关系。
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1, 
                 gnn_type='gcn'):
        """
        初始化技能路径编码器
        
        参数:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
            output_dim (int): 输出特征维度
            num_layers (int): GNN层数
            dropout (float): Dropout概率
            gnn_type (str): GNN类型，可选 'gcn', 'sage', 'gat'
        """
        super(SkillPathEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        
        # 选择GNN类型
        if gnn_type == 'gcn':
            GNNLayer = GCNConv
        elif gnn_type == 'sage':
            GNNLayer = SAGEConv
        elif gnn_type == 'gat':
            GNNLayer = GATConv
        else:
            raise ValueError(f"不支持的GNN类型: {gnn_type}")
        
        # 构建GNN层
        self.gnn_layers = nn.ModuleList()
        
        # 第一层: 输入维度 -> 隐藏维度
        self.gnn_layers.append(GNNLayer(input_dim, hidden_dim))
        
        # 中间层: 隐藏维度 -> 隐藏维度
        for _ in range(num_layers - 2):
            self.gnn_layers.append(GNNLayer(hidden_dim, hidden_dim))
        
        # 最后一层: 隐藏维度 -> 输出维度
        if num_layers > 1:
            self.gnn_layers.append(GNNLayer(hidden_dim, output_dim))
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 节点特征矩阵
            edge_index (torch.Tensor): 边索引
            edge_attr (torch.Tensor, optional): 边特征
            
        返回:
            torch.Tensor: 节点嵌入
        """
        for i, gnn_layer in enumerate(self.gnn_layers):
            # 对于GAT，需要提供边特征
            if self.gnn_type == 'gat' and edge_attr is not None:
                x = gnn_layer(x, edge_index, edge_attr)
            else:
                x = gnn_layer(x, edge_index)
            
            # 除了最后一层外，应用ReLU和Dropout
            if i < len(self.gnn_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class OccupationEncoder(nn.Module):
    """
    职业编码器
    
    将职业特征编码为向量表示。
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        """
        初始化职业编码器
        
        参数:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
            output_dim (int): 输出特征维度
            num_layers (int): 层数
            dropout (float): Dropout概率
        """
        super(OccupationEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 构建MLP层
        self.layers = nn.ModuleList()
        
        # 第一层: 输入维度 -> 隐藏维度
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # 中间层: 隐藏维度 -> 隐藏维度
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # 最后一层: 隐藏维度 -> 输出维度
        if num_layers > 1:
            self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 职业特征
            
        返回:
            torch.Tensor: 职业嵌入
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # 除了最后一层外，应用ReLU和Dropout
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class SkillMatchingModel(nn.Module):
    """
    技能匹配模型
    
    结合技能路径编码器和职业编码器，预测职业和技能之间的匹配程度。
    """
    
    def __init__(self, skill_input_dim, occupation_input_dim, hidden_dim=128, 
                 embedding_dim=64, num_gnn_layers=2, num_mlp_layers=2, 
                 dropout=0.1, gnn_type='gcn'):
        """
        初始化技能匹配模型
        
        参数:
            skill_input_dim (int): 技能特征维度
            occupation_input_dim (int): 职业特征维度
            hidden_dim (int): 隐藏层维度
            embedding_dim (int): 嵌入维度
            num_gnn_layers (int): GNN层数
            num_mlp_layers (int): MLP层数
            dropout (float): Dropout概率
            gnn_type (str): GNN类型，可选 'gcn', 'sage', 'gat'
        """
        super(SkillMatchingModel, self).__init__()
        
        # 技能路径编码器
        self.skill_encoder = SkillPathEncoder(
            input_dim=skill_input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
            gnn_type=gnn_type
        )
        
        # 职业编码器
        self.occupation_encoder = OccupationEncoder(
            input_dim=occupation_input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_mlp_layers,
            dropout=dropout
        )
        
        # 匹配预测层
        self.matching_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 技能图
        self.skill_graph = None
        self.loss_fn = nn.BCELoss()
    
    def forward(self, occupation_features, skill_idx, match=None, importance=None, level=None):
        """
        前向传播
        
        参数:
            occupation_features (torch.Tensor): 职业特征，形状为 [batch_size, occupation_feature_dim]
            skill_idx (torch.Tensor): 技能索引，形状为 [batch_size]
            match (torch.Tensor, optional): 匹配标签，形状为 [batch_size]
            importance (torch.Tensor, optional): 重要性，形状为 [batch_size]
            level (torch.Tensor, optional): 水平，形状为 [batch_size]
            
        返回:
            torch.Tensor: 匹配概率，形状为 [batch_size]
        """
        # 确保技能图存在
        if not hasattr(self, 'skill_graph'):
            raise ValueError("模型缺少技能图，请先设置model.skill_graph")
        
        # 将技能图移动到与输入相同的设备
        device = occupation_features.device
        skill_graph = self.skill_graph.to(device)
        
        # 使用GNN编码技能图
        skill_node_embeddings = self.skill_encoder(skill_graph.x, skill_graph.edge_index, skill_graph.edge_attr)
        
        # 获取批次中每个技能的嵌入
        batch_size = skill_idx.size(0)
        skill_embeddings = skill_node_embeddings[skill_idx]
        
        # 使用MLP编码职业特征
        occupation_embeddings = self.occupation_encoder(occupation_features)
        
        # 计算匹配分数
        match_scores = self.matching_predictor(
            torch.cat([occupation_embeddings, skill_embeddings], dim=1)
        )
        
        # 计算损失（如果提供了匹配标签）
        loss = None
        if match is not None:
            loss = self.loss_fn(match_scores.squeeze(), match)
            
            # 如果提供了重要性，则对损失进行加权
            if importance is not None:
                loss = loss * importance
                
            # 如果提供了水平，则对损失进行加权
            if level is not None:
                # 将水平归一化到0-1范围
                normalized_level = level / 5.0  # 假设水平范围为0-5
                loss = loss * (1.0 + normalized_level)  # 高水平的样本损失权重更大
                
            # 计算平均损失
            loss = loss.mean()
        
        return match_scores.squeeze(), loss


class SkillPathMatchingModel(nn.Module):
    """
    技能路径匹配模型
    
    扩展基本的技能匹配模型，考虑技能路径和顺序。
    """
    
    def __init__(self, skill_input_dim, occupation_input_dim, hidden_dim=128, 
                 embedding_dim=64, num_gnn_layers=2, num_mlp_layers=2, 
                 dropout=0.1, gnn_type='gcn'):
        """
        初始化技能路径匹配模型
        
        参数:
            skill_input_dim (int): 技能特征维度
            occupation_input_dim (int): 职业特征维度
            hidden_dim (int): 隐藏层维度
            embedding_dim (int): 嵌入维度
            num_gnn_layers (int): GNN层数
            num_mlp_layers (int): MLP层数
            dropout (float): Dropout概率
            gnn_type (str): GNN类型，可选 'gcn', 'sage', 'gat'
        """
        super(SkillPathMatchingModel, self).__init__()
        
        # 基本技能匹配模型
        self.base_model = SkillMatchingModel(
            skill_input_dim=skill_input_dim,
            occupation_input_dim=occupation_input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_gnn_layers=num_gnn_layers,
            num_mlp_layers=num_mlp_layers,
            dropout=dropout,
            gnn_type=gnn_type
        )
        
        # 路径注意力层，用于关注技能路径中的重要技能
        self.path_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=dropout
        )
        
        # 路径聚合层
        self.path_aggregation = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 最终预测层
        self.final_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, skill_graph, occupation_features, skill_paths):
        """
        前向传播
        
        参数:
            skill_graph (Data): 技能图
            occupation_features (torch.Tensor): 职业特征
            skill_paths (list of list): 技能路径，每个路径是一个技能ID列表
            
        返回:
            torch.Tensor: 匹配概率
        """
        # 确保数据在正确的设备上
        device = occupation_features.device
        skill_graph.x = skill_graph.x.to(device)
        skill_graph.edge_index = skill_graph.edge_index.to(device)
        if hasattr(skill_graph, 'edge_attr'):
            skill_graph.edge_attr = skill_graph.edge_attr.to(device)
        
        # 获取基本技能嵌入
        skill_embeddings = self.base_model.skill_encoder(
            skill_graph.x, 
            skill_graph.edge_index, 
            skill_graph.edge_attr if hasattr(skill_graph, 'edge_attr') else None
        )
        
        # 获取职业嵌入
        occupation_embedding = self.base_model.occupation_encoder(occupation_features)
        
        # 处理每个技能路径
        path_embeddings = []
        for path in skill_paths:
            # 获取路径中的技能嵌入
            path_skill_embeddings = skill_embeddings[path]
            
            # 应用注意力机制
            attn_output, _ = self.path_attention(
                path_skill_embeddings.unsqueeze(1),
                path_skill_embeddings.unsqueeze(1),
                path_skill_embeddings.unsqueeze(1)
            )
            
            # 聚合路径嵌入
            path_embedding = self.path_aggregation(attn_output.squeeze(1))
            
            # 取平均作为最终路径嵌入
            path_embedding = path_embedding.mean(dim=0)
            
            path_embeddings.append(path_embedding)
        
        # 将所有路径嵌入堆叠
        path_embeddings = torch.stack(path_embeddings)
        
        # 将职业嵌入与路径嵌入拼接
        combined_embedding = torch.cat([
            occupation_embedding.expand(len(path_embeddings), -1),
            path_embeddings
        ], dim=1)
        
        # 预测最终匹配度
        match_prob = self.final_predictor(combined_embedding)
        
        return match_prob.squeeze()


# 用于测试的简单模型
def create_test_model(num_skills=35, num_occupation_features=10):
    """
    创建一个用于测试的简单模型
    
    参数:
        num_skills (int): 技能数量
        num_occupation_features (int): 职业特征数量
        
    返回:
        SkillMatchingModel: 技能匹配模型
    """
    # 假设每个技能有10个特征
    skill_input_dim = 10
    occupation_input_dim = num_occupation_features
    
    model = SkillMatchingModel(
        skill_input_dim=skill_input_dim,
        occupation_input_dim=occupation_input_dim,
        hidden_dim=64,
        embedding_dim=32,
        num_gnn_layers=2,
        num_mlp_layers=2,
        dropout=0.1,
        gnn_type='gcn'
    )
    
    return model


def get_device():
    """
    获取可用的计算设备
    
    返回:
        torch.device: 计算设备
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


if __name__ == "__main__":
    # 测试模型
    import numpy as np
    
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 创建一个简单的技能图
    num_skills = 35
    num_occupation_features = 10
    
    # 随机生成技能特征
    skill_features = torch.randn(num_skills, 10).to(device)
    
    # 随机生成边索引 (假设每个技能与其他3个技能相连)
    edge_index = []
    for i in range(num_skills):
        # 随机选择3个不同的技能
        connected_skills = np.random.choice(
            [j for j in range(num_skills) if j != i],
            size=3,
            replace=False
        )
        for j in connected_skills:
            edge_index.append([i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    
    # 随机生成边特征 (假设每条边有1个特征)
    edge_attr = torch.randn(edge_index.size(1), 1).to(device)
    
    # 创建技能图
    skill_graph = Data(x=skill_features, edge_index=edge_index, edge_attr=edge_attr)
    
    # 随机生成职业特征
    batch_size = 4
    occupation_features = torch.randn(batch_size, num_occupation_features).to(device)
    
    # 创建模型
    model = create_test_model(num_skills, num_occupation_features).to(device)
    
    # 设置技能图
    model.skill_graph = skill_graph
    
    # 前向传播
    match_prob, loss = model(occupation_features, torch.randint(0, num_skills, (batch_size,)))
    
    print(f"匹配概率形状: {match_prob.shape}")
    print(f"匹配概率: {match_prob}")
    print(f"损失: {loss}")
