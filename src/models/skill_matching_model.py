"""
技能匹配模型

这个模块实现了基于能力路径建模的人才和技能匹配深度学习模型。
模型使用图神经网络来学习技能之间的关系，并预测职业和技能之间的匹配程度。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance

    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t is the predicted probability of the target class
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        Initialize Focal Loss

        Args:
            alpha (float): Weighting factor for the rare class
            gamma (float): Focusing parameter that reduces the loss contribution from easy examples
            reduction (str): Reduction method, options: 'mean', 'sum', 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6  # Small constant to prevent numerical instability

    def forward(self, inputs, targets):
        """
        Forward pass

        Args:
            inputs (torch.Tensor): Predicted probabilities, shape [batch_size]
            targets (torch.Tensor): Target labels (0 or 1), shape [batch_size]

        Returns:
            torch.Tensor: Focal loss
        """
        # Ensure inputs are between eps and 1-eps for numerical stability
        inputs = torch.clamp(inputs, self.eps, 1.0 - self.eps)

        # Calculate binary cross entropy
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")

        # Calculate p_t
        p_t = inputs * targets + (1 - inputs) * (1 - targets)

        # Calculate focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class SkillPathEncoder(nn.Module):
    """
    技能路径编码器

    使用图神经网络对技能图进行编码，学习技能之间的关系。
    增强版本支持有序消息传递和边特征处理。
    """

    def _get_gnn_layer(self):
        """根据GNN类型返回相应的GNN层类和参数"""
        if self.gnn_type == "gcn":
            return GCNConv, {}
        elif self.gnn_type == "sage":
            return SAGEConv, {}
        elif self.gnn_type == "gat":
            return GATConv, {"heads": self.heads}
        else:
            raise ValueError(f"不支持的GNN类型: {self.gnn_type}")

    def _create_edge_encoders(self, hidden_dim):
        """创建边特征编码器"""
        if self.edge_dim is not None:
            self.edge_encoders = nn.ModuleList()
            for _ in range(self.num_layers):
                self.edge_encoders.append(nn.Linear(self.edge_dim, hidden_dim))

    def _create_first_layer(self, GNNLayer, hidden_dim, gnn_kwargs):
        """创建第一个GNN层"""
        if self.gnn_type == "gat":
            if self.edge_dim is not None:
                return GNNLayer(self.input_dim, hidden_dim, edge_dim=hidden_dim, **gnn_kwargs)
            return GNNLayer(self.input_dim, hidden_dim, **gnn_kwargs)
        return GNNLayer(self.input_dim, hidden_dim)

    def _create_middle_layer(self, GNNLayer, hidden_dim, gnn_kwargs):
        """创建中间GNN层"""
        if self.gnn_type == "gat":
            if self.edge_dim is not None:
                return GNNLayer(hidden_dim * self.heads, hidden_dim, edge_dim=hidden_dim, **gnn_kwargs)
            return GNNLayer(hidden_dim * self.heads, hidden_dim, **gnn_kwargs)
        return GNNLayer(hidden_dim, hidden_dim)

    def _create_final_layer(self, GNNLayer, hidden_dim):
        """创建最后一个GNN层"""
        if self.gnn_type == "gat":
            if self.edge_dim is not None:
                return GNNLayer(hidden_dim * self.heads, self.output_dim, edge_dim=hidden_dim, heads=1)
            return GNNLayer(hidden_dim * self.heads, self.output_dim, heads=1)
        return GNNLayer(hidden_dim, self.output_dim)

    def _create_order_attention(self, hidden_dim):
        """创建有序消息传递的注意力层"""
        if self.use_ordered_msg_passing:
            self.order_attention = nn.ModuleList()
            for i in range(self.num_layers):
                if i == 0:
                    self.order_attention.append(nn.Linear(self.input_dim, 1))
                elif i == self.num_layers - 1:
                    self.order_attention.append(nn.Linear(self.output_dim, 1))
                else:
                    self.order_attention.append(nn.Linear(hidden_dim, 1))

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        dropout=0.1,
        gnn_type="gcn",
        heads=4,
        edge_dim=None,
        use_ordered_msg_passing=True,
    ):
        """
        初始化技能路径编码器

        参数:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
            output_dim (int): 输出特征维度
            num_layers (int): GNN层数
            dropout (float): Dropout概率
            gnn_type (str): GNN类型，可选 'gcn', 'sage', 'gat'
            heads (int): GAT的注意力头数
            edge_dim (int, optional): 边特征维度，如果为None则不使用边特征
            use_ordered_msg_passing (bool): 是否使用有序消息传递
        """
        super().__init__()

        # 保存参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.heads = heads
        self.edge_dim = edge_dim
        self.use_ordered_msg_passing = use_ordered_msg_passing

        # 获取GNN层类型和参数
        GNNLayer, self.gnn_kwargs = self._get_gnn_layer()

        # 调整隐藏维度（用于GAT）
        if gnn_type == "gat" and num_layers > 1:
            hidden_dim = hidden_dim // heads

        # 创建边特征编码器
        self._create_edge_encoders(hidden_dim)

        # 构建GNN层
        self.gnn_layers = nn.ModuleList()

        # 添加第一层
        self.gnn_layers.append(self._create_first_layer(GNNLayer, hidden_dim, self.gnn_kwargs))

        # 添加中间层
        for _ in range(num_layers - 2):
            self.gnn_layers.append(self._create_middle_layer(GNNLayer, hidden_dim, self.gnn_kwargs))

        # 添加最后一层
        if num_layers > 1:
            self.gnn_layers.append(self._create_final_layer(GNNLayer, hidden_dim))

        # 创建有序消息传递的注意力层
        self._create_order_attention(hidden_dim)

        # Dropout层和激活函数
        self.dropout_layer = nn.Dropout(dropout)
        self.activation = nn.ReLU()

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
        # 确保数据在同一设备上
        device = x.device
        edge_index = edge_index.to(device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)

        # 保存原始节点特征用于残差连接
        original_x = x

        for i, gnn_layer in enumerate(self.gnn_layers):
            # 处理边特征（如果有）
            transformed_edge_attr = None
            if edge_attr is not None and hasattr(self, "edge_encoders"):
                transformed_edge_attr = self.edge_encoders[i](edge_attr)
                transformed_edge_attr = F.relu(transformed_edge_attr)

            # 有序消息传递
            if self.use_ordered_msg_passing and hasattr(self, "order_attention"):
                # 计算节点重要性分数
                node_scores = self.order_attention[i](x).squeeze(-1)

                # 为每条边计算源节点的重要性权重
                src_nodes, dst_nodes = edge_index
                edge_weights = torch.exp(node_scores[src_nodes])

                # 如果有边特征，将边权重与边特征结合
                if transformed_edge_attr is not None:
                    # 扩展边权重维度以匹配边特征维度
                    edge_weights = edge_weights.unsqueeze(-1).expand_as(transformed_edge_attr)
                    transformed_edge_attr = transformed_edge_attr * edge_weights

                # 对于GAT，需要提供边特征
                if self.gnn_type == "gat" and transformed_edge_attr is not None:
                    x_new = gnn_layer(x, edge_index, transformed_edge_attr)
                else:
                    # 对于不支持边特征的GNN，我们可以通过修改邻接矩阵来实现加权
                    if self.gnn_type != "gat" and edge_weights is not None:
                        # 创建一个带权重的稀疏邻接矩阵
                        edge_weights_flat = edge_weights
                        if edge_weights_flat.dim() > 1:
                            edge_weights_flat = edge_weights_flat.mean(dim=1)
                        x_new = gnn_layer(x, edge_index, edge_weights_flat)
                    else:
                        x_new = gnn_layer(x, edge_index)
            else:
                # 标准GNN前向传播
                if self.gnn_type == "gat" and transformed_edge_attr is not None:
                    x_new = gnn_layer(x, edge_index, transformed_edge_attr)
                else:
                    x_new = gnn_layer(x, edge_index)

            # 应用残差连接（如果维度匹配）
            if x.size(-1) == x_new.size(-1):
                x = x_new + x
            else:
                x = x_new

            # 除了最后一层外，应用ReLU和Dropout
            if i < len(self.gnn_layers) - 1:
                x = self.activation(x)
                x = self.dropout_layer(x)

        # 最终添加全局残差连接（如果维度匹配）
        if original_x.size(-1) == x.size(-1):
            x = x + original_x

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
        super().__init__()

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

    def __init__(
        self,
        skill_input_dim,
        occupation_input_dim,
        hidden_dim=128,
        embedding_dim=64,
        num_gnn_layers=2,
        num_mlp_layers=2,
        dropout=0.1,
        gnn_type="gat",
        focal_alpha=0.25,
        focal_gamma=2.0,
    ):
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
            focal_alpha (float): Focal Loss的alpha参数
            focal_gamma (float): Focal Loss的gamma参数
        """
        super().__init__()

        # 确保embedding_dim是4的倍数，以便于GAT的多头注意力
        if gnn_type == "gat" and embedding_dim % 4 != 0:
            embedding_dim = (embedding_dim // 4 + 1) * 4
            print(f"调整embedding_dim为{embedding_dim}以适应GAT的4个注意力头")

        # 技能路径编码器
        self.skill_encoder = SkillPathEncoder(
            input_dim=skill_input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            heads=4 if gnn_type == "gat" else 1,
        )

        # 职业编码器
        self.occupation_encoder = OccupationEncoder(
            input_dim=occupation_input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_mlp_layers,
            dropout=dropout,
        )

        # 特征交互层 - 使用注意力机制
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, dropout=dropout)

        # 特征增强层
        self.feature_enhancement = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),  # 包含原始特征和交互特征
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # 匹配预测层 - 更深的MLP
        self.matching_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        # 技能图
        self.skill_graph = None

        # 使用Focal Loss替代BCE Loss
        self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

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
        # 确保技能图已设置
        if self.skill_graph is None:
            raise ValueError("技能图未设置，请先调用set_skill_graph方法")

        device = occupation_features.device
        x = self.skill_graph.x.to(device)
        edge_index = self.skill_graph.edge_index.to(device)

        # 使用技能路径编码器编码技能图
        skill_embeddings = self.skill_encoder(x, edge_index)

        # 获取批次中每个技能的嵌入
        batch_skill_embeddings = skill_embeddings[skill_idx]

        # 使用职业编码器编码职业特征
        occupation_embeddings = self.occupation_encoder(occupation_features)

        # 使用注意力机制增强特征交互
        # 将张量形状调整为注意力层所需的形状 [seq_len, batch_size, embedding_dim]
        batch_skill_embeddings_reshaped = batch_skill_embeddings.unsqueeze(0)  # [1, batch_size, embedding_dim]
        occupation_embeddings_reshaped = occupation_embeddings.unsqueeze(0)  # [1, batch_size, embedding_dim]

        # 计算交互特征
        attn_output, _ = self.cross_attention(
            batch_skill_embeddings_reshaped, occupation_embeddings_reshaped, occupation_embeddings_reshaped
        )

        # 调整形状回 [batch_size, embedding_dim]
        attn_output = attn_output.squeeze(0)

        # 特征增强 - 结合原始特征和交互特征
        enhanced_skill_features = self.feature_enhancement(
            torch.cat([batch_skill_embeddings, attn_output, batch_skill_embeddings * attn_output], dim=1)
        )

        # 连接职业嵌入和技能嵌入
        combined_features = torch.cat([occupation_embeddings, enhanced_skill_features], dim=1)

        # 预测匹配概率
        match_prob = self.matching_predictor(combined_features).squeeze(-1)

        # 如果提供了匹配标签，计算损失
        loss = None
        if match is not None:
            loss = self.loss_fn(match_prob, match)

            # 如果提供了重要性和水平，可以在这里使用它们进行加权
            if importance is not None and level is not None:
                # 根据重要性和水平调整损失权重
                weights = importance * level
                loss = loss * weights
                loss = loss.mean()

        return match_prob, loss


class SkillPathMatchingModel(nn.Module):
    """
    技能路径匹配模型

    扩展基本的技能匹配模型，考虑技能路径和顺序。
    """

    def __init__(
        self,
        skill_input_dim,
        occupation_input_dim,
        hidden_dim=128,
        embedding_dim=64,
        num_gnn_layers=2,
        num_mlp_layers=2,
        dropout=0.1,
        gnn_type="gat",
        focal_alpha=0.25,
        focal_gamma=2.0,
    ):
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
            focal_alpha (float): Focal Loss的alpha参数
            focal_gamma (float): Focal Loss的gamma参数
        """
        super().__init__()

        # 确保embedding_dim是4的倍数，以便于GAT的多头注意力
        if gnn_type == "gat" and embedding_dim % 4 != 0:
            embedding_dim = (embedding_dim // 4 + 1) * 4
            print(f"调整embedding_dim为{embedding_dim}以适应GAT的4个注意力头")

        # 基本技能匹配模型
        self.base_model = SkillMatchingModel(
            skill_input_dim=skill_input_dim,
            occupation_input_dim=occupation_input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_gnn_layers=num_gnn_layers,
            num_mlp_layers=num_mlp_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )

        # 路径注意力层，用于关注技能路径中的重要技能
        self.path_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, dropout=dropout)

        # 路径聚合层
        self.path_aggregation = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, embedding_dim)
        )

        # 最终预测层
        self.final_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 使用Focal Loss
        self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

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
        if hasattr(skill_graph, "edge_attr"):
            skill_graph.edge_attr = skill_graph.edge_attr.to(device)

        # 获取基本技能嵌入
        skill_embeddings = self.base_model.skill_encoder(
            skill_graph.x, skill_graph.edge_index, skill_graph.edge_attr if hasattr(skill_graph, "edge_attr") else None
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
                path_skill_embeddings.unsqueeze(1),
            )

            # 聚合路径嵌入
            path_embedding = self.path_aggregation(attn_output.squeeze(1))

            # 取平均作为最终路径嵌入
            path_embedding = path_embedding.mean(dim=0)

            path_embeddings.append(path_embedding)

        # 将所有路径嵌入堆叠
        path_embeddings = torch.stack(path_embeddings)

        # 将职业嵌入与路径嵌入拼接
        combined_embedding = torch.cat([occupation_embedding.expand(len(path_embeddings), -1), path_embeddings], dim=1)

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
        gnn_type="gat",
        focal_alpha=0.25,
        focal_gamma=2.0,
    )

    return model


def get_device(device_str=None):
    """
    获取可用的计算设备

    参数:
        device_str (str, optional): 指定的设备，如果为None则自动检测

    返回:
        torch.device: 计算设备
    """
    if device_str is not None:
        return torch.device(device_str)

    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


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
        connected_skills = np.random.choice([j for j in range(num_skills) if j != i], size=3, replace=False)
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
