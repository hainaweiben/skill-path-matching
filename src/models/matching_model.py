"""
人才-职位匹配模型
基于深度学习的人才与职位匹配模型，结合能力点路径信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

class DeepMatchingModel(nn.Module):
    """深度匹配模型基类"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        """
        初始化模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
        """
        super(DeepMatchingModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # 构建多层感知机
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征
            
        Returns:
            匹配分数
        """
        x = self.mlp(x)
        x = self.output_layer(x)
        return self.sigmoid(x)


class DualEncoderMatchingModel(nn.Module):
    """双编码器匹配模型，分别编码人才和职位信息"""
    
    def __init__(self, resume_dim: int, job_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        """
        初始化模型
        
        Args:
            resume_dim: 简历特征维度
            job_dim: 职位特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出嵌入维度
        """
        super(DualEncoderMatchingModel, self).__init__()
        
        # 简历编码器
        self.resume_encoder = nn.Sequential(
            nn.Linear(resume_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 职位编码器
        self.job_encoder = nn.Sequential(
            nn.Linear(job_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, resume_features, job_features):
        """
        前向传播
        
        Args:
            resume_features: 简历特征
            job_features: 职位特征
            
        Returns:
            匹配分数
        """
        # 编码简历和职位
        resume_embedding = self.resume_encoder(resume_features)
        job_embedding = self.job_encoder(job_features)
        
        # 归一化嵌入向量
        resume_embedding = F.normalize(resume_embedding, p=2, dim=1)
        job_embedding = F.normalize(job_embedding, p=2, dim=1)
        
        # 计算余弦相似度作为匹配分数
        similarity = torch.sum(resume_embedding * job_embedding, dim=1, keepdim=True)
        
        return (similarity + 1) / 2  # 将[-1, 1]范围映射到[0, 1]


class SkillPathAwareMatchingModel(nn.Module):
    """能力点路径感知的匹配模型"""
    
    def __init__(self, resume_dim: int, job_dim: int, skill_graph_dim: int, 
                 hidden_dim: int = 128, output_dim: int = 64):
        """
        初始化模型
        
        Args:
            resume_dim: 简历特征维度
            job_dim: 职位特征维度
            skill_graph_dim: 技能图特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出嵌入维度
        """
        super(SkillPathAwareMatchingModel, self).__init__()
        
        # 简历编码器
        self.resume_encoder = nn.Sequential(
            nn.Linear(resume_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 职位编码器
        self.job_encoder = nn.Sequential(
            nn.Linear(job_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 技能图特征编码器
        self.skill_graph_encoder = nn.Sequential(
            nn.Linear(skill_graph_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(output_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1)
        )
        
        # 匹配层
        self.matching_layer = nn.Sequential(
            nn.Linear(output_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, resume_features, job_features, skill_graph_features):
        """
        前向传播
        
        Args:
            resume_features: 简历特征
            job_features: 职位特征
            skill_graph_features: 技能图特征
            
        Returns:
            匹配分数
        """
        # 编码特征
        resume_embedding = self.resume_encoder(resume_features)
        job_embedding = self.job_encoder(job_features)
        skill_graph_embedding = self.skill_graph_encoder(skill_graph_features)
        
        # 归一化嵌入向量
        resume_embedding = F.normalize(resume_embedding, p=2, dim=1)
        job_embedding = F.normalize(job_embedding, p=2, dim=1)
        skill_graph_embedding = F.normalize(skill_graph_embedding, p=2, dim=1)
        
        # 计算简历-职位相似度
        resume_job_similarity = torch.sum(resume_embedding * job_embedding, dim=1, keepdim=True)
        
        # 计算简历-技能图相似度
        resume_skill_similarity = torch.sum(resume_embedding * skill_graph_embedding, dim=1, keepdim=True)
        
        # 计算职位-技能图相似度
        job_skill_similarity = torch.sum(job_embedding * skill_graph_embedding, dim=1, keepdim=True)
        
        # 连接所有特征
        combined_features = torch.cat([
            resume_embedding, 
            job_embedding, 
            skill_graph_embedding
        ], dim=1)
        
        # 使用注意力机制融合特征
        attention_weights = self.attention(combined_features)
        attention_weights = attention_weights.unsqueeze(2)  # [batch_size, 3, 1]
        
        embeddings = torch.stack([resume_embedding, job_embedding, skill_graph_embedding], dim=1)  # [batch_size, 3, output_dim]
        
        weighted_embedding = torch.sum(embeddings * attention_weights, dim=1)  # [batch_size, output_dim]
        
        # 连接所有特征和相似度分数
        final_features = torch.cat([
            weighted_embedding,
            resume_embedding * job_embedding,  # 元素级乘法，捕捉交互特征
            resume_embedding * skill_graph_embedding,
            job_embedding * skill_graph_embedding
        ], dim=1)
        
        # 计算最终匹配分数
        matching_score = self.matching_layer(final_features)
        
        return matching_score


class CareerPathAwareMatchingModel(nn.Module):
    """职业路径感知的匹配模型"""
    
    def __init__(self, resume_dim: int, job_dim: int, career_path_dim: int, 
                 hidden_dim: int = 128, output_dim: int = 64):
        """
        初始化模型
        
        Args:
            resume_dim: 简历特征维度
            job_dim: 职位特征维度
            career_path_dim: 职业路径特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出嵌入维度
        """
        super(CareerPathAwareMatchingModel, self).__init__()
        
        # 简历编码器
        self.resume_encoder = nn.Sequential(
            nn.Linear(resume_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 职位编码器
        self.job_encoder = nn.Sequential(
            nn.Linear(job_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 职业路径编码器（使用LSTM捕捉序列信息）
        self.career_path_encoder = nn.LSTM(
            input_size=career_path_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # LSTM输出转换
        self.career_path_fc = nn.Linear(hidden_dim * 2, output_dim)  # 双向LSTM，所以是hidden_dim*2
        
        # 匹配层
        self.matching_layer = nn.Sequential(
            nn.Linear(output_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, resume_features, job_features, career_path_features, career_path_lengths):
        """
        前向传播
        
        Args:
            resume_features: 简历特征
            job_features: 职位特征
            career_path_features: 职业路径特征序列，形状为[batch_size, seq_len, career_path_dim]
            career_path_lengths: 每个样本的职业路径长度
            
        Returns:
            匹配分数
        """
        # 编码简历和职位
        resume_embedding = self.resume_encoder(resume_features)
        job_embedding = self.job_encoder(job_features)
        
        # 对职业路径序列进行打包，处理变长序列
        packed_career_path = nn.utils.rnn.pack_padded_sequence(
            career_path_features, career_path_lengths, batch_first=True, enforce_sorted=False
        )
        
        # 编码职业路径
        _, (hidden, _) = self.career_path_encoder(packed_career_path)
        
        # 获取最后一层的隐藏状态，并合并双向结果
        hidden = hidden[-2:].transpose(0, 1).contiguous().view(-1, hidden.size(2) * 2)
        career_path_embedding = self.career_path_fc(hidden)
        
        # 归一化嵌入向量
        resume_embedding = F.normalize(resume_embedding, p=2, dim=1)
        job_embedding = F.normalize(job_embedding, p=2, dim=1)
        career_path_embedding = F.normalize(career_path_embedding, p=2, dim=1)
        
        # 连接所有特征
        combined_features = torch.cat([
            resume_embedding, 
            job_embedding, 
            career_path_embedding
        ], dim=1)
        
        # 计算最终匹配分数
        matching_score = self.matching_layer(combined_features)
        
        return matching_score
