"""
技能匹配数据集处理模块

"""

import os
import logging
import json
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from gensim.models import KeyedVectors

# 配置日志系统
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class SkillMatchingDataset(Dataset):
    """
    技能匹配数据集类
    
    """
    
    # 类属性用于共享投影矩阵
    occupation_projection = None
    skill_scaler = None

    def __init__(self, data_dir, split='train', 
                 word2vec_path=None, normalize=True,
                 transform=None, log_level=logging.INFO):
        """
        初始化数据集
        
        参数:
            data_dir (str): 数据目录
            split (str): 数据集划分，可选 'train', 'val', 'test'
            word2vec_path (str): 预训练词向量路径
            normalize (bool): 是否标准化特征
            transform (callable): 数据增强变换
            log_level: 日志级别
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.normalize = normalize
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        # 初始化词向量模型
        self.word2vec = None
        if word2vec_path and os.path.exists(word2vec_path):
            try:
                self.word2vec = KeyedVectors.load(word2vec_path)
                self.logger.info(f"成功加载词向量模型：{word2vec_path}")
            except Exception as e:
                self.logger.error(f"加载词向量失败：{str(e)}")

        # 加载数据
        self.load_data()
        self.logger.info(f"{split} 数据集初始化完成，样本数：{len(self.data)}")

    def load_data(self):
        """优化后的数据加载方法"""
        # 使用低内存模式读取数据
        self.logger.info("开始加载数据...")
        
        # 技能匹配数据
        dataset_path = os.path.join(self.data_dir, 'skill_matching_dataset.csv')
        self.data = pd.read_csv(dataset_path, low_memory=False)
        
        # 元数据分块读取
        chunksize = 1000  # 根据实际情况调整
        occupation_path = os.path.join(self.data_dir, 'job_metadata.csv')
        self.occupations = pd.concat([chunk for chunk in pd.read_csv(occupation_path, chunksize=chunksize)])
        
        skill_path = os.path.join(self.data_dir, 'skill_metadata.csv')
        self.skills = pd.concat([chunk for chunk in pd.read_csv(skill_path, chunksize=chunksize)])
        
        # 加载技能图
        graph_path = os.path.join(self.data_dir, 'skill_graph.json')
        with open(graph_path, 'r') as f:
            self.skill_graph_data = json.load(f)
        
        # 构建映射关系
        self.occupation_code_to_idx = {
            code: i for i, code in enumerate(self.occupations['occupation_code'])
        }
        
        # 数据预处理
        self.split_dataset()
        self.build_skill_graph()

    def split_dataset(self):
        """改进的数据划分方法"""
        self.logger.info("开始划分数据集...")
        
        # 分层抽样保证各类别比例
        unique_occupations = self.data['occupation_code'].unique()
        np.random.seed(42)
        np.random.shuffle(unique_occupations)
        
        # 动态划分比例
        n = len(unique_occupations)
        train_idx = int(n * 0.7)
        val_idx = int(n * 0.85)
        
        # 划分职业集合
        split_occupations = {
            'train': unique_occupations[:train_idx],
            'val': unique_occupations[train_idx:val_idx],
            'test': unique_occupations[val_idx:]
        }
        self.data = self.data[self.data['occupation_code'].isin(split_occupations[self.split])]
        
        # 训练集进行样本平衡
        if self.split == 'train':
            self.balance_samples()

    def balance_samples(self):
        """改进的样本平衡方法"""
        pos = self.data[self.data['match'] == 1]
        neg = self.data[self.data['match'] == 0]
        self.logger.info(f"正样本数: {len(pos)}, 负样本数: {len(neg)}")

        # 自动平衡逻辑
        if abs(len(pos)/len(neg) - 1) > 0.1:
            self.data = self.resample(pos, neg)
            self.logger.info(f"平衡后样本数: {len(self.data)}")

    def resample(self, pos, neg):
        """优化的过采样方法"""
        from sklearn.utils import resample
        
        # 确定目标数量
        target = max(len(pos), len(neg))
        
        # 对少数类进行过采样
        if len(pos) < len(neg):
            pos = resample(pos, replace=True, n_samples=target, random_state=42)
        else:
            neg = resample(neg, replace=True, n_samples=target, random_state=42)
        
        return pd.concat([pos, neg]).sample(frac=1, random_state=42)

    def build_skill_graph(self):
        """改进的技能图构建方法"""
        self.logger.info("开始构建技能图...")
        
        # 添加未知节点
        nodes = self.skill_graph_data['nodes'] + [{
            'id': 'UNK', 
            'name': 'Unknown Skill',
            'type': 'unknown'
        }]
        
        # 创建映射关系
        self.node_id_to_idx = {node['id']: i for i, node in enumerate(nodes)}
        self.node_idx_to_id = {i: node_id for node_id, i in self.node_id_to_idx.items()}
        
        # 初始化节点特征
        num_nodes = len(nodes)
        feature_dim = 300 if self.word2vec else 128  # 根据是否使用词向量调整维度
        node_features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        
        # 生成节点特征
        for i, node in enumerate(nodes):
            # 优先使用词向量
            if self.word2vec:
                name = node.get('name', '')
                if name in self.word2vec:
                    node_features[i] = self.word2vec[name]
                else:
                    # 使用随机向量+特殊标记
                    node_features[i] = np.random.normal(loc=0.5, scale=0.1, size=feature_dim)
                    node_features[i][0] = -1  # 标记未知词汇
            else:
                # 原始特征生成逻辑
                node_features[i] = self.generate_basic_features(node)
        
        # 特征标准化
        if self.normalize:
            self.skill_scaler = StandardScaler()
            node_features = self.skill_scaler.fit_transform(node_features)
            self.logger.info("已完成节点特征标准化")
        
        # 转换为Tensor
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        # 构建边数据
        edge_index = []
        for edge in self.skill_graph_data['edges']:
            src = self.node_id_to_idx.get(edge['source'], self.node_id_to_idx['UNK'])
            tgt = self.node_id_to_idx.get(edge['target'], self.node_id_to_idx['UNK'])
            edge_index.append([src, tgt])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # 创建图数据对象
        self.skill_graph = Data(
            x=node_features,
            edge_index=edge_index,
            num_nodes=num_nodes
        )
        self.logger.info(f"技能图构建完成：{num_nodes}节点，{edge_index.shape[1]}边")

    def generate_basic_features(self, node):
        """基础特征生成方法（当不使用词向量时）"""
        features = np.zeros(128)
        
        # 名称哈希特征
        name_hash = hash(node.get('name', '')) % 100000
        for j in range(16):
            features[j] = (name_hash >> j) & 1
        
        # 类型编码
        type_map = {'technical': 0, 'business': 1, 'soft': 2, 'unknown': 3}
        features[16:20] = np.eye(4)[type_map.get(node.get('type', 'unknown'), 3)]
        
        # 其他统计特征
        features[20] = len(node.get('name', '')) / 50  # 名称长度归一化
        features[21] = node.get('importance', 0) / 5
        
        # 剩余维度使用随机初始化
        features[22:] = np.random.normal(0, 0.1, 106)
        return features

    def get_occupation_features(self, occupation_code):
        """优化的职业特征生成方法"""
        # 投影矩阵初始化
        if SkillMatchingDataset.occupation_projection is None:
            torch.manual_seed(42)
            dim = 128
            SkillMatchingDataset.occupation_projection = torch.randn(
                len(self.occupation_code_to_idx), dim
            )
            self.logger.debug("初始化职业特征投影矩阵")
        
        # 生成one-hot编码
        idx = self.occupation_code_to_idx.get(occupation_code, -1)
        if idx == -1:
            self.logger.warning(f"未知职业代码: {occupation_code}")
            return torch.zeros(128)
        
        one_hot = torch.zeros(len(self.occupation_code_to_idx))
        one_hot[idx] = 1.0
        
        # 投影到低维空间
        return torch.matmul(one_hot, SkillMatchingDataset.occupation_projection)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """改进的数据获取方法"""
        row = self.data.iloc[idx]
        
        # 职业特征
        occ_feat = self.get_occupation_features(row['occupation_code'])
        
        # 技能节点处理
        skill_id = row['skill_id']
        skill_idx = self.node_id_to_idx.get(skill_id, self.node_id_to_idx['UNK'])
        
        # 标签和元数据
        match = torch.tensor(row['match'], dtype=torch.float)
        importance = torch.tensor(row['importance'], dtype=torch.float)
        level = torch.tensor(row['level'], dtype=torch.float)
        
        # 数据增强
        if self.transform:
            occ_feat, skill_idx, match, importance, level = self.transform(
                occ_feat, skill_idx, match, importance, level
            )
        
        return (occ_feat, 
                torch.tensor(skill_idx, dtype=torch.long),
                match,
                importance,
                level)

class NoiseAugmentation:
    """数据增强：添加随机噪声"""
    def __init__(self, noise_level=0.01):
        self.noise_level = noise_level
        
    def __call__(self, occ_feat, skill_idx, match, imp, level):
        # 对职业特征添加噪声
        occ_feat += torch.randn_like(occ_feat) * self.noise_level
        return occ_feat, skill_idx, match, imp, level

def create_dataloader(data_dir, batch_size=32, split='train', 
                     num_workers=4, **dataset_kwargs):
    """
    改进的数据加载器创建函数
    
    参数:
        data_dir: 数据目录
        batch_size: 批次大小
        split: 数据集划分
        num_workers: 并行工作进程数
        **dataset_kwargs: 传递给数据集的参数
    """
    dataset = SkillMatchingDataset(data_dir=data_dir, split=split, **dataset_kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    ), dataset