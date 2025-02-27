"""
数据处理工具模块
提供数据加载、预处理和特征提取等功能
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class SkillMatchingDataset(Dataset):
    """技能匹配数据集类"""
    
    def __init__(self, resume_features: np.ndarray, job_features: np.ndarray, 
                 skill_graph_features: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None):
        """
        初始化数据集
        
        Args:
            resume_features: 简历特征矩阵
            job_features: 职位特征矩阵
            skill_graph_features: 技能图特征矩阵（可选）
            labels: 标签（可选）
        """
        self.resume_features = torch.FloatTensor(resume_features)
        self.job_features = torch.FloatTensor(job_features)
        
        if skill_graph_features is not None:
            self.skill_graph_features = torch.FloatTensor(skill_graph_features)
        else:
            self.skill_graph_features = None
        
        if labels is not None:
            self.labels = torch.FloatTensor(labels)
        else:
            self.labels = None
        
        self.has_skill_graph = skill_graph_features is not None
        self.has_labels = labels is not None
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.resume_features)
    
    def __getitem__(self, idx):
        """获取数据样本"""
        sample = {
            'resume_features': self.resume_features[idx],
            'job_features': self.job_features[idx]
        }
        
        if self.has_skill_graph:
            sample['skill_graph_features'] = self.skill_graph_features[idx]
        
        if self.has_labels:
            sample['label'] = self.labels[idx]
        
        return sample


def load_data(data_path: str) -> pd.DataFrame:
    """
    加载数据
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        数据DataFrame
    """
    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    elif data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
        return pd.read_excel(data_path)
    else:
        raise ValueError(f"不支持的文件格式: {data_path}")


def extract_text_features(texts: List[str], max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    从文本中提取TF-IDF特征
    
    Args:
        texts: 文本列表
        max_features: 最大特征数
        ngram_range: n-gram范围
        
    Returns:
        特征矩阵和TF-IDF向量化器
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    features = vectorizer.fit_transform(texts)
    return features, vectorizer


def extract_categorical_features(categories: List[Any], one_hot: bool = True) -> Tuple[np.ndarray, Optional[OneHotEncoder]]:
    """
    从类别数据中提取特征
    
    Args:
        categories: 类别数据列表
        one_hot: 是否进行独热编码
        
    Returns:
        特征矩阵和编码器（如果使用独热编码）
    """
    categories = np.array(categories).reshape(-1, 1)
    
    if one_hot:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        features = encoder.fit_transform(categories)
        return features, encoder
    else:
        # 将类别转换为整数
        unique_categories = {cat: i for i, cat in enumerate(set(categories.flatten()))}
        features = np.array([unique_categories.get(cat, -1) for cat in categories.flatten()]).reshape(-1, 1)
        return features, None


def extract_numerical_features(values: List[float], normalize: bool = True) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    """
    从数值数据中提取特征
    
    Args:
        values: 数值列表
        normalize: 是否进行标准化
        
    Returns:
        特征矩阵和标准化器（如果进行标准化）
    """
    values = np.array(values).reshape(-1, 1)
    
    if normalize:
        scaler = StandardScaler()
        features = scaler.fit_transform(values)
        return features, scaler
    else:
        return values, None


def combine_features(feature_list: List[np.ndarray]) -> np.ndarray:
    """
    组合多个特征矩阵
    
    Args:
        feature_list: 特征矩阵列表
        
    Returns:
        组合后的特征矩阵
    """
    # 确保所有特征矩阵的样本数相同
    n_samples = feature_list[0].shape[0]
    assert all(f.shape[0] == n_samples for f in feature_list), "所有特征矩阵的样本数必须相同"
    
    # 将稀疏矩阵转换为密集矩阵
    dense_features = []
    for features in feature_list:
        if hasattr(features, 'toarray'):
            dense_features.append(features.toarray())
        else:
            dense_features.append(features)
    
    # 水平拼接特征
    return np.hstack(dense_features)


def create_data_loaders(dataset: Dataset, batch_size: int = 32, 
                        train_ratio: float = 0.7, val_ratio: float = 0.15,
                        random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批量大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        random_seed: 随机种子
        
    Returns:
        训练、验证和测试数据加载器
    """
    # 设置随机种子
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # 计算数据集划分
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    train_end = int(train_ratio * dataset_size)
    val_end = train_end + int(val_ratio * dataset_size)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # 创建数据子集
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


def save_model(model: torch.nn.Module, save_path: str, metadata: Dict = None):
    """
    保存模型
    
    Args:
        model: PyTorch模型
        save_path: 保存路径
        metadata: 元数据（可选）
    """
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 准备保存数据
    save_data = {
        'model_state_dict': model.state_dict()
    }
    
    if metadata:
        save_data['metadata'] = metadata
    
    # 保存模型
    torch.save(save_data, save_path)


def load_model(model: torch.nn.Module, load_path: str) -> Tuple[torch.nn.Module, Optional[Dict]]:
    """
    加载模型
    
    Args:
        model: PyTorch模型
        load_path: 加载路径
        
    Returns:
        加载后的模型和元数据（如果有）
    """
    # 加载模型数据
    checkpoint = torch.load(load_path)
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 返回模型和元数据
    return model, checkpoint.get('metadata')
