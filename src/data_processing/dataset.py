"""
扩展后的技能匹配数据集处理模块
利用了 OnetProcessor 处理后的全部数据：
- 职位数据（occupations.csv/job_metadata.csv）
- 技能数据（skills.csv/skill_metadata.csv）
- 职业-技能关系（occupation_skills.csv）
- 技术技能（tech_skills.csv）
- 教育要求（education.csv）
- 知识、能力、工作活动、工作情境（knowledge.csv, abilities.csv, work_activities.csv, work_context.csv）
- 技能匹配数据（skill_matching_dataset.csv）
- 技能图（skill_graph.json）
"""

import os
import logging
import json
import random
import numpy as np
import pandas as pd
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
    扩展后的技能匹配数据集类，整合了 OnetProcessor 处理后的全部数据。
    """
    
    # 类属性：共享职业特征投影矩阵和技能特征标准化器
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
            word2vec_path (str): 预训练词向量路径（用于文本嵌入）
            normalize (bool): 是否标准化技能节点特征
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

        # 加载主要数据文件
        self.load_data()
        self.logger.info(f"{split} 数据集初始化完成，样本数：{len(self.data)}")
    
    def load_csv_if_exists(self, filename):
        """辅助方法：加载 CSV 文件（若存在）"""
        path = os.path.join(self.data_dir, filename)
        if os.path.exists(path):
            self.logger.info(f"加载文件：{filename}")
            return pd.read_csv(path, low_memory=False)
        else:
            self.logger.warning(f"文件不存在: {filename}")
            return None

    def load_data(self):
        self.logger.info("开始加载数据...")
        # 主样本数据：技能匹配数据集
        self.data = self.load_csv_if_exists('skill_matching_dataset.csv')
        
        # 加载职业及技能元数据（job_metadata、skill_metadata），优先使用生成的文件
        self.job_metadata = self.load_csv_if_exists('job_metadata.csv')
        self.skill_metadata = self.load_csv_if_exists('skill_metadata.csv')
        
        # 加载扩展数据
        self.occupations_extended = self.load_csv_if_exists('occupations.csv')
        self.occupation_skills = self.load_csv_if_exists('occupation_skills.csv')
        self.tech_skills = self.load_csv_if_exists('tech_skills.csv')
        self.education = self.load_csv_if_exists('education.csv')
        self.knowledge = self.load_csv_if_exists('knowledge.csv')
        self.abilities = self.load_csv_if_exists('abilities.csv')
        self.work_activities = self.load_csv_if_exists('work_activities.csv')
        self.work_context = self.load_csv_if_exists('work_context.csv')
        
        # 加载技能图
        graph_path = os.path.join(self.data_dir, 'skill_graph.json')
        if os.path.exists(graph_path):
            with open(graph_path, 'r') as f:
                self.skill_graph_data = json.load(f)
        else:
            self.logger.error("技能图文件不存在！")
            self.skill_graph_data = {'nodes': [], 'edges': []}
        
        # 构建映射关系：职业代码到索引（基于 job_metadata 优先，如果没有，则 occupations_extended）
        if self.job_metadata is not None and 'occupation_code' in self.job_metadata.columns:
            occ_df = self.job_metadata
        elif self.occupations_extended is not None and 'occupation_code' in self.occupations_extended.columns:
            occ_df = self.occupations_extended
        else:
            occ_df = pd.DataFrame(columns=['occupation_code'])
        self.occupation_code_to_idx = {
            code: i for i, code in enumerate(occ_df['occupation_code'].unique())
        }
        
        # 划分数据集（按职业代码划分，确保各类别均衡）
        self.split_dataset()
        # 构建技能图（用于技能节点特征）
        self.build_skill_graph()

    def split_dataset(self):
        """数据划分：按职业代码进行分层划分"""
        self.logger.info("开始划分数据集...")
        unique_occupations = self.data['occupation_code'].unique()
        np.random.seed(42)
        np.random.shuffle(unique_occupations)
        
        n = len(unique_occupations)
        train_idx = int(n * 0.7)
        val_idx = int(n * 0.85)
        split_occupations = {
            'train': unique_occupations[:train_idx],
            'val': unique_occupations[train_idx:val_idx],
            'test': unique_occupations[val_idx:]
        }
        self.data = self.data[self.data['occupation_code'].isin(split_occupations[self.split])]
        
        # 如果是训练集，进行正负样本平衡
        if self.split == 'train':
            self.balance_samples()

    def balance_samples(self):
        """样本平衡：对正负样本自动采样"""
        pos = self.data[self.data['match'] == 1]
        neg = self.data[self.data['match'] == 0]
        self.logger.info(f"平衡前样本数：正样本 {len(pos)}，负样本 {len(neg)}")
        if abs(len(pos) / (len(neg) + 1e-5) - 1) > 0.1:
            self.data = self.resample(pos, neg)
            self.logger.info(f"平衡后样本总数: {len(self.data)}")

    def resample(self, pos, neg):
        """对少数类进行过采样"""
        from sklearn.utils import resample
        target = max(len(pos), len(neg))
        if len(pos) < len(neg):
            pos = resample(pos, replace=True, n_samples=target, random_state=42)
        else:
            neg = resample(neg, replace=True, n_samples=target, random_state=42)
        return pd.concat([pos, neg]).sample(frac=1, random_state=42)

    def build_skill_graph(self):
        """构建技能图，生成节点特征并构建边数据"""
        self.logger.info("开始构建技能图...")
        # 获取原始节点数据
        raw_nodes = self.skill_graph_data.get('nodes', [])
        
        # 处理节点数据 - 将列表格式 [id, {属性}] 转换为字典格式 {id: id, ...属性}
        processed_nodes = []
        for node_data in raw_nodes:
            if isinstance(node_data, list) and len(node_data) == 2:
                node_id, node_attrs = node_data
                node_dict = {'id': node_id}
                if isinstance(node_attrs, dict):
                    node_dict.update(node_attrs)
                processed_nodes.append(node_dict)
            elif isinstance(node_data, dict) and 'id' in node_data:
                # 如果已经是字典格式，直接使用
                processed_nodes.append(node_data)
        
        # 增加一个“未知技能”节点，确保所有技能都有对应特征
        processed_nodes.append({
            'id': 'UNK', 
            'name': 'Unknown Skill',
            'type': 'unknown'
        })
        self.node_id_to_idx = {node['id']: i for i, node in enumerate(processed_nodes)}
        self.node_idx_to_id = {i: node_id for node_id, i in self.node_id_to_idx.items()}
        num_nodes = len(processed_nodes)
        # 如果使用词向量，维度根据词向量调整，否则默认128维
        feature_dim = self.word2vec.vector_size if self.word2vec else 128
        node_features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        
        for i, node in enumerate(processed_nodes):
            # 使用词向量生成节点特征
            if self.word2vec:
                name = node.get('name', '')
                if name in self.word2vec:
                    node_features[i] = self.word2vec[name]
                else:
                    # 对未知或缺失词汇，使用随机向量并做特殊标记
                    node_features[i] = np.random.normal(loc=0.5, scale=0.1, size=feature_dim)
                    node_features[i][0] = -1  # 标记未知
            else:
                # 不使用词向量时，调用基础特征生成方法
                node_features[i] = self.generate_basic_features(node)
        
        if self.normalize:
            self.skill_scaler = StandardScaler()
            node_features = self.skill_scaler.fit_transform(node_features)
            self.logger.info("完成技能节点特征标准化")
        
        node_features = torch.tensor(node_features, dtype=torch.float)
        # 构建边索引
        edge_index = []
        edge_attr = []
        for edge_data in self.skill_graph_data.get('edges', []):
            # 处理边数据 - 将列表格式 [source_id, target_id, {属性}] 转换为所需格式
            if isinstance(edge_data, list):
                if len(edge_data) >= 2:
                    source_id, target_id = edge_data[0], edge_data[1]
                    # 获取边属性（如果存在）
                    edge_attrs = edge_data[2] if len(edge_data) > 2 and isinstance(edge_data[2], dict) else {}
                else:
                    continue  # 跳过无效的边数据
            elif isinstance(edge_data, dict) and 'source' in edge_data and 'target' in edge_data:
                # 如果已经是字典格式，直接使用
                source_id, target_id = edge_data['source'], edge_data['target']
                edge_attrs = {k: v for k, v in edge_data.items() if k not in ['source', 'target']}
            else:
                continue  # 跳过无效的边数据
                
            # 获取节点索引
            src = self.node_id_to_idx.get(source_id, self.node_id_to_idx['UNK'])
            tgt = self.node_id_to_idx.get(target_id, self.node_id_to_idx['UNK'])
            edge_index.append([src, tgt])
            
            # 提取边权重作为边属性（如果存在）
            weight = edge_attrs.get('weight', 1.0)
            edge_attr.append([float(weight)])
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        # 构建图数据对象
        if edge_attr:
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            self.skill_graph = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_nodes
            )
        else:
            self.skill_graph = Data(
                x=node_features,
                edge_index=edge_index,
                num_nodes=num_nodes
            )
        self.logger.info(f"技能图构建完成：{num_nodes} 个节点，{edge_index.shape[1]} 条边")

    def generate_basic_features(self, node):
        """基础特征生成方法（当不使用词向量时）"""
        features = np.zeros(128)
        # 简单的名称哈希特征
        name_hash = hash(node.get('name', '')) % 100000
        for j in range(16):
            features[j] = (name_hash >> j) & 1
        # 类型 one-hot 编码（假设类别有 technical, business, soft, unknown）
        type_map = {'technical': 0, 'business': 1, 'soft': 2, 'unknown': 3}
        features[16:20] = np.eye(4)[type_map.get(node.get('type', 'unknown'), 3)]
        features[20] = len(node.get('name', '')) / 50  # 名称长度归一化
        features[21] = node.get('importance', 0) / 5
        features[22:] = np.random.normal(0, 0.1, 106)
        return features

    def get_extended_occupation_features(self, occupation_code):
        """
        生成扩展的职业特征向量，整合以下信息：
          1. 基础投影（通过 one-hot +随机投影）
          2. 职位标题与描述文本嵌入（利用词向量求平均）
          3. 教育要求（转为数值）
          4. 技术技能信息（数量及热度平均值）
        """
        # 1. 基础特征（使用共享投影矩阵）
        if SkillMatchingDataset.occupation_projection is None:
            torch.manual_seed(42)
            dim = 128
            num_occ = len(self.occupation_code_to_idx)
            SkillMatchingDataset.occupation_projection = torch.randn(num_occ, dim)
            self.logger.debug("初始化职业投影矩阵")
        idx = self.occupation_code_to_idx.get(occupation_code, -1)
        if idx == -1:
            self.logger.warning(f"未知职业代码: {occupation_code}")
            base_feat = torch.zeros(128)
        else:
            one_hot = torch.zeros(len(self.occupation_code_to_idx))
            one_hot[idx] = 1.0
            base_feat = torch.matmul(one_hot, SkillMatchingDataset.occupation_projection)
        
        # 2. 文本特征：职位标题与描述
        if self.occupations_extended is not None:
            occ_info = self.occupations_extended[self.occupations_extended['occupation_code'] == occupation_code]
            if not occ_info.empty:
                title = occ_info.iloc[0].get('title', '')
                description = occ_info.iloc[0].get('description', '')
            else:
                title, description = '', ''
        else:
            title, description = '', ''
        # 利用词向量求平均嵌入
        def avg_embedding(text, dim):
            tokens = text.split()
            emb_list = []
            if self.word2vec:
                for token in tokens:
                    if token in self.word2vec:
                        emb_list.append(self.word2vec[token])
            if emb_list:
                return np.mean(emb_list, axis=0)
            else:
                return np.zeros(dim)
        vec_dim = self.word2vec.vector_size if self.word2vec else 128
        title_emb = avg_embedding(title, vec_dim)
        desc_emb = avg_embedding(description, vec_dim)
        
        # 3. 教育要求特征：尝试将教育水平转为数值（根据实际映射调整）
        if self.education is not None:
            edu_info = self.education[self.education['occupation_code'] == occupation_code]
            if not edu_info.empty:
                try:
                    edu_level = float(edu_info.iloc[0]['education_level'])
                except:
                    mapping = {
                        'Less than high school': 1,
                        'High school diploma or equivalent': 2,
                        'Some college': 3,
                        "Bachelor's degree": 4,
                        "Master's degree": 5,
                        "Doctoral or professional degree": 6
                    }
                    edu_level = mapping.get(edu_info.iloc[0]['education_level'], 0)
            else:
                edu_level = 0.0
        else:
            edu_level = 0.0
        edu_feat = np.array([edu_level])
        
        # 4. 技术技能特征：统计技术技能数量及平均热度
        if self.tech_skills is not None:
            tech_info = self.tech_skills[self.tech_skills['occupation_code'] == occupation_code]
            if not tech_info.empty:
                tech_count = len(tech_info)
                try:
                    avg_hot = tech_info['is_hot'].astype(float).mean()
                except:
                    avg_hot = 0.0
            else:
                tech_count = 0.0
                avg_hot = 0.0
        else:
            tech_count = 0.0
            avg_hot = 0.0
        tech_feat = np.array([tech_count, avg_hot])
        
        # 将所有特征拼接：基础投影、标题嵌入、描述嵌入、教育、技术技能
        base_feat_np = base_feat.detach().numpy()
        extended_feat = np.concatenate([base_feat_np, title_emb, desc_emb, edu_feat, tech_feat])
        return torch.tensor(extended_feat, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """返回扩展后的样本数据"""
        row = self.data.iloc[idx]
        # 使用扩展版的职业特征
        occ_feat = self.get_extended_occupation_features(row['occupation_code'])
        # 技能节点特征：使用技能图中预构建的映射
        skill_id = row['skill_id']
        skill_idx = self.node_id_to_idx.get(skill_id, self.node_id_to_idx.get('UNK'))
        # 标签及其他数值特征
        match = torch.tensor(row['match'], dtype=torch.float)
        importance = torch.tensor(row['importance'], dtype=torch.float)
        level = torch.tensor(row['level'], dtype=torch.float)
        
        # 数据增强（如果有）
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
    """数据增强：为职业特征添加随机噪声"""
    def __init__(self, noise_level=0.01):
        self.noise_level = noise_level
        
    def __call__(self, occ_feat, skill_idx, match, imp, level):
        occ_feat = occ_feat + torch.randn_like(occ_feat) * self.noise_level
        return occ_feat, skill_idx, match, imp, level

def create_dataloader(data_dir, batch_size=32, split='train', 
                     num_workers=4, **dataset_kwargs):
    """
    创建数据加载器
    
    参数:
        data_dir: 数据目录
        batch_size: 批次大小
        split: 数据集划分 ('train', 'val', 'test')
        num_workers: 并行工作进程数
        **dataset_kwargs: 传递给数据集的其他参数
    """
    dataset = SkillMatchingDataset(data_dir=data_dir, split=split, **dataset_kwargs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    return loader, dataset
