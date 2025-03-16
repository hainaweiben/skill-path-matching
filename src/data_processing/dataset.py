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

import json
import logging
import os

import community as community_louvain
import networkx as nx
import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

# 配置日志系统
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)


class SkillMatchingDataset(Dataset):
    """
    扩展后的技能匹配数据集类，整合了 OnetProcessor 处理后的全部数据。
    """

    # 类属性：共享职业特征投影矩阵和技能特征标准化器
    occupation_projection = None
    skill_scaler = None

    def __init__(
        self, data_dir, split="train", word2vec_path=None, normalize=True, transform=None, log_level=logging.INFO
    ):
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
                # 检测文件格式并使用相应的加载方法
                # 通过文件头部判断文件类型
                with open(word2vec_path, "rb") as f:
                    header = f.read(2)
                    f.seek(0)  # 重置文件指针

                    if header.startswith(b"\x80\x04"):  # pickle 协议4标记
                        self.word2vec = KeyedVectors.load(word2vec_path)
                        self.logger.info(f"成功加载 pickle 格式词向量模型：{word2vec_path}")
                    else:  # 假设是原始 word2vec 二进制格式
                        self.word2vec = KeyedVectors.load_word2vec_format(
                            word2vec_path, binary=True, unicode_errors="ignore"
                        )
                        self.logger.info(f"成功加载原始 word2vec 格式模型：{word2vec_path}")
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
        self.data = self.load_csv_if_exists("skill_matching_dataset.csv")

        # 加载职业及技能元数据（job_metadata、skill_metadata），优先使用生成的文件
        self.job_metadata = self.load_csv_if_exists("job_metadata.csv")
        self.skill_metadata = self.load_csv_if_exists("skill_metadata.csv")

        # 加载扩展数据
        self.occupations_extended = self.load_csv_if_exists("occupations.csv")
        self.occupation_skills = self.load_csv_if_exists("occupation_skills.csv")
        self.tech_skills = self.load_csv_if_exists("tech_skills.csv")
        self.education = self.load_csv_if_exists("education.csv")
        self.knowledge = self.load_csv_if_exists("knowledge.csv")
        self.abilities = self.load_csv_if_exists("abilities.csv")
        self.work_activities = self.load_csv_if_exists("work_activities.csv")
        self.work_context = self.load_csv_if_exists("work_context.csv")

        # 加载技能图
        graph_path = os.path.join(self.data_dir, "skill_graph.json")
        if os.path.exists(graph_path):
            with open(graph_path) as f:
                self.skill_graph_data = json.load(f)
        else:
            self.logger.error("技能图文件不存在！")
            self.skill_graph_data = {"nodes": [], "edges": []}

        # 构建映射关系：职业代码到索引（基于 job_metadata 优先，如果没有，则 occupations_extended）
        if self.job_metadata is not None and "occupation_code" in self.job_metadata.columns:
            occ_df = self.job_metadata
        elif self.occupations_extended is not None and "occupation_code" in self.occupations_extended.columns:
            occ_df = self.occupations_extended
        else:
            occ_df = pd.DataFrame(columns=["occupation_code"])
        self.occupation_code_to_idx = {code: i for i, code in enumerate(occ_df["occupation_code"].unique())}

        # 划分数据集（按职业代码划分，确保各类别均衡）
        self.split_dataset()
        # 构建技能图（用于技能节点特征）
        self.build_skill_graph()

    def split_dataset(self):
        """数据划分：按职业代码进行分层划分"""
        self.logger.info("开始划分数据集...")
        unique_occupations = self.data["occupation_code"].unique()
        np.random.seed(42)
        np.random.shuffle(unique_occupations)

        n = len(unique_occupations)
        train_idx = int(n * 0.7)
        val_idx = int(n * 0.85)
        split_occupations = {
            "train": unique_occupations[:train_idx],
            "val": unique_occupations[train_idx:val_idx],
            "test": unique_occupations[val_idx:],
        }
        self.data = self.data[self.data["occupation_code"].isin(split_occupations[self.split])]

        # 如果是训练集，进行正负样本平衡
        if self.split == "train":
            self.balance_samples()

    def balance_samples(self):
        """样本平衡：对正负样本自动采样"""
        pos = self.data[self.data["match"] == 1]
        neg = self.data[self.data["match"] == 0]
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

    def _process_nodes(self, raw_nodes):
        """处理原始节点数据，将其转换为标准格式"""
        processed_nodes = []
        for node_data in raw_nodes:
            if isinstance(node_data, list) and len(node_data) == 2:
                node_id, node_attrs = node_data
                node_dict = {"id": node_id}
                if isinstance(node_attrs, dict):
                    node_dict.update(node_attrs)
                processed_nodes.append(node_dict)
            elif isinstance(node_data, dict) and "id" in node_data:
                # 如果已经是字典格式，直接使用
                processed_nodes.append(node_data)

        # 增加一个"未知技能"节点，确保所有技能都有对应特征
        processed_nodes.append({"id": "UNK", "name": "Unknown Skill", "type": "unknown"})
        return processed_nodes

    def _create_node_mappings(self, processed_nodes):
        """创建节点ID和索引之间的映射"""
        node_id_to_idx = {node["id"]: i for i, node in enumerate(processed_nodes)}
        node_idx_to_id = {i: node_id for node_id, i in node_id_to_idx.items()}
        return node_id_to_idx, node_idx_to_id, len(processed_nodes)

    def _generate_node_features(self, processed_nodes, num_nodes):
        """为节点生成特征向量"""
        # 如果使用词向量，维度根据词向量调整，否则默认128维
        feature_dim = 128
        if self.word2vec is not None:
            feature_dim = self.word2vec.vector_size

        node_features = np.zeros((num_nodes, feature_dim), dtype=np.float32)

        for i, node in enumerate(processed_nodes):
            node_features[i] = self._get_node_feature_vector(node, feature_dim)

        if self.normalize:
            self.skill_scaler = StandardScaler()
            node_features = self.skill_scaler.fit_transform(node_features)
            self.logger.info("完成技能节点特征标准化")

        return torch.tensor(node_features, dtype=torch.float)

    def _get_node_feature_vector(self, node, feature_dim):
        """为单个节点生成特征向量"""
        # 使用词向量生成节点特征
        name = node.get("name", "")
        # 将名称转换为小写并分词，以提高匹配率
        name_tokens = name.lower().split()

        if self.word2vec is not None and name_tokens:
            # 尝试获取所有可用词的词向量平均值
            valid_vectors = self._get_valid_word_vectors(name_tokens)

            if valid_vectors:
                return np.mean(valid_vectors, axis=0)

        # 对未知或缺失词汇，使用随机向量并做特殊标记
        feature_vector = np.random.normal(loc=0.5, scale=0.1, size=feature_dim)
        feature_vector[0] = -1  # 标记未知
        return feature_vector

    def _get_valid_word_vectors(self, tokens):
        """获取有效的词向量"""
        valid_vectors = []
        for token in tokens:
            try:
                if token in self.word2vec:
                    valid_vectors.append(self.word2vec[token])
            except KeyError:
                continue
        return valid_vectors

    def _process_edges(self):
        """处理边数据，生成边索引和边属性"""
        edge_index = []
        edge_attr = []

        for edge_data in self.skill_graph_data.get("edges", []):
            source_id, target_id, edge_attrs = self._parse_edge_data(edge_data)
            if source_id is None or target_id is None:
                continue  # 跳过无效的边数据

            # 获取节点索引
            src = self.node_id_to_idx.get(source_id, self.node_id_to_idx["UNK"])
            tgt = self.node_id_to_idx.get(target_id, self.node_id_to_idx["UNK"])
            edge_index.append([src, tgt])

            # 提取边权重作为边属性
            weight = edge_attrs.get("weight", 1.0)
            edge_attr.append([float(weight)])

        return edge_index, edge_attr

    def _parse_edge_data(self, edge_data):
        """解析边数据，返回源节点ID、目标节点ID和边属性"""
        if isinstance(edge_data, list):
            if len(edge_data) >= 2:
                source_id, target_id = edge_data[0], edge_data[1]
                # 获取边属性（如果存在）
                edge_attrs = edge_data[2] if len(edge_data) > 2 and isinstance(edge_data[2], dict) else {}
                return source_id, target_id, edge_attrs
        elif isinstance(edge_data, dict) and "source" in edge_data and "target" in edge_data:
            # 如果已经是字典格式，直接使用
            source_id, target_id = edge_data["source"], edge_data["target"]
            edge_attrs = {k: v for k, v in edge_data.items() if k not in ["source", "target"]}
            return source_id, target_id, edge_attrs

        # 如果数据格式无效，返回 None
        return None, None, {}

    def _create_graph_data(self, node_features, edge_index, edge_attr, num_nodes):
        """创建图数据对象"""
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        if edge_attr:
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
        else:
            return Data(x=node_features, edge_index=edge_index, num_nodes=num_nodes)

    def build_skill_graph(self):
        """构建技能图，生成节点特征并构建边数据"""
        self.logger.info("开始构建技能图...")

        # 1. 处理节点数据
        raw_nodes = self.skill_graph_data.get("nodes", [])
        processed_nodes = self._process_nodes(raw_nodes)

        # 2. 创建节点映射
        self.node_id_to_idx, self.node_idx_to_id, num_nodes = self._create_node_mappings(processed_nodes)

        # 3. 生成节点特征
        node_features = self._generate_node_features(processed_nodes, num_nodes)

        # 4. 处理边数据
        edge_index, edge_attr = self._process_edges()

        # 5. 创建图数据对象
        self.skill_graph = self._create_graph_data(node_features, edge_index, edge_attr, num_nodes)

        self.logger.info(f"技能图构建完成：{num_nodes} 个节点，{len(edge_index)} 条边")
        self.logger.info("Basic skill graph construction completed successfully")

        # Convert to NetworkX graph for advanced feature engineering
        self.logger.info("Converting to NetworkX graph for advanced feature engineering...")
        nx_graph = nx.Graph()

        # Add nodes
        for i, node in enumerate(processed_nodes):
            nx_graph.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})

        # Add edges
        for edge_data in self.skill_graph_data.get("edges", []):
            if isinstance(edge_data, list) and len(edge_data) >= 2:
                source_id, target_id = edge_data[0], edge_data[1]
                # Get edge weight if available
                weight = 1.0
                if len(edge_data) > 2 and isinstance(edge_data[2], dict) and "weight" in edge_data[2]:
                    weight = float(edge_data[2]["weight"])
                nx_graph.add_edge(source_id, target_id, weight=weight)
            elif isinstance(edge_data, dict) and "source" in edge_data and "target" in edge_data:
                source_id, target_id = edge_data["source"], edge_data["target"]
                weight = float(edge_data.get("weight", 1.0))
                nx_graph.add_edge(source_id, target_id, weight=weight)

        # Generate enhanced node features
        node_ids = [node["id"] for node in processed_nodes]
        centrality_features = self.enhance_node_features(nx_graph, node_ids)

        # Store centrality features in a dictionary for easy access
        self.node_centrality_features = {}
        for i, node_id in enumerate(node_ids):
            self.node_centrality_features[node_id] = centrality_features[i]

        # Generate community features
        self.community_features = self.generate_community_features(nx_graph)

        # Enhance the original node features with centrality and community features
        enhanced_node_features = torch.zeros((num_nodes, node_features.shape[1] + 9), dtype=torch.float)

        # Copy original word vector features
        enhanced_node_features[:, : node_features.shape[1]] = node_features

        # Add centrality and community features
        for i, node in enumerate(processed_nodes):
            node_id = node["id"]
            # Add centrality features
            if node_id in self.node_centrality_features:
                enhanced_node_features[i, node_features.shape[1] : node_features.shape[1] + 6] = torch.tensor(
                    self.node_centrality_features[node_id], dtype=torch.float
                )

            # Add community features
            if node_id in self.community_features:
                comm_features = self.community_features[node_id]
                # Normalize community ID
                enhanced_node_features[i, node_features.shape[1] + 6] = comm_features["community_id"] / max(
                    1, self.max_community_id
                )
                enhanced_node_features[i, node_features.shape[1] + 7] = comm_features["community_size"] / len(
                    processed_nodes
                )
                enhanced_node_features[i, node_features.shape[1] + 8] = comm_features["normalized_size"]

        # Update the graph with enhanced features
        if hasattr(self.skill_graph, "edge_attr"):
            self.skill_graph = Data(
                x=enhanced_node_features,
                edge_index=self.skill_graph.edge_index,
                edge_attr=self.skill_graph.edge_attr,
                num_nodes=num_nodes,
            )
        else:
            self.skill_graph = Data(
                x=enhanced_node_features, edge_index=self.skill_graph.edge_index, num_nodes=num_nodes
            )

        self.logger.info(
            "Enhanced node features with centrality and community features.\n"
            f"New feature dimension: {enhanced_node_features.shape[1]}"
        )

    def enhance_node_features(self, graph, node_ids):
        """
        Generate enhanced node centrality features from graph structure

        Args:
            graph (networkx.Graph): The skill graph
            node_ids (list): List of node IDs to generate features for

        Returns:
            np.ndarray: Array of enhanced node features
        """
        # Calculate various centrality metrics
        self.logger.info("Computing node centrality features...")
        pagerank = nx.pagerank(graph)
        betweenness = nx.betweenness_centrality(graph)
        clustering = nx.clustering(graph)
        degree_centrality = nx.degree_centrality(graph)
        closeness_centrality = nx.closeness_centrality(graph)

        # Get node degrees
        degrees = dict(graph.degree())
        max_degree = max(degrees.values()) if degrees else 1

        enhanced_features = []
        for node_id in node_ids:
            node_features = [
                pagerank.get(node_id, 0),
                betweenness.get(node_id, 0),
                clustering.get(node_id, 0),
                degree_centrality.get(node_id, 0),
                closeness_centrality.get(node_id, 0),
                degrees.get(node_id, 0) / max_degree,
            ]
            enhanced_features.append(node_features)

        self.logger.info(f"Generated centrality features for {len(node_ids)} nodes")
        return np.array(enhanced_features)

    def generate_community_features(self, graph):
        """
        Generate community features using Louvain community detection algorithm

        Args:
            graph (networkx.Graph): The skill graph

        Returns:
            dict: Dictionary mapping node IDs to community features
        """
        self.logger.info("Detecting skill communities using Louvain algorithm...")
        # Apply Louvain community detection
        communities = community_louvain.best_partition(graph)

        # Count community sizes
        community_sizes = {}
        for community_id in communities.values():
            community_sizes[community_id] = community_sizes.get(community_id, 0) + 1

        # Calculate max community size for normalization
        max_community_size = max(community_sizes.values()) if community_sizes else 1
        self.max_community_id = max(communities.values()) if communities else 0

        # Generate community features for each node
        community_features = {}
        for node_id, community_id in communities.items():
            community_size = community_sizes.get(community_id, 0)
            community_features[node_id] = {
                "community_id": community_id,
                "community_size": community_size,
                "normalized_size": community_size / max_community_size,
            }

        self.logger.info(f"Detected {len(set(communities.values()))} communities in the skill graph")
        return community_features

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
            occ_info = self.occupations_extended[self.occupations_extended["occupation_code"] == occupation_code]
            if not occ_info.empty:
                title = occ_info.iloc[0].get("title", "")
                description = occ_info.iloc[0].get("description", "")
            else:
                title, description = "", ""
        else:
            title, description = "", ""

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
            edu_info = self.education[self.education["occupation_code"] == occupation_code]
            if not edu_info.empty:
                try:
                    edu_level = float(edu_info.iloc[0]["education_level"])
                except (ValueError, TypeError) as e:
                    # 当教育水平是文本描述而不是数值时，使用映射表转换
                    self.logger.debug(f"教育水平转换为数值时出错: {e}, 尝试使用映射表")
                    mapping = {
                        "Less than high school": 1,
                        "High school diploma or equivalent": 2,
                        "Some college": 3,
                        "Bachelor's degree": 4,
                        "Master's degree": 5,
                        "Doctoral or professional degree": 6,
                    }
                    edu_level = mapping.get(edu_info.iloc[0]["education_level"], 0)
            else:
                edu_level = 0.0
        else:
            edu_level = 0.0
        edu_feat = np.array([edu_level])

        # 4. 技术技能特征：统计技术技能数量及平均热度
        if self.tech_skills is not None:
            tech_info = self.tech_skills[self.tech_skills["occupation_code"] == occupation_code]
            if not tech_info.empty:
                tech_count = len(tech_info)
                try:
                    # Convert 'Y'/'N' to 1.0/0.0 before calculating mean
                    avg_hot = tech_info["is_hot"].map({"Y": 1.0, "N": 0.0}).mean()
                except (ValueError, TypeError, KeyError) as e:
                    # 当 is_hot 列不存在或数据类型无法转换时触发
                    self.logger.warning(f"计算技能热度平均值时出错: {e}, 使用默认值 0.0")
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
        occ_feat = self.get_extended_occupation_features(row["occupation_code"])
        # 技能节点特征：使用技能图中预构建的映射
        skill_id = row["skill_id"]
        skill_idx = self.node_id_to_idx.get(skill_id, self.node_id_to_idx.get("UNK"))
        # 标签及其他数值特征
        match = torch.tensor(row["match"], dtype=torch.float)
        importance = torch.tensor(row["importance"], dtype=torch.float)
        level = torch.tensor(row["level"], dtype=torch.float)

        # 数据增强（如果有）
        if self.transform:
            occ_feat, skill_idx, match, importance, level = self.transform(
                occ_feat, skill_idx, match, importance, level
            )
        return (occ_feat, torch.tensor(skill_idx, dtype=torch.long), match, importance, level)


class NoiseAugmentation:
    """数据增强：为职业特征添加随机噪声"""

    def __init__(self, noise_level=0.01):
        self.noise_level = noise_level

    def __call__(self, occ_feat, skill_idx, match, imp, level):
        occ_feat = occ_feat + torch.randn_like(occ_feat) * self.noise_level
        return occ_feat, skill_idx, match, imp, level


def create_dataloader(data_dir, batch_size=32, split="train", num_workers=4, **dataset_kwargs):
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
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return loader, dataset
