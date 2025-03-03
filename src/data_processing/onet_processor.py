"""
O*NET数据处理模块
用于处理O*NET数据库，提取技能、职业和关系信息
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
import networkx as nx
import json

class OnetProcessor:
    """O*NET数据处理器"""
    
    def __init__(self, data_dir: str):
        """
        初始化处理器
        
        Args:
            data_dir: O*NET数据目录
        """
        self.data_dir = data_dir
        self.occupation_data = None
        self.skills_data = None
        self.knowledge_data = None
        self.abilities_data = None
        self.work_activities_data = None
        self.skills_to_work_activities = None
        
    def load_data(self):
        """加载O*NET数据"""
        # 加载职业数据
        self.occupation_data = pd.read_excel(
            os.path.join(self.data_dir, 'Occupation Data.xlsx')
        )
        
        # 加载技能数据
        self.skills_data = pd.read_excel(
            os.path.join(self.data_dir, 'Skills.xlsx')
        )
        
        # 加载知识数据
        self.knowledge_data = pd.read_excel(
            os.path.join(self.data_dir, 'Knowledge.xlsx')
        )
        
        # 加载能力数据
        self.abilities_data = pd.read_excel(
            os.path.join(self.data_dir, 'Abilities.xlsx')
        )
        
        # 加载工作活动数据
        self.work_activities_data = pd.read_excel(
            os.path.join(self.data_dir, 'Work Activities.xlsx')
        )
        
        # 加载技能到工作活动的映射
        try:
            self.skills_to_work_activities = pd.read_excel(
                os.path.join(self.data_dir, 'Skills to Work Activities.xlsx')
            )
        except FileNotFoundError:
            print("警告: 'Skills to Work Activities.xlsx' 文件不存在，将无法构建完整的技能关系图")
            self.skills_to_work_activities = None
        
        print("数据加载完成")
    
    def process_occupations(self) -> pd.DataFrame:
        """
        处理职业数据
        
        Returns:
            处理后的职业数据框
        """
        if self.occupation_data is None:
            self.load_data()
        
        # 选择需要的列
        occupations = self.occupation_data[['O*NET-SOC Code', 'Title', 'Description']].copy()
        
        # 重命名列
        occupations.columns = ['occupation_code', 'title', 'description']
        
        return occupations
    
    def process_skills(self) -> pd.DataFrame:
        """
        处理技能数据
        
        Returns:
            处理后的技能数据框
        """
        if self.skills_data is None:
            self.load_data()
        
        # 获取唯一技能
        unique_skills = self.skills_data[['Element ID', 'Element Name']].drop_duplicates()
        
        # 重命名列
        unique_skills.columns = ['skill_id', 'skill_name']
        
        
        return unique_skills
    
    def process_occupation_skills(self) -> pd.DataFrame:
        """
        处理职业-技能关系数据
        
        Returns:
            处理后的职业-技能关系数据框
        """
        if self.skills_data is None:
            self.load_data()
        
        # 选择需要的列
        occupation_skills = self.skills_data[['O*NET-SOC Code', 'Element ID', 'Scale ID', 'Data Value']].copy()
        
        # 重命名列
        occupation_skills.columns = ['occupation_code', 'skill_id', 'scale_id', 'value']
        
        # 只保留重要性和水平评分
        occupation_skills = occupation_skills[occupation_skills['scale_id'].isin(['IM', 'LV'])]
        
        # 转换为宽格式
        occupation_skills_wide = occupation_skills.pivot_table(
            index=['occupation_code', 'skill_id'],
            columns='scale_id',
            values='value'
        ).reset_index()
        
        # 重命名列
        occupation_skills_wide.columns.name = None
        
        return occupation_skills_wide
    
    def build_skill_graph(self) -> nx.DiGraph:
        """
        构建技能关系图
        
        Returns:
            技能关系有向图
        """
        # 创建有向图
        G = nx.DiGraph()
        
        # 获取唯一技能
        unique_skills = self.process_skills()
        
        # 添加节点
        for _, row in unique_skills.iterrows():
            G.add_node(row['skill_id'], name=row['skill_name'])
        
        # 如果有技能到工作活动的映射，则使用它来建立技能之间的关系
        if self.skills_to_work_activities is not None:
            # 通过工作活动建立技能之间的关系
            # 如果两个技能关联到同一个工作活动，则它们之间有关系
            skill_to_activity = self.skills_to_work_activities[['Skills Element ID', 'Work Activities Element ID']].copy()
            skill_to_activity.columns = ['skill_id', 'activity_id']
            
            # 对于每个活动，找出相关的技能
            activity_to_skills = {}
            for _, row in skill_to_activity.iterrows():
                activity_id = row['activity_id']
                skill_id = row['skill_id']
                
                if activity_id not in activity_to_skills:
                    activity_to_skills[activity_id] = []
                
                activity_to_skills[activity_id].append(skill_id)
            
            # 为相关的技能对添加边
            for activity_id, skills in activity_to_skills.items():
                for i in range(len(skills)):
                    for j in range(i + 1, len(skills)):
                        # 添加双向边
                        if G.has_edge(skills[i], skills[j]):
                            # 增加权重
                            G[skills[i]][skills[j]]['weight'] += 1
                        else:
                            G.add_edge(skills[i], skills[j], weight=1, type='related')
                        
                        if G.has_edge(skills[j], skills[i]):
                            G[skills[j]][skills[i]]['weight'] += 1
                        else:
                            G.add_edge(skills[j], skills[i], weight=1, type='related')
        else:
            # 如果没有技能到工作活动的映射，则使用职业-技能关系来建立技能之间的关系
            # 如果两个技能经常出现在同一个职业中，则它们之间有关系
            occupation_skills = self.process_occupation_skills()
            
            # 对于每个职业，找出相关的技能
            occupation_to_skills = {}
            for _, row in occupation_skills.iterrows():
                occupation_code = row['occupation_code']
                skill_id = row['skill_id']
                
                if occupation_code not in occupation_to_skills:
                    occupation_to_skills[occupation_code] = []
                
                occupation_to_skills[occupation_code].append(skill_id)
            
            # 为相关的技能对添加边
            for occupation_code, skills in occupation_to_skills.items():
                for i in range(len(skills)):
                    for j in range(i + 1, len(skills)):
                        # 添加双向边
                        if G.has_edge(skills[i], skills[j]):
                            # 增加权重
                            G[skills[i]][skills[j]]['weight'] += 1
                        else:
                            G.add_edge(skills[i], skills[j], weight=1, type='co-occurrence')
                        
                        if G.has_edge(skills[j], skills[i]):
                            G[skills[j]][skills[i]]['weight'] += 1
                        else:
                            G.add_edge(skills[j], skills[i], weight=1, type='co-occurrence')
        
        return G
    
    def save_processed_data(self, output_dir: str):
        """
        保存处理后的数据
        
        Args:
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存处理后的职业数据
        occupations = self.process_occupations()
        occupations.to_csv(os.path.join(output_dir, 'occupations.csv'), index=False)
        
        # 保存处理后的技能数据
        skills = self.process_skills()
        skills.to_csv(os.path.join(output_dir, 'skills.csv'), index=False)
        
        # 保存处理后的职业-技能关系数据
        occupation_skills = self.process_occupation_skills()
        occupation_skills.to_csv(os.path.join(output_dir, 'occupation_skills.csv'), index=False)
        
        # 构建并保存技能图
        skill_graph = self.build_skill_graph()
        
        # 将NetworkX图转换为JSON格式
        nodes = []
        for node, data in skill_graph.nodes(data=True):
            node_data = {'id': node}
            node_data.update(data)
            nodes.append(node_data)
        
        edges = []
        for u, v, data in skill_graph.edges(data=True):
            edge_data = {'source': u, 'target': v}
            edge_data.update(data)
            edges.append(edge_data)
        
        # 保存技能图
        with open(os.path.join(output_dir, 'skill_graph.json'), 'w') as f:
            json.dump({'nodes': nodes, 'edges': edges}, f)
        
        print(f"处理后的数据已保存到 {output_dir}")
    
    def generate_job_skill_dataset(self, output_dir: str):
        """
        生成职位-技能数据集
        
        Args:
            output_dir: 输出目录
        """
        # 获取处理后的数据
        occupations = self.process_occupations()
        skills = self.process_skills()
        occupation_skills = self.process_occupation_skills()
        
        # 获取重要技能（重要性 >= 3.0）

        important_skills = occupation_skills[occupation_skills['IM'] >= 2.0].copy()
        
        # 合并职业和技能信息
        important_skills = important_skills.merge(
            occupations[['occupation_code', 'title']],
            on='occupation_code',
            how='left'
        )
        
        important_skills = important_skills.merge(
            skills[['skill_id', 'skill_name']],
            on='skill_id',
            how='left'
        )
        
        # 创建职位-技能矩阵（假设每个 occupation_code 和 skill_id 组合唯一）
        job_skill_matrix = occupation_skills.pivot_table(
            values='IM',
            index='occupation_code',
            columns='skill_id',
            aggfunc='first',  
            fill_value=0
        )

        # 创建职位元数据
        job_metadata = occupations.set_index('occupation_code')
        
        # 创建技能元数据
        skill_metadata = skills.set_index('skill_id')
            
        # 保存数据
        job_skill_matrix.to_csv(os.path.join(output_dir, 'job_skill_matrix.csv'))
        job_metadata.to_csv(os.path.join(output_dir, 'job_metadata.csv'))
        skill_metadata.to_csv(os.path.join(output_dir, 'skill_metadata.csv'))
        
        # 创建匹配样本
        # 每个职业与其所需技能是正样本，与其他随机技能是负样本
        matching_samples = []
        
        # 跟踪每个职业的样本数量，确保正负样本平衡
        occupation_sample_counts = {}
        
        # 计算每个职业的技能频率分布
        occupation_skill_freq = {}
        for occ_code in occupations['occupation_code'].unique():
            occ_skills = occupation_skills[occupation_skills['occupation_code'] == occ_code]
            total_importance = occ_skills['IM'].sum()
            if total_importance > 0:
                skill_freq = occ_skills.set_index('skill_id')['IM'] / total_importance
                occupation_skill_freq[occ_code] = skill_freq.to_dict()
        
        for _, row in important_skills.iterrows():
            occupation_code = row['occupation_code']
            
            # 初始化计数器
            if occupation_code not in occupation_sample_counts:
                occupation_sample_counts[occupation_code] = {'positive': 0, 'negative': 0}
            
            # 计算技能在该职业中的相对重要性
            skill_freq = occupation_skill_freq.get(occupation_code, {}).get(row['skill_id'], 0)
            
            # 正样本
            matching_samples.append({
                'occupation_code': occupation_code,
                'skill_id': row['skill_id'],
                'occupation_title': row['title'],
                'skill_name': row['skill_name'],
                'importance': row['IM'],
                'level': row['LV'],
                'relative_importance': skill_freq,
                'match': 1  # 匹配
            })
            occupation_sample_counts[occupation_code]['positive'] += 1
        
        # 为每个职业生成平衡的负样本
        for occupation_code, counts in occupation_sample_counts.items():
            # 获取该职业的所有技能ID
            occupation_skill_ids = important_skills[important_skills['occupation_code'] == occupation_code]['skill_id'].tolist()
            # 获取不在该职业技能列表中的技能
            non_occupation_skills = skills[~skills['skill_id'].isin(occupation_skill_ids)]
            
            # 确定需要生成的负样本数量，确保与正样本数量相等
            num_negative_samples = counts['positive']
            
            # 如果有足够的非职业技能，生成负样本
            if len(non_occupation_skills) > 0:
                # 如果非职业技能数量少于需要的负样本数量，则进行有放回抽样
                if len(non_occupation_skills) < num_negative_samples:
                    negative_skills = non_occupation_skills.sample(num_negative_samples, replace=True)
                else:
                    negative_skills = non_occupation_skills.sample(num_negative_samples, replace=False)
                
                # 获取职业标题
                occupation_title = occupations[occupations['occupation_code'] == occupation_code]['title'].iloc[0]
                
                # 为每个负样本添加到数据集
                for _, skill_row in negative_skills.iterrows():
                    matching_samples.append({
                        'occupation_code': occupation_code,
                        'skill_id': skill_row['skill_id'],
                        'occupation_title': occupation_title,
                        'skill_name': skill_row['skill_name'],
                        'importance': 0,
                        'level': 0,
                        'relative_importance': 0,
                        'match': 0  # 不匹配
                    })
                    occupation_sample_counts[occupation_code]['negative'] += 1
        
        # 转换为DataFrame并保存
        matching_df = pd.DataFrame(matching_samples)
        
        # 打乱数据顺序
        matching_df = matching_df.sample(frac=1).reset_index(drop=True)
        
        # 保存数据集
        matching_df.to_csv(os.path.join(output_dir, 'skill_matching_dataset.csv'), index=False)
        
        # 打印样本统计信息
        total_positive = sum(counts['positive'] for counts in occupation_sample_counts.values())
        total_negative = sum(counts['negative'] for counts in occupation_sample_counts.values())
        print(f"职位-技能匹配数据集已保存到 {output_dir}")
        print(f"总样本数: {len(matching_df)}, 正样本: {total_positive}, 负样本: {total_negative}, 正负样本比例: {total_positive/total_negative:.2f}")
