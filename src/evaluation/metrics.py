"""
评估指标模块
用于评估人才-职位匹配模型的性能
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, ndcg_score
)
from typing import List, Dict, Tuple, Union, Optional


def calculate_ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray, k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    计算排序指标
    
    Args:
        y_true: 真实标签，二进制数组
        y_pred: 预测分数，浮点数数组
        k_values: 计算@k指标的k值列表
        
    Returns:
        包含各项指标的字典
    """
    metrics = {}
    
    # 计算AUC
    try:
        metrics['auc'] = roc_auc_score(y_true, y_pred)
    except:
        metrics['auc'] = 0.0
    
    # 计算平均精度（AP）
    try:
        metrics['ap'] = average_precision_score(y_true, y_pred)
    except:
        metrics['ap'] = 0.0
    
    # 计算各k值下的指标
    for k in k_values:
        if len(y_true) >= k:
            # 获取预测分数最高的k个索引
            top_k_indices = np.argsort(y_pred)[-k:]
            
            # 计算Precision@k
            precision_at_k = np.sum(y_true[top_k_indices]) / k
            metrics[f'precision@{k}'] = precision_at_k
            
            # 计算Recall@k
            if np.sum(y_true) > 0:
                recall_at_k = np.sum(y_true[top_k_indices]) / np.sum(y_true)
                metrics[f'recall@{k}'] = recall_at_k
            else:
                metrics[f'recall@{k}'] = 0.0
            
            # 计算F1@k
            if precision_at_k + recall_at_k > 0:
                f1_at_k = 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
                metrics[f'f1@{k}'] = f1_at_k
            else:
                metrics[f'f1@{k}'] = 0.0
    
    return metrics


def calculate_classification_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    计算分类指标
    
    Args:
        y_true: 真实标签，二进制数组
        y_pred_proba: 预测概率，浮点数数组
        threshold: 分类阈值
        
    Returns:
        包含各项指标的字典
    """
    # 根据阈值将概率转换为二进制预测
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {}
    
    # 计算准确率
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # 计算精确率
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    
    # 计算召回率
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    
    # 计算F1分数
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # 计算AUC
    try:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
    except:
        metrics['auc'] = 0.0
    
    return metrics


def calculate_ndcg(y_true: np.ndarray, y_pred: np.ndarray, k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    计算NDCG指标
    
    Args:
        y_true: 真实标签，二进制或多级别数组
        y_pred: 预测分数，浮点数数组
        k_values: 计算NDCG@k的k值列表
        
    Returns:
        包含各k值下NDCG的字典
    """
    metrics = {}
    
    # 确保输入是二维数组
    y_true_reshaped = y_true.reshape(1, -1)
    y_pred_reshaped = y_pred.reshape(1, -1)
    
    # 计算各k值下的NDCG
    for k in k_values:
        if len(y_true) >= k:
            try:
                ndcg = ndcg_score(y_true_reshaped, y_pred_reshaped, k=k)
                metrics[f'ndcg@{k}'] = ndcg
            except:
                metrics[f'ndcg@{k}'] = 0.0
    
    return metrics


def calculate_mrr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均倒数排名（Mean Reciprocal Rank, MRR）
    
    Args:
        y_true: 真实标签，二进制数组
        y_pred: 预测分数，浮点数数组
        
    Returns:
        MRR值
    """
    # 获取预测分数的排序索引（降序）
    sorted_indices = np.argsort(y_pred)[::-1]
    
    # 找到第一个相关项的排名
    for i, idx in enumerate(sorted_indices):
        if y_true[idx] == 1:
            return 1.0 / (i + 1)
    
    return 0.0


def calculate_map(y_true_list: List[np.ndarray], y_pred_list: List[np.ndarray]) -> float:
    """
    计算平均精度均值（Mean Average Precision, MAP）
    
    Args:
        y_true_list: 真实标签列表，每个元素是一个二进制数组
        y_pred_list: 预测分数列表，每个元素是一个浮点数数组
        
    Returns:
        MAP值
    """
    if len(y_true_list) == 0:
        return 0.0
    
    ap_sum = 0.0
    count = 0
    
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        if np.sum(y_true) > 0:  # 只有当有相关项时才计算AP
            try:
                ap = average_precision_score(y_true, y_pred)
                ap_sum += ap
                count += 1
            except:
                pass
    
    if count == 0:
        return 0.0
    
    return ap_sum / count


def calculate_skill_coverage(candidate_skills: List[str], job_skills: List[str]) -> Dict[str, float]:
    """
    计算技能覆盖率指标
    
    Args:
        candidate_skills: 候选人技能列表
        job_skills: 职位所需技能列表
        
    Returns:
        包含技能覆盖率指标的字典
    """
    metrics = {}
    
    # 转换为集合
    candidate_skills_set = set(candidate_skills)
    job_skills_set = set(job_skills)
    
    # 计算交集
    common_skills = candidate_skills_set.intersection(job_skills_set)
    
    # 计算覆盖率
    if len(job_skills_set) > 0:
        metrics['job_skill_coverage'] = len(common_skills) / len(job_skills_set)
    else:
        metrics['job_skill_coverage'] = 0.0
    
    # 计算候选人技能利用率
    if len(candidate_skills_set) > 0:
        metrics['candidate_skill_utilization'] = len(common_skills) / len(candidate_skills_set)
    else:
        metrics['candidate_skill_utilization'] = 0.0
    
    # 计算F1分数
    if metrics['job_skill_coverage'] + metrics['candidate_skill_utilization'] > 0:
        metrics['skill_f1'] = 2 * metrics['job_skill_coverage'] * metrics['candidate_skill_utilization'] / (
            metrics['job_skill_coverage'] + metrics['candidate_skill_utilization']
        )
    else:
        metrics['skill_f1'] = 0.0
    
    return metrics


def calculate_skill_path_metrics(candidate_skill_paths: List[List[str]], job_skill_paths: List[List[str]]) -> Dict[str, float]:
    """
    计算技能路径相关指标
    
    Args:
        candidate_skill_paths: 候选人技能路径列表，每个路径是一个技能序列
        job_skill_paths: 职位所需技能路径列表，每个路径是一个技能序列
        
    Returns:
        包含技能路径指标的字典
    """
    metrics = {}
    
    # 如果没有路径，返回0
    if not candidate_skill_paths or not job_skill_paths:
        metrics['path_coverage'] = 0.0
        metrics['path_similarity'] = 0.0
        return metrics
    
    # 计算路径覆盖率
    path_coverage_scores = []
    for job_path in job_skill_paths:
        max_coverage = 0.0
        for candidate_path in candidate_skill_paths:
            # 计算两个路径的交集大小
            common_skills = set(job_path).intersection(set(candidate_path))
            coverage = len(common_skills) / len(job_path) if job_path else 0.0
            max_coverage = max(max_coverage, coverage)
        path_coverage_scores.append(max_coverage)
    
    metrics['path_coverage'] = np.mean(path_coverage_scores) if path_coverage_scores else 0.0
    
    # 计算路径相似度（使用最长公共子序列）
    path_similarity_scores = []
    for job_path in job_skill_paths:
        max_similarity = 0.0
        for candidate_path in candidate_skill_paths:
            similarity = longest_common_subsequence(job_path, candidate_path) / max(len(job_path), len(candidate_path))
            max_similarity = max(max_similarity, similarity)
        path_similarity_scores.append(max_similarity)
    
    metrics['path_similarity'] = np.mean(path_similarity_scores) if path_similarity_scores else 0.0
    
    return metrics


def longest_common_subsequence(seq1: List[str], seq2: List[str]) -> int:
    """
    计算两个序列的最长公共子序列长度
    
    Args:
        seq1: 第一个序列
        seq2: 第二个序列
        
    Returns:
        最长公共子序列长度
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]
