"""
可视化工具模块
提供数据和模型结果的可视化功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any, Union, Optional
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_skill_graph(graph: nx.Graph, figsize: Tuple[int, int] = (12, 10), 
                     node_size: int = 100, node_color: str = 'skyblue',
                     edge_color: str = 'gray', font_size: int = 8,
                     title: str = '技能关系图', save_path: Optional[str] = None):
    """
    绘制技能关系图
    
    Args:
        graph: NetworkX图对象
        figsize: 图像大小
        node_size: 节点大小
        node_color: 节点颜色
        edge_color: 边颜色
        font_size: 字体大小
        title: 图标题
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=figsize)
    
    # 使用spring布局
    pos = nx.spring_layout(graph, seed=42)
    
    # 绘制节点
    nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=node_color, alpha=0.8)
    
    # 绘制边
    nx.draw_networkx_edges(graph, pos, edge_color=edge_color, alpha=0.5)
    
    # 绘制标签
    nx.draw_networkx_labels(graph, pos, font_size=font_size, font_family='sans-serif')
    
    # 设置标题和布局
    plt.title(title, fontsize=15)
    plt.axis('off')
    
    # 保存图像（如果指定了路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_skill_communities(graph: nx.Graph, communities: List[List[str]], 
                           figsize: Tuple[int, int] = (15, 12),
                           title: str = '技能社区检测', save_path: Optional[str] = None):
    """
    绘制技能社区
    
    Args:
        graph: NetworkX图对象
        communities: 社区列表，每个社区是一个节点ID列表
        figsize: 图像大小
        title: 图标题
        save_path: 保存路径（可选）
    """
    plt.figure(figsize=figsize)
    
    # 使用spring布局
    pos = nx.spring_layout(graph, seed=42)
    
    # 为每个社区分配一个颜色
    colors = plt.cm.rainbow(np.linspace(0, 1, len(communities)))
    
    # 绘制每个社区
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(graph, pos, nodelist=community, 
                               node_color=[colors[i]] * len(community),
                               node_size=100, alpha=0.8)
    
    # 绘制边
    nx.draw_networkx_edges(graph, pos, edge_color='gray', alpha=0.5)
    
    # 绘制标签
    nx.draw_networkx_labels(graph, pos, font_size=8, font_family='sans-serif')
    
    # 设置标题和布局
    plt.title(title, fontsize=15)
    plt.axis('off')
    
    # 保存图像（如果指定了路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_skill_embeddings(embeddings: np.ndarray, skill_names: List[str], 
                          method: str = 'tsne', n_components: int = 2,
                          figsize: Tuple[int, int] = (12, 10),
                          title: str = '技能嵌入可视化', save_path: Optional[str] = None):
    """
    可视化技能嵌入
    
    Args:
        embeddings: 嵌入矩阵，形状为[n_skills, embedding_dim]
        skill_names: 技能名称列表
        method: 降维方法，'tsne'或'pca'
        n_components: 降维后的维度
        figsize: 图像大小
        title: 图标题
        save_path: 保存路径（可选）
    """
    # 降维
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"不支持的降维方法: {method}")
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # 创建DataFrame
    df = pd.DataFrame(reduced_embeddings, columns=[f'Dimension {i+1}' for i in range(n_components)])
    df['Skill'] = skill_names
    
    # 绘图
    plt.figure(figsize=figsize)
    
    if n_components == 2:
        # 二维散点图
        sns.scatterplot(x='Dimension 1', y='Dimension 2', data=df, s=100, alpha=0.7)
        
        # 添加标签
        for i, row in df.iterrows():
            plt.text(row['Dimension 1'], row['Dimension 2'], row['Skill'], 
                     fontsize=9, ha='center', va='center')
    
    elif n_components == 3:
        # 三维散点图
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(df['Dimension 1'], df['Dimension 2'], df['Dimension 3'], s=100, alpha=0.7)
        
        # 添加标签
        for i, row in df.iterrows():
            ax.text(row['Dimension 1'], row['Dimension 2'], row['Dimension 3'], 
                    row['Skill'], fontsize=9)
    
    # 设置标题
    plt.title(title, fontsize=15)
    
    # 保存图像（如果指定了路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_learning_curves(train_metrics: Dict[str, List[float]], val_metrics: Dict[str, List[float]],
                         figsize: Tuple[int, int] = (15, 10), title: str = '学习曲线',
                         save_path: Optional[str] = None):
    """
    绘制学习曲线
    
    Args:
        train_metrics: 训练指标字典，键为指标名称，值为每个epoch的指标值列表
        val_metrics: 验证指标字典，键为指标名称，值为每个epoch的指标值列表
        figsize: 图像大小
        title: 图标题
        save_path: 保存路径（可选）
    """
    # 获取所有指标名称
    metrics = list(train_metrics.keys())
    n_metrics = len(metrics)
    
    # 计算子图布局
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # 创建图像
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # 绘制每个指标的学习曲线
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        epochs = range(1, len(train_metrics[metric]) + 1)
        
        # 绘制训练曲线
        ax.plot(epochs, train_metrics[metric], 'b-', label=f'训练 {metric}')
        
        # 绘制验证曲线（如果有）
        if metric in val_metrics:
            ax.plot(epochs, val_metrics[metric], 'r-', label=f'验证 {metric}')
        
        # 设置标题和标签
        ax.set_title(f'{metric} 曲线')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)
    
    # 隐藏多余的子图
    for i in range(n_metrics, n_rows * n_cols):
        axes[i].axis('off')
    
    # 设置整体标题和布局
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # 保存图像（如果指定了路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5,
                          figsize: Tuple[int, int] = (8, 6), title: str = '混淆矩阵',
                          save_path: Optional[str] = None):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测概率
        threshold: 分类阈值
        figsize: 图像大小
        title: 图标题
        save_path: 保存路径（可选）
    """
    # 将概率转换为二进制预测
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # 绘制混淆矩阵
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    # 设置标题和标签
    plt.title(title, fontsize=15)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    # 保存图像（如果指定了路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, 
                   figsize: Tuple[int, int] = (8, 6), title: str = 'ROC曲线',
                   save_path: Optional[str] = None):
    """
    绘制ROC曲线
    
    Args:
        y_true: 真实标签
        y_pred: 预测概率
        figsize: 图像大小
        title: 图标题
        save_path: 保存路径（可选）
    """
    from sklearn.metrics import roc_curve, auc
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # 设置标题和标签
    plt.title(title, fontsize=15)
    plt.xlabel('假正例率')
    plt.ylabel('真正例率')
    plt.legend(loc='lower right')
    
    # 保存图像（如果指定了路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_precision_recall_curve(y_true: np.ndarray, y_pred: np.ndarray,
                                figsize: Tuple[int, int] = (8, 6), title: str = '精确率-召回率曲线',
                                save_path: Optional[str] = None):
    """
    绘制精确率-召回率曲线
    
    Args:
        y_true: 真实标签
        y_pred: 预测概率
        figsize: 图像大小
        title: 图标题
        save_path: 保存路径（可选）
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # 计算精确率-召回率曲线
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    
    # 绘制精确率-召回率曲线
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR曲线 (AP = {ap:.3f})')
    
    # 设置标题和标签
    plt.title(title, fontsize=15)
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc='lower left')
    
    # 保存图像（如果指定了路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_skill_distribution(skills: List[str], figsize: Tuple[int, int] = (12, 8),
                            top_n: int = 20, title: str = '技能分布',
                            save_path: Optional[str] = None):
    """
    绘制技能分布
    
    Args:
        skills: 技能列表
        figsize: 图像大小
        top_n: 显示前N个最常见的技能
        title: 图标题
        save_path: 保存路径（可选）
    """
    # 计算技能频率
    skill_counts = pd.Series(skills).value_counts()
    
    # 获取前N个技能
    top_skills = skill_counts.head(top_n)
    
    # 绘制条形图
    plt.figure(figsize=figsize)
    sns.barplot(x=top_skills.values, y=top_skills.index, palette='viridis')
    
    # 设置标题和标签
    plt.title(title, fontsize=15)
    plt.xlabel('频率')
    plt.ylabel('技能')
    
    # 保存图像（如果指定了路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_skill_path(path: List[str], figsize: Tuple[int, int] = (10, 6),
                    title: str = '技能路径', save_path: Optional[str] = None):
    """
    绘制技能路径
    
    Args:
        path: 技能路径（技能列表）
        figsize: 图像大小
        title: 图标题
        save_path: 保存路径（可选）
    """
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加节点和边
    for i in range(len(path)):
        G.add_node(path[i])
        if i > 0:
            G.add_edge(path[i-1], path[i])
    
    # 绘制图
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='skyblue', alpha=0.8)
    
    # 绘制边和箭头
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, arrowsize=20)
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    # 设置标题和布局
    plt.title(title, fontsize=15)
    plt.axis('off')
    
    # 保存图像（如果指定了路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
