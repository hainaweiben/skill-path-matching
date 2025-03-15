"""
数据处理工具模块
提供模型保存功能
"""

import os


import torch


def save_model(model: torch.nn.Module, save_path: str, metadata: dict = None):
    """
    保存模型

    Args:
        model: PyTorch模型
        save_path: 保存路径
        metadata: 元数据（可选）
    """
    # 创建保存目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 准备保存数据
    save_data = {"model_state_dict": model.state_dict()}

    # 添加元数据（如果有）
    if metadata is not None:
        save_data["metadata"] = metadata

    # 保存模型
    torch.save(save_data, save_path)
    print(f"模型已保存到: {save_path}")
