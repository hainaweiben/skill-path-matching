#!/usr/bin/env python

"""
评估脚本，用于评估训练好的技能匹配模型
"""

# 标准库导入
import argparse
import json
import logging
import os

# 第三方库导入
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# 本地模块导入
from src.data_processing.dataset import SkillMatchingDataset
from src.models.skill_matching_model import SkillMatchingModel, get_device

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估技能匹配模型")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="匹配阈值")
    parser.add_argument("--device", type=str, default=None, help="计算设备 (cpu 或 cuda)")
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(model, data_loader, threshold=0.5, device=None):
    """评估模型"""
    if device is None:
        device = get_device()

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for batch in data_loader:
            # 解包批次数据
            if isinstance(batch, dict):
                occupation_features = batch["occupation_features"].to(device)
                skill_idx = batch["skill_idx"].to(device)
                labels = batch["match"].to(device)
            else:
                # 如果batch是元组而不是字典
                occupation_features, skill_idx, labels, _, _ = batch
                occupation_features = occupation_features.to(device)
                skill_idx = skill_idx.to(device)
                labels = labels.to(device)

            # 前向传播
            scores, _ = model(occupation_features, skill_idx)

            # 保存预测结果
            preds = (scores > threshold).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_scores.append(scores.cpu().numpy())

    # 合并所有批次的结果
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)

    # 计算评估指标
    results = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "auc": roc_auc_score(all_labels, all_scores),
    }

    return results, all_preds, all_labels, all_scores


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 加载配置
    config = load_config(args.config)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取设备
    device = get_device(args.device)
    logger.info(f"使用设备: {device}")

    # 加载数据集
    logger.info("加载测试数据集...")
    test_dataset = SkillMatchingDataset(data_dir=config["data"]["data_dir"], split="test")

    # 创建数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    # 加载模型
    logger.info(f"加载模型从 {args.model_path}...")
    model = SkillMatchingModel(
        skill_input_dim=config["model"]["skill_input_dim"],
        occupation_input_dim=config["model"]["occupation_input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        embedding_dim=config["model"]["embedding_dim"],
        num_gnn_layers=config["model"]["num_gnn_layers"],
        num_mlp_layers=config["model"]["num_mlp_layers"],
        dropout=config["model"]["dropout"],
        gnn_type=config["model"]["gnn_type"],
        focal_alpha=config.get("model", {}).get("focal_alpha", 0.25),
        focal_gamma=config.get("model", {}).get("focal_gamma", 2.0),
    )

    # 加载模型权重
    checkpoint = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # 设置技能图
    model.skill_graph = test_dataset.skill_graph

    # 评估模型
    logger.info("评估模型...")
    results, predictions, labels, scores = evaluate_model(model, test_loader, args.threshold, device)

    # 输出评估结果
    logger.info(
        "评估结果:\n"
        f"  准确率={results['accuracy']:.4f}, 精确率={results['precision']:.4f}\n"
        f"  召回率={results['recall']:.4f}, F1={results['f1']:.4f}"
    )

    # 保存评估结果
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    # 保存预测结果
    predictions_df = pd.DataFrame({"prediction": predictions, "label": labels, "score": scores})
    predictions_path = os.path.join(args.output_dir, "predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)

    logger.info(f"评估结果已保存到 {results_path}")
    logger.info(f"预测结果已保存到 {predictions_path}")


if __name__ == "__main__":
    main()
