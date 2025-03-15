"""
模型训练模块
提供模型训练、验证和测试功能
"""

import json
import os
import time
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.skill_matching_model import get_device
from ..utils.data_utils import save_model


class Trainer:
    """模型训练器"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        early_stopping_patience: int = 10,
        gradient_clipping: float = None,
        log_interval: int = 1,
        save_best_only: bool = True,
        eval_metric: str = "val_loss",
        monitor_mode: str = "min",
        model_save_path: str | None = None,
        log_file_path: str | None = None,
        **kwargs,
    ):
        """
        初始化训练器

        Args:
            model: PyTorch模型
            device: 计算设备
            optimizer: 优化器（可选）
            scheduler: 学习率调度器（可选）
            early_stopping_patience: 早停耐心值
            gradient_clipping: 梯度裁剪阈值（可选）
            log_interval: 日志打印间隔
            save_best_only: 是否只保存最佳模型
            eval_metric: 评估指标
            monitor_mode: 监控模式，'min'或'max'
            model_save_path: 模型保存路径（可选）
            log_file_path: 日志文件保存路径（可选）
            kwargs: 其他参数
        """
        self.model = model
        self.device = device if device is not None else get_device()
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clipping = gradient_clipping
        self.log_interval = log_interval
        self.save_best_only = save_best_only
        self.eval_metric = eval_metric
        self.monitor_mode = monitor_mode
        self.model_save_path = model_save_path
        self.log_file_path = log_file_path

        # 将模型移动到设备
        self.model.to(self.device)

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {}
        self.val_metrics = {}
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.no_improvement_count = 0

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict[str, float]:
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch

        Returns:
            包含训练损失和指标的字典
        """
        self.model.train()
        total_loss = 0

        # 创建进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in pbar:
            # 解包批次数据
            inputs = self._prepare_batch(batch)
            batch["label"].to(self.device) if "label" in batch else None

            # 清除梯度
            self.optimizer.zero_grad()

            # 前向传播
            outputs, loss = self._forward(inputs)

            # 反向传播和优化
            loss.backward()
            if self.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
            self.optimizer.step()

            # 更新总损失
            total_loss += loss.item()

            # 更新进度条
            pbar.set_postfix({"loss": loss.item()})

        # 计算平均损失
        avg_loss = total_loss / len(train_loader)

        # 计算指标
        metrics = {"loss": avg_loss}

        return metrics

    def validate(self, val_loader: DataLoader, epoch: int) -> dict[str, float]:
        """
        验证模型

        Args:
            val_loader: 验证数据加载器
            epoch: 当前epoch

        Returns:
            包含验证损失和指标的字典
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        # 创建进度条
        pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}")

        with torch.no_grad():
            for batch in pbar:
                # 解包批次数据
                occupation_features, skill_idx, match, importance, level = batch

                # 将数据移动到设备
                occupation_features = occupation_features.to(self.device)
                skill_idx = skill_idx.to(self.device)
                match = match.to(self.device)
                importance = importance.to(self.device)
                level = level.to(self.device)

                # 前向传播
                preds, loss = self.model(occupation_features, skill_idx, match, importance, level)

                # 更新总损失
                total_loss += loss.item()

                # 收集预测和目标
                all_preds.append(preds.cpu().numpy())
                all_targets.append(match.cpu().numpy())

                # 更新进度条
                pbar.set_postfix({"val_loss": loss.item()})

        # 计算平均损失
        avg_loss = total_loss / len(val_loader)

        # 合并所有批次的预测和目标
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # 计算评估指标
        metrics = {
            "val_loss": avg_loss,
            "val_accuracy": accuracy_score(all_targets > 0.5, all_preds > 0.5),
            "val_precision": precision_score(all_targets > 0.5, all_preds > 0.5),
            "val_recall": recall_score(all_targets > 0.5, all_preds > 0.5),
            "val_f1": f1_score(all_targets > 0.5, all_preds > 0.5),
            "val_auc": roc_auc_score(all_targets, all_preds),
        }

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        early_stopping: bool = True,
        patience: int = 5,
    ):
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            early_stopping: 是否使用早停
            patience: 早停耐心值
        """
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.no_improvement_count = 0

        print(f"开始训练，共{num_epochs}个epoch")

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # 训练一个epoch
            train_metrics = self.train_epoch(train_loader, epoch)

            # 验证
            val_metrics = self.validate(val_loader, epoch)

            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # 更新指标
            self._update_metrics(train_metrics, val_metrics)

            # 检查是否是最佳模型
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.best_epoch = epoch
                self.no_improvement_count = 0

                # 保存最佳模型
                if self.model_save_path:
                    save_model(self.model, self.model_save_path)
                    print(f"保存最佳模型到 {self.model_save_path}")
            else:
                self.no_improvement_count += 1

            # 打印进度
            epoch_time = time.time() - epoch_start_time
            self._print_progress(epoch, num_epochs, epoch_time, train_metrics, val_metrics)

            # 早停
            if early_stopping and self.no_improvement_count >= patience:
                print(f"早停: {patience} 个epoch没有改善")
                break

        print(f"训练完成，最佳模型在epoch {self.best_epoch}，验证损失: {self.best_val_loss:.6f}")

    def test(self, test_loader: DataLoader, metric_functions: dict[str, Callable] | None = None) -> dict[str, float]:
        """
        测试模型

        Args:
            test_loader: 测试数据加载器
            metric_functions: 评估指标函数字典

        Returns:
            包含测试损失和指标的字典
        """
        # 测试过程与验证相同
        return self.validate(test_loader, 0)

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        使用模型进行预测

        Args:
            data_loader: 数据加载器

        Returns:
            预测结果
        """
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for batch in data_loader:
                # 将数据移动到设备
                inputs = self._prepare_batch(batch)

                # 前向传播
                outputs = self._forward(inputs)

                # 收集预测
                all_predictions.append(outputs.detach().cpu().numpy())

        # 合并所有批次的预测
        return np.concatenate(all_predictions)

    def _prepare_batch(self, batch):
        """
        准备批次数据

        参数:
            batch: 批次数据

        返回:
            dict: 准备好的批次数据
        """
        # 解包批次数据
        occupation_features, skill_idx, match, importance, level = batch

        # 将数据移动到设备
        occupation_features = occupation_features.to(self.device)
        skill_idx = skill_idx.to(self.device)
        match = match.to(self.device)
        importance = importance.to(self.device)
        level = level.to(self.device)

        return {
            "occupation_features": occupation_features,
            "skill_idx": skill_idx,
            "match": match,
            "importance": importance,
            "level": level,
        }

    def _forward(self, inputs):
        """
        前向传播

        参数:
            inputs (dict): 输入数据

        返回:
            tuple: (输出, 损失)
        """
        # 前向传播
        outputs, loss = self.model(
            inputs["occupation_features"], inputs["skill_idx"], inputs["match"], inputs["importance"], inputs["level"]
        )

        return outputs, loss

    def _update_metrics(self, train_metrics: dict[str, float], val_metrics: dict[str, float]):
        """
        更新指标记录

        Args:
            train_metrics: 训练指标
            val_metrics: 验证指标
        """
        # 记录损失
        self.train_losses.append(train_metrics["loss"])
        self.val_losses.append(val_metrics["val_loss"])

        # 记录其他指标
        for metric, value in train_metrics.items():
            if metric != "loss":
                if metric not in self.train_metrics:
                    self.train_metrics[metric] = []
                self.train_metrics[metric].append(value)

        for metric, value in val_metrics.items():
            if metric != "val_loss":
                if metric not in self.val_metrics:
                    self.val_metrics[metric] = []
                self.val_metrics[metric].append(value)

    def _print_progress(
        self,
        epoch: int,
        num_epochs: int,
        epoch_time: float,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ):
        """
        打印训练进度

        Args:
            epoch: 当前epoch
            num_epochs: 总epoch数
            epoch_time: epoch耗时
            train_metrics: 训练指标
            val_metrics: 验证指标
        """
        # 构建进度字符串
        progress = f"Epoch {epoch+1}/{num_epochs} [{epoch_time:.2f}s]"

        # 添加损失信息
        progress += f" - loss: {train_metrics['loss']:.6f} - val_loss: {val_metrics['val_loss']:.6f}"

        # 添加其他指标信息
        for metric in val_metrics:
            if metric != "val_loss":
                progress += f" - {metric}: {val_metrics[metric]:.4f}"

        print(progress)

        # 保存日志到文件
        if self.log_file_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

            # 创建日志条目
            log_entry = {"epoch": epoch + 1, "epoch_time": epoch_time, **train_metrics, **val_metrics}

            # 追加到日志文件
            with open(self.log_file_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")


class EarlyStopping:
    """早停类"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        初始化早停

        Args:
            patience: 耐心值，即在多少个epoch没有改善后停止
            min_delta: 最小改善阈值
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.stopped_epoch = 0

    def __call__(self, model: nn.Module, val_loss: float) -> bool:
        """
        检查是否应该停止训练

        Args:
            model: 模型
            val_loss: 验证损失

        Returns:
            是否应该停止训练
        """
        if self.best_score is None:
            # 首次调用
            self.best_score = val_loss
            if self.restore_best_weights:
                self.best_weights = {k: v.clone().detach() for k, v in model.state_dict().items()}
        elif val_loss > self.best_score - self.min_delta:
            # 验证损失没有足够改善
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = self.counter
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            # 验证损失有足够改善
            self.best_score = val_loss
            if self.restore_best_weights:
                self.best_weights = {k: v.clone().detach() for k, v in model.state_dict().items()}
            self.counter = 0

        return False
