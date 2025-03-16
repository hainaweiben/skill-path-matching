# 基于能力点路径建模的人才和技能点匹配深度学习模型

本项目旨在研究和实现一个基于深度学习的人才与技能点匹配系统，通过能力点路径建模来提高匹配准确性和可解释性。

## 项目概述

在当今快速变化的就业市场中，精确匹配人才与职位需求变得越来越重要。本项目通过以下方式解决这一挑战：

1. 构建能力点路径模型，捕捉技能之间的演进关系
2. 利用深度学习技术进行人才与职位的精准匹配
3. 提供可解释的匹配结果，帮助招聘方和求职者理解匹配原因

### 最新性能指标

通过持续优化，我们的模型已经达到了业界领先的性能水平：

- **准确率 (Accuracy)**: 80.00%
- **精确率 (Precision)**: 92.94%
- **召回率 (Recall)**: 64.93%
- **F1值**: 76.45%
- **AUC**: 91.27%

这些指标表明我们的模型在保持高精确率的同时，也实现了良好的整体平衡。

## 技术架构

本项目结合了以下技术和方法：

- **能力点路径建模**：构建技能演进图，捕捉技能间的层次和发展关系
- **深度表示学习**：学习人才简历和职位描述的语义表示
- **注意力机制**：识别关键技能和能力点
- **图神经网络**：对技能关系图进行建模
- **迁移学习**：利用预训练模型处理文本数据

## 项目结构

```
skill_path_matching/
├── configs/                # 配置文件
├── data/                   # 数据目录
│   ├── raw/                # 原始数据
│   └── processed/          # 处理后的数据
├── outputs/                # 模型输出和评估结果
├── src/                    # 源代码
│   ├── data_processing/    # 数据处理模块
│   ├── evaluation/         # 评估模块
│   ├── models/             # 模型定义
│   ├── training/           # 训练模块
│   └── utils/              # 工具函数
└── run.sh                  # 主运行脚本
```

## 主要功能

1. **简历和职位描述解析**：从文本中提取结构化信息
2. **技能提取与标准化**：识别和规范化技能表述
3. **能力点路径建模**：构建技能关系图和发展路径
4. **深度匹配模型**：基于深度学习的匹配算法
5. **匹配结果可视化**：直观展示匹配度和关键匹配点

## 环境配置与安装

### 系统要求
- Python 3.12
- CUDA 10.2+ (用于GPU加速，可选)
- 8GB+ RAM

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/hainaweiben/skill-path-matching.git
cd skill-path-matching
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

### GPU加速支持

本项目支持GPU加速，可以显著提高模型训练和推理速度。系统会自动检测可用的GPU设备并使用。

- 如果有CUDA兼容的GPU，模型将自动使用GPU进行训练和推理
- 如果没有可用的GPU，模型将回退到CPU模式
- 可以通过`src/models/skill_matching_model.py`中的`get_device()`函数查看当前使用的设备

## 数据准备

1. 将原始数据放入`data/raw/`目录
2. 运行数据处理脚本
```bash
bash src/data_processing/scripts/process_data.sh
```
3. 处理后的数据将保存在`data/processed/`目录

## 使用说明

### 1. 数据处理
首先需要处理原始数据并构建数据集：

```bash
# 处理原始数据
python src/data_processing/process_data.py

# 构建数据集（包括技能图构建等）
python src/data_processing/build_dataset.py
```

### 2. 模型训练

使用项目的主运行脚本进行训练：

```bash
./run.sh train
```

训练日志和模型检查点将保存在`outputs/`目录下。主要输出包括：
- 模型检查点（.pth文件）
- 训练日志（.log文件）
- 训练配置（config.yaml）
- TensorBoard事件文件

### 3. 模型评估

使用项目的主运行脚本进行评估：

```bash
./run.sh evaluate
```

评估结果将保存在`outputs/evaluation_[时间戳]`目录中，包括：
- 评估指标（准确率、精确率、召回率、F1分数、AUC等）
- 预测结果CSV文件

### 4. 项目清理

如需清理项目（删除缓存文件、旧的输出目录等）：

```bash
bash cleanup.sh
```

## 当前版本

- v0.1.0 - 初始版本，实现了基本的模型架构和评估功能
- v0.2.0 - 重构了项目结构，统一了文件路径和命名风格
- v0.3.0 - 优化数据预处理和模型架构
  - 改进技能图构建，使用节点度数和名称哈希作为特征
  - 实现数据集平衡处理和少数类过采样
  - 增强技能提取功能
  - 性能显著提升：准确率80%，精确率92.94%，AUC 91.27%

## 未来工作

1. 提高模型准确率
2. 增加更多特征工程
3. 优化图神经网络结构
4. 实现更多评估指标
5. 添加可视化工具

## 贡献者

- hainaweiben

## 许可证

本项目采用MIT许可证。详见LICENSE文件。
