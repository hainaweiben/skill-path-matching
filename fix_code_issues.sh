#!/bin/bash

echo "开始修复代码问题..."

# 1. 使用 isort 修复导入问题
echo "1. 修复导入问题..."
isort --profile black --line-length 120 src/

# 2. 使用 black 修复格式问题
echo "2. 修复代码格式..."
black --line-length 120 src/

# 3. 使用 autoflake 移除未使用的导入和变量
echo "3. 移除未使用的导入和变量..."
pip install autoflake > /dev/null 2>&1
find src/ -name "*.py" -type f -exec autoflake --in-place --remove-all-unused-imports --remove-unused-variables {} +

# 4. 使用 pyupgrade 升级 Python 语法
echo "4. 升级 Python 语法..."
pip install pyupgrade > /dev/null 2>&1
find src/ -name "*.py" -type f -exec pyupgrade --py312-plus {} +

echo "基础修复完成！"
echo "运行 flake8 检查剩余问题..."
flake8 src/

echo "
注意：以下问题需要手动修复：
1. 函数复杂度问题 (C901):
   - src/data_processing/dataset.py: SkillMatchingDataset.build_skill_graph
   - src/models/skill_matching_model.py: SkillPathEncoder.__init__

2. 裸异常捕获 (E722):
   - 请在相关文件中指定具体的异常类型而不是使用 bare except

请检查这些文件并进行必要的重构。"
