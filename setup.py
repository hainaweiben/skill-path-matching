from setuptools import setup, find_packages

setup(
    name="skill_path_matching",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "scikit-learn",
        "networkx",
        "gensim",
        "pyyaml",
        "tqdm",
    ],
    python_requires=">=3.8",
)
