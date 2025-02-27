"""
职位描述解析模块
用于从职位描述文本中提取结构化信息，包括职位要求、职责和所需技能
"""

import re
import spacy
import pandas as pd
from typing import Dict, List, Any, Set

# 加载spaCy模型
try:
    nlp = spacy.load("zh_core_web_sm")
except:
    nlp = spacy.load("en_core_web_sm")

class JobParser:
    """职位描述解析类，用于从职位描述中提取结构化信息"""
    
    def __init__(self, skill_taxonomy_path: str = None):
        """
        初始化职位描述解析器
        
        Args:
            skill_taxonomy_path: 技能分类体系文件路径
        """
        self.skill_patterns = self._load_skill_taxonomy(skill_taxonomy_path) if skill_taxonomy_path else []
        
    def _load_skill_taxonomy(self, path: str) -> List[str]:
        """
        加载技能分类体系
        
        Args:
            path: 技能分类体系文件路径
            
        Returns:
            技能列表
        """
        try:
            skills_df = pd.read_csv(path)
            return skills_df['skill_name'].tolist()
        except Exception as e:
            print(f"加载技能分类体系失败: {e}")
            return []
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        解析职位描述文本
        
        Args:
            text: 职位描述文本
            
        Returns:
            包含解析结果的字典
        """
        # 基本信息提取
        basic_info = self._extract_basic_info(text)
        
        # 职责提取
        responsibilities = self._extract_responsibilities(text)
        
        # 要求提取
        requirements = self._extract_requirements(text)
        
        # 技能提取
        skills = self._extract_skills(text)
        
        # 经验要求
        experience = self._extract_experience_requirement(text)
        
        # 学历要求
        education = self._extract_education_requirement(text)
        
        return {
            "basic_info": basic_info,
            "responsibilities": responsibilities,
            "requirements": requirements,
            "skills": skills,
            "experience": experience,
            "education": education
        }
    
    def _extract_basic_info(self, text: str) -> Dict[str, str]:
        """提取基本信息"""
        info = {}
        
        # 提取职位名称
        title_match = re.search(r'(职位名称|岗位名称|Position)[：:]\s*([^\n]+)', text)
        if title_match:
            info['title'] = title_match.group(2).strip()
        
        # 提取部门
        department_match = re.search(r'(部门|Department)[：:]\s*([^\n]+)', text)
        if department_match:
            info['department'] = department_match.group(2).strip()
        
        # 提取薪资
        salary_match = re.search(r'(薪资|工资|Salary)[：:]\s*([^\n]+)', text)
        if salary_match:
            info['salary'] = salary_match.group(2).strip()
        
        # 提取工作地点
        location_match = re.search(r'(工作地点|Location)[：:]\s*([^\n]+)', text)
        if location_match:
            info['location'] = location_match.group(2).strip()
        
        return info
    
    def _extract_responsibilities(self, text: str) -> List[str]:
        """提取职责"""
        responsibilities = []
        
        # 提取职责部分
        resp_sections = re.findall(r'(工作职责|岗位职责|Job Responsibilities|Responsibilities|职位描述).*?(?=任职要求|岗位要求|工作要求|Job Requirements|$)', text, re.DOTALL | re.IGNORECASE)
        
        if resp_sections:
            resp_text = resp_sections[0]
            # 提取列表项
            items = re.findall(r'[•·\-\d+、]+\s*([^\n•·\-\d+、]+)', resp_text)
            responsibilities.extend([item.strip() for item in items if item.strip()])
            
            # 如果没有找到列表项，尝试按句子分割
            if not responsibilities:
                sentences = re.split(r'[.。;；]', resp_text)
                responsibilities.extend([s.strip() for s in sentences if s.strip() and len(s.strip()) > 5])
        
        return responsibilities
    
    def _extract_requirements(self, text: str) -> List[str]:
        """提取要求"""
        requirements = []
        
        # 提取要求部分
        req_sections = re.findall(r'(任职要求|岗位要求|工作要求|Job Requirements|Requirements).*?(?=工作职责|岗位职责|Job Responsibilities|$)', text, re.DOTALL | re.IGNORECASE)
        
        if req_sections:
            req_text = req_sections[0]
            # 提取列表项
            items = re.findall(r'[•·\-\d+、]+\s*([^\n•·\-\d+、]+)', req_text)
            requirements.extend([item.strip() for item in items if item.strip()])
            
            # 如果没有找到列表项，尝试按句子分割
            if not requirements:
                sentences = re.split(r'[.。;；]', req_text)
                requirements.extend([s.strip() for s in sentences if s.strip() and len(s.strip()) > 5])
        
        return requirements
    
    def _extract_skills(self, text: str) -> List[str]:
        """提取技能"""
        skills = set()
        
        # 使用预定义的技能列表进行匹配
        for skill in self.skill_patterns:
            if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
                skills.add(skill)
        
        # 使用NLP识别技能实体
        doc = nlp(text)
        
        # 提取可能的技能词汇（这里是简化实现）
        skill_candidates = re.findall(r'([A-Za-z+#]+(?:\s[A-Za-z+#]+)*)', text)
        for candidate in skill_candidates:
            if len(candidate) > 1 and candidate.lower() not in [s.lower() for s in skills]:
                skills.add(candidate)
        
        return list(skills)
    
    def _extract_experience_requirement(self, text: str) -> Dict[str, Any]:
        """提取经验要求"""
        experience = {}
        
        # 提取工作年限
        years_match = re.search(r'(\d+)[+]?\s*[年-]\s*(以上)?\s*(工作经验|相关经验)', text)
        if years_match:
            experience['years'] = int(years_match.group(1))
            experience['minimum'] = '以上' in years_match.group(0) if years_match.group(2) else False
        
        return experience
    
    def _extract_education_requirement(self, text: str) -> Dict[str, Any]:
        """提取学历要求"""
        education = {}
        
        # 提取学历要求
        degree_match = re.search(r'(本科|硕士|博士|大专|学士|Bachelor|Master|PhD)[及或以及及其]?(以上)?', text)
        if degree_match:
            education['degree'] = degree_match.group(1)
            education['minimum'] = degree_match.group(2) == '以上' if degree_match.group(2) else False
        
        return education
