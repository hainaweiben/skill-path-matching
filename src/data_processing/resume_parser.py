"""
简历解析模块
用于从简历文本中提取结构化信息，包括个人信息、教育背景、工作经验和技能
"""

import re
import nltk
import spacy
from typing import Dict, List, Any
import pandas as pd

# 加载spaCy模型
try:
    nlp = spacy.load("zh_core_web_sm")
except:
    nlp = spacy.load("en_core_web_sm")

class ResumeParser:
    """简历解析类，用于从简历文本中提取结构化信息"""
    
    def __init__(self, skill_taxonomy_path: str = None):
        """
        初始化简历解析器
        
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
        解析简历文本
        
        Args:
            text: 简历文本
            
        Returns:
            包含解析结果的字典
        """
        # 基本信息提取
        basic_info = self._extract_basic_info(text)
        
        # 教育背景提取
        education = self._extract_education(text)
        
        # 工作经验提取
        experience = self._extract_experience(text)
        
        # 技能提取
        skills = self._extract_skills(text)
        
        return {
            "basic_info": basic_info,
            "education": education,
            "experience": experience,
            "skills": skills
        }
    
    def _extract_basic_info(self, text: str) -> Dict[str, str]:
        """提取基本信息"""
        # 示例实现，实际应用中需要更复杂的规则或机器学习模型
        info = {}
        
        # 提取姓名
        name_match = re.search(r'姓名[：:]\s*([^\n,，]+)', text)
        if name_match:
            info['name'] = name_match.group(1).strip()
        
        # 提取联系方式
        email_match = re.search(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', text)
        if email_match:
            info['email'] = email_match.group(1)
        
        phone_match = re.search(r'电话[：:]\s*([0-9-+]{7,15})', text) or re.search(r'手机[：:]\s*([0-9-+]{7,15})', text)
        if phone_match:
            info['phone'] = phone_match.group(1)
        
        return info
    
    def _extract_education(self, text: str) -> List[Dict[str, str]]:
        """提取教育背景"""
        education_list = []
        
        # 使用NLP模型识别教育机构和时间
        doc = nlp(text)
        
        # 这里是简化实现，实际应用中需要更复杂的逻辑
        edu_sections = re.findall(r'(教育背景|学历|Education).*?(?=工作经验|技能|$)', text, re.DOTALL | re.IGNORECASE)
        
        if edu_sections:
            edu_text = edu_sections[0]
            # 提取学校和学位
            schools = re.findall(r'([^\n,，:：]+大学|[^\n,，:：]+学院)', edu_text)
            degrees = re.findall(r'(学士|硕士|博士|本科|研究生|PhD|Master|Bachelor)', edu_text)
            
            for i, school in enumerate(schools):
                edu_info = {"institution": school}
                if i < len(degrees):
                    edu_info["degree"] = degrees[i]
                education_list.append(edu_info)
        
        return education_list
    
    def _extract_experience(self, text: str) -> List[Dict[str, str]]:
        """提取工作经验"""
        experience_list = []
        
        # 提取工作经验部分
        exp_sections = re.findall(r'(工作经验|工作经历|Work Experience).*?(?=教育背景|技能|$)', text, re.DOTALL | re.IGNORECASE)
        
        if exp_sections:
            exp_text = exp_sections[0]
            # 提取公司名称
            companies = re.findall(r'([^\n,，:：]+公司|[^\n,，:：]+集团|[^\n,，:：]+企业|[^\n,，:：]+Corporation|[^\n,，:：]+Inc\.)', exp_text)
            
            for company in companies:
                exp_info = {"company": company}
                # 提取职位
                position_match = re.search(r'职位[：:]\s*([^\n]+)', exp_text)
                if position_match:
                    exp_info["position"] = position_match.group(1).strip()
                
                # 提取时间段
                time_match = re.search(r'(\d{4}[年/-]\d{1,2}[月/-]?\s*[-至到~]\s*\d{4}[年/-]\d{1,2}[月/-]?)', exp_text)
                if time_match:
                    exp_info["time_period"] = time_match.group(1)
                
                experience_list.append(exp_info)
        
        return experience_list
    
    def _extract_skills(self, text: str) -> List[str]:
        """提取技能"""
        skills = []
        
        # 使用预定义的技能列表进行匹配
        for skill in self.skill_patterns:
            if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
                skills.append(skill)
        
        # 提取技能部分
        skill_sections = re.findall(r'(技能|Skills|专业技能).*?(?=教育背景|工作经验|$)', text, re.DOTALL | re.IGNORECASE)
        
        if skill_sections:
            skill_text = skill_sections[0]
            # 使用NLP识别技能实体
            doc = nlp(skill_text)
            
            # 提取可能的技能词汇（这里是简化实现）
            skill_candidates = re.findall(r'([A-Za-z+#]+(?:\s[A-Za-z+#]+)*)', skill_text)
            for candidate in skill_candidates:
                if len(candidate) > 1 and candidate.lower() not in [s.lower() for s in skills]:
                    skills.append(candidate)
        
        return skills
