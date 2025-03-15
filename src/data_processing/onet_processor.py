import itertools
import json
import os

import pandas as pd
from networkx import DiGraph


class OnetProcessor:
    """O*NET数据处理器"""

    def __init__(self, data_dir: str):
        """
        初始化处理器

        Args:
            data_dir: O*NET数据目录
        """
        self.data_dir = data_dir
        self.occupation_data = None
        self.skills_data = None
        self.knowledge_data = None
        self.abilities_data = None
        self.work_activities_data = None
        self.skills_to_work_activities = None
        self.tech_skills_data = None
        self.task_statements_data = None
        self.tools_used_data = None
        self.task_ratings_data = None
        self.education_data = None
        self.work_context_data = None

    def load_data(self) -> None:
        """加载O*NET数据"""
        data_files = {
            "occupation_data": "Occupation Data.xlsx",
            "skills_data": "Skills.xlsx",
            "knowledge_data": "Knowledge.xlsx",
            "abilities_data": "Abilities.xlsx",
            "work_activities_data": "Work Activities.xlsx",
            "skills_to_work_activities": "Skills to Work Activities.xlsx",
            "tech_skills_data": "Technology Skills.xlsx",
            "task_statements_data": "Task Statements.xlsx",
            "tools_used_data": "Tools Used.xlsx",
            "task_ratings_data": "Task Ratings.xlsx",
            "education_data": "Education, Training, and Experience.xlsx",
            "work_context_data": "Work Context.xlsx",
        }
        data_frames = {}
        for attr, file_name in data_files.items():
            file_path = os.path.join(self.data_dir, file_name)
            try:
                data_frames[attr] = pd.read_excel(file_path)
                print(f"成功加载: {file_name}")
            except FileNotFoundError:
                print(f"警告: {file_name} 文件不存在")
                data_frames[attr] = None
            except Exception as e:
                print(f"错误: {file_name} 加载失败，原因: {e}")
                data_frames[attr] = None
        self.occupation_data = data_frames["occupation_data"]
        self.skills_data = data_frames["skills_data"]
        self.knowledge_data = data_frames["knowledge_data"]
        self.abilities_data = data_frames["abilities_data"]
        self.work_activities_data = data_frames["work_activities_data"]
        self.skills_to_work_activities = data_frames["skills_to_work_activities"]
        self.tech_skills_data = data_frames["tech_skills_data"]
        self.task_statements_data = data_frames["task_statements_data"]
        self.tools_used_data = data_frames["tools_used_data"]
        self.task_ratings_data = data_frames["task_ratings_data"]
        self.education_data = data_frames["education_data"]
        self.work_context_data = data_frames["work_context_data"]
        print("数据加载完成")

    def _process_data(self, df: pd.DataFrame, column_mapping: dict) -> pd.DataFrame:
        """处理数据，重命名列并去重"""
        try:
            df.rename(columns=column_mapping, inplace=True)
            return df.drop_duplicates()
        except Exception as e:
            print(f"错误: 数据处理失败，原因: {e}")
            return df

    def _filter_by_column_values(self, df: pd.DataFrame, column: str, values: list) -> pd.DataFrame:
        """根据列值过滤DataFrame"""
        try:
            return df[df[column].isin(values)]
        except Exception as e:
            print(f"错误: 数据过滤失败，原因: {e}")
            return df

    def _process_occupation_data(self) -> pd.DataFrame:
        """处理职业数据"""
        if self.occupation_data is None:
            return pd.DataFrame()
        occupations = self.occupation_data[["O*NET-SOC Code", "Title", "Description"]].copy()
        occupations = self._process_data(
            occupations, {"O*NET-SOC Code": "occupation_code", "Title": "title", "Description": "description"}
        )
        return occupations

    def _process_single_attribute_type(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理单类型属性数据，如 Knowledge, Abilities, Work Activities, Work Context"""
        return self._process_data(
            data[["Element ID", "Element Name"]].copy(), {"Element ID": "id", "Element Name": "name"}
        )

    def process_occupations(self) -> pd.DataFrame:
        """处理职业数据"""
        if self.occupation_data is None:
            self.load_data()
        return self._process_occupation_data()

    def process_skills(self) -> pd.DataFrame:
        """处理技能数据"""
        unique_skills = self.skills_data[["Element ID", "Element Name"]].drop_duplicates()
        unique_skills.columns = ["skill_id", "skill_name"]
        return unique_skills

    def process_occupation_skills(self) -> pd.DataFrame:
        """处理职业-技能关系数据"""
        if self.skills_data is None:
            self.load_data()
        occupation_skills = self.skills_data[["O*NET-SOC Code", "Element ID", "Scale ID", "Data Value"]].copy()
        occupation_skills = self._process_data(
            occupation_skills,
            {
                "O*NET-SOC Code": "occupation_code",
                "Element ID": "skill_id",
                "Scale ID": "scale_id",
                "Data Value": "value",
            },
        )
        occupation_skills = self._filter_by_column_values(occupation_skills, "scale_id", ["IM", "LV"])
        occupation_skills_wide = occupation_skills.pivot_table(
            index=["occupation_code", "skill_id"], columns="scale_id", values="value"
        ).reset_index()
        occupation_skills_wide.columns.name = None
        return occupation_skills_wide

    def process_tech_skills(self) -> pd.DataFrame:
        """处理技术技能数据"""
        if self.tech_skills_data is None:
            return pd.DataFrame()
        tech = self.tech_skills_data[["O*NET-SOC Code", "Commodity Title", "Hot Technology"]].copy()
        tech.columns = ["occupation_code", "tech_name", "is_hot"]
        return tech

    def process_education(self) -> pd.DataFrame:
        """处理教育要求数据"""
        if self.education_data is None:
            return pd.DataFrame()
        edu = self.education_data[self.education_data["Scale ID"] == "Education Level"][
            ["O*NET-SOC Code", "Data Value"]
        ]
        edu.columns = ["occupation_code", "education_level"]
        return edu

    def process_knowledge(self) -> pd.DataFrame:
        """处理知识数据"""
        if self.knowledge_data is None:
            return pd.DataFrame()
        knowledge = self._process_single_attribute_type(self.knowledge_data)
        knowledge.columns = ["knowledge_id", "knowledge_name"]
        return knowledge

    def process_abilities(self) -> pd.DataFrame:
        """处理能力数据"""
        if self.abilities_data is None:
            return pd.DataFrame()
        abilities = self._process_single_attribute_type(self.abilities_data)
        abilities.columns = ["ability_id", "ability_name"]
        return abilities

    def process_work_activities(self) -> pd.DataFrame:
        """处理工作活动数据"""
        if self.work_activities_data is None:
            return pd.DataFrame()
        activities = self._process_single_attribute_type(self.work_activities_data)
        activities.columns = ["activity_id", "activity_name"]
        return activities

    def process_work_context(self) -> pd.DataFrame:
        """处理工作情境数据"""
        if self.work_context_data is None:
            return pd.DataFrame()
        context = self._process_single_attribute_type(self.work_context_data)
        context.columns = ["context_id", "context_name"]
        return context

    def build_skill_graph(self) -> DiGraph:
        """
        构建技能关系图
        Returns:
            技能关系有向图
        """
        G = DiGraph()
        unique_skills = self.process_skills()
        for _, row in unique_skills.iterrows():
            G.add_node(row["skill_id"], name=row["skill_name"])
        if self.skills_to_work_activities is not None:
            skill_activity = self.skills_to_work_activities[["Skills Element ID", "Work Activities Element ID"]].copy()
            skill_activity.columns = ["skill_id", "activity_id"]
            activity_to_skills = {}
            for _, row in skill_activity.iterrows():
                activity_id = row["activity_id"]
                skill_id = row["skill_id"]
                if activity_id not in activity_to_skills:
                    activity_to_skills[activity_id] = []
                activity_to_skills[activity_id].append(skill_id)
            for skills in activity_to_skills.values():
                if len(skills) > 1:
                    for skill_pair in itertools.combinations(skills, 2):
                        G.add_edge(skill_pair[0], skill_pair[1], weight=1)
                        G.add_edge(skill_pair[1], skill_pair[0], weight=1)
        else:
            occupation_skills = self.process_occupation_skills()
            occ_to_skills = {}
            for _, row in occupation_skills.iterrows():
                occ_code = row["occupation_code"]
                skill_id = row["skill_id"]
                if occ_code not in occ_to_skills:
                    occ_to_skills[occ_code] = set()
                occ_to_skills[occ_code].add(skill_id)
            for skills in occ_to_skills.values():
                if len(skills) > 1:
                    skills_list = list(skills)
                    for skill_pair in itertools.combinations(skills_list, 2):
                        G.add_edge(skill_pair[0], skill_pair[1], weight=1)
                        G.add_edge(skill_pair[1], skill_pair[0], weight=1)
        return G

    def save_processed_data(self, output_dir: str):
        """
        保存处理后的数据

        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        occupations = self.process_occupations()
        skills = self.process_skills()
        occupation_skills = self.process_occupation_skills()
        tech_skills = self.process_tech_skills()
        education = self.process_education()
        knowledge = self.process_knowledge()
        abilities = self.process_abilities()
        work_activities = self.process_work_activities()
        work_context = self.process_work_context()
        skill_graph = self.build_skill_graph()

        occupations.to_csv(os.path.join(output_dir, "occupations.csv"), index=False)
        skills.to_csv(os.path.join(output_dir, "skills.csv"), index=False)
        occupation_skills.to_csv(os.path.join(output_dir, "occupation_skills.csv"), index=False)
        tech_skills.to_csv(os.path.join(output_dir, "tech_skills.csv"), index=False)
        education.to_csv(os.path.join(output_dir, "education.csv"), index=False)
        knowledge.to_csv(os.path.join(output_dir, "knowledge.csv"), index=False)
        abilities.to_csv(os.path.join(output_dir, "abilities.csv"), index=False)
        work_activities.to_csv(os.path.join(output_dir, "work_activities.csv"), index=False)
        work_context.to_csv(os.path.join(output_dir, "work_context.csv"), index=False)

        nodes = [{"id": node[0], **node[1]} for node in skill_graph.nodes(data=True)]
        edges = [{"source": edge[0], "target": edge[1], **edge[2]} for edge in skill_graph.edges(data=True)]
        with open(os.path.join(output_dir, "skill_graph.json"), "w") as f:
            json.dump({"nodes": nodes, "edges": edges}, f)
        print(f"处理后的数据已保存到 {output_dir}")

    def generate_job_skill_dataset(self, output_dir: str):
        """
        生成职位-技能数据集

        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        if any(
            df is None or df.empty
            for df in [self.process_occupations(), self.process_skills(), self.process_occupation_skills()]
        ):
            print("无法生成职位-技能数据集，缺少必要的数据文件")
            return

        occupations = self.process_occupations()
        skills = self.process_skills()
        occupation_skills = self.process_occupation_skills()

        important_skills = occupation_skills[occupation_skills["IM"] >= 3.0].copy()
        important_skills = important_skills.merge(
            occupations[["occupation_code", "title"]], on="occupation_code", how="left"
        )
        important_skills = important_skills.merge(skills[["skill_id", "skill_name"]], on="skill_id", how="left")

        job_skill_matrix = occupation_skills.pivot_table(
            values="IM", index="occupation_code", columns="skill_id", aggfunc="first", fill_value=0
        )
        job_skill_matrix.to_csv(os.path.join(output_dir, "job_skill_matrix.csv"))

        job_metadata = occupations.set_index("occupation_code")
        job_metadata.to_csv(os.path.join(output_dir, "job_metadata.csv"))

        skill_metadata = skills.set_index("skill_id")
        skill_metadata.to_csv(os.path.join(output_dir, "skill_metadata.csv"))

        occupation_skill_freq = {}
        for occ_code in occupations["occupation_code"].unique():
            occ_skills = occupation_skills[occupation_skills["occupation_code"] == occ_code]
            total_importance = occ_skills["IM"].sum()
            if total_importance > 0:
                skill_freq = occ_skills.set_index("skill_id")["IM"] / total_importance
                occupation_skill_freq[occ_code] = skill_freq.to_dict()

        matching_samples = []
        occupation_sample_counts = {}

        for _, row in important_skills.iterrows():
            occupation_code = row["occupation_code"]
            if occupation_code not in occupation_sample_counts:
                occupation_sample_counts[occupation_code] = {"positive": 0, "negative": 0}
            skill_freq = occupation_skill_freq.get(occupation_code, {}).get(row["skill_id"], 0)
            matching_samples.append(
                {
                    "occupation_code": occupation_code,
                    "skill_id": row["skill_id"],
                    "title": row["title"],
                    "skill_name": row["skill_name"],
                    "importance": row["IM"],  # 修改为 importance
                    "level": row["LV"],  # 修改为 level
                    "relative_importance": skill_freq,
                    "match": 1,
                }
            )
            occupation_sample_counts[occupation_code]["positive"] += 1

        for occupation_code, counts in occupation_sample_counts.items():
            occupation_skill_ids = important_skills[important_skills["occupation_code"] == occupation_code][
                "skill_id"
            ].tolist()
            non_occupation_skills = skills[~skills["skill_id"].isin(occupation_skill_ids)]
            num_negative_samples = counts["positive"]
            if len(non_occupation_skills) < num_negative_samples:
                negative_skills = non_occupation_skills.sample(n=num_negative_samples, replace=True)
            else:
                negative_skills = non_occupation_skills.sample(n=num_negative_samples)
            occupation_title = occupations[occupations["occupation_code"] == occupation_code]["title"].iloc[0]
            for _, skill_row in negative_skills.iterrows():
                matching_samples.append(
                    {
                        "occupation_code": occupation_code,
                        "skill_id": skill_row["skill_id"],
                        "title": occupation_title,
                        "skill_name": skill_row["skill_name"],
                        "importance": 0,  # 修改为 importance
                        "level": 0,  # 修改为 level
                        "relative_importance": 0,
                        "match": 0,
                    }
                )
                counts["negative"] += 1

        matching_df = pd.DataFrame(matching_samples)
        matching_df = matching_df.sample(frac=1).reset_index(drop=True)
        matching_df.to_csv(os.path.join(output_dir, "skill_matching_dataset.csv"), index=False)

        total_positive = sum(counts["positive"] for counts in occupation_sample_counts.values())
        total_negative = sum(counts["negative"] for counts in occupation_sample_counts.values())
        print(f"职位-技能匹配数据集已保存到 {output_dir}")
        print(
            "数据集统计:\n"
            f"  总样本数: {len(matching_df)}\n"
            f"  正样本: {total_positive}, 负样本: {total_negative}\n"
            f"  正负样本比例: {total_positive/total_negative:.2f}"
        )
