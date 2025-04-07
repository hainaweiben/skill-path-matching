"""
Skill Matching Model module

This module implements the main skill matching model that combines skill path encoding
and occupation encoding to predict the matching degree between occupations and skills.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from skill_path_matching.src.train.models.skill_path_encoder import SkillPathEncoder
from skill_path_matching.src.train.models.occupation_encoder import OccupationEncoder

{{ ... }}