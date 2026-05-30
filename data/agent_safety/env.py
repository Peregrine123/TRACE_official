from data.base_env import StaticEnv
from data.utils.safe_utils import compute_score
import re
from typing import Optional, List


class SafetyJudgeEnv(StaticEnv):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def compute_reward(
        cls,
        completions: List[str],
        solution: List[str],
        **kwargs
    ) -> List[float]:
        return [compute_score(c, s) for c, s in zip(completions, solution)]
    
# def compute_score(completion: str, ground_truth: str) -> float:
#     """
#     completion: model ouput
#     ground_truth: reference answer
#     """
#     try:
#         pred = extract_binary_label(completion)
#         gt = extract_binary_label(ground_truth)

#         if pred is None or gt is None:
#             return 0.0
#         return 1.0 if pred == gt else 0.0
#     except Exception as e:
#         print(e)
#         return 0.0


# def extract_binary_label(text: str) -> Optional[str]:
#     """
#     Extract binary label ('0' or '1') from the given text.
#     """
#     if text is None:
#         return None

#     s = text.strip()

#     # (1) strict single-token
#     m = re.fullmatch(r"[01]", s)
#     if m:
#         return m.group(0)

#     # (2) anchored patterns like "Answer: 1"
#     # Answer / Final Answer / Prediction / Label 等
#     patterns = [
#         r"(?im)^\s*answer\s*[:：]?\s*([01])\s*$",
#         r"(?im)^\s*final\s*answer\s*[:：]?\s*([01])\s*$",
#         r"(?im)^\s*label\s*[:：]?\s*([01])\s*$",
#         r"(?im)^\s*prediction\s*[:：]?\s*([01])\s*$",
#         # "... Answer with a single token ... Answer: 1"
#         r"(?i)\banswer\s*[:：]\s*([01])\b",
#         r"(?i)\bfinal\s*answer\s*[:：]\s*([01])\b",
#         r"(?i)\blabel\s*[:：]\s*([01])\b",
#     ]
#     for pat in patterns:
#         m = re.search(pat, s)
#         if m:
#             return m.group(1)

#     # (3) fallback: find all standalone 0/1 tokens
#     tokens = re.findall(r"(?<!\d)([01])(?!\d)", s)
#     if not tokens:
#         return None

#     uniq = set(tokens)
#     if len(uniq) == 1:
#         return tokens[-1]  
#     return None 
