"""
Angle trigger: compares user_question embedding vs aggregated (action+tool_output).
- angle > threshold_deg => trigger weaver
- angle > hard_reject_deg => optional hard reject signal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray: ...


@dataclass
class AngleSignal:
    cos_sim: float
    angle_deg: float
    triggered: bool
    hard_reject: bool


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


class AngleGate:
    def __init__(
        self,
        embedder: Embedder,
        threshold_deg: float = 90.0,
        hard_reject_deg: Optional[float] = None,
    ):
        self.embedder = embedder
        self.threshold_deg = float(threshold_deg)
        self.hard_reject_deg = float(hard_reject_deg) if hard_reject_deg is not None else None

    def __call__(self, user_question: str, turn_text: str) -> AngleSignal:
        v = self.embedder.embed([user_question, turn_text])
        c = _cos(v[0], v[1])

        c_clip = max(-1.0, min(1.0, c))
        angle = float(np.degrees(np.arccos(c_clip)))
        trig = angle >= self.threshold_deg
        hard = False
        if self.hard_reject_deg is not None and angle >= self.hard_reject_deg:
            hard = True
        return AngleSignal(cos_sim=c, angle_deg=angle, triggered=trig, hard_reject=hard)
