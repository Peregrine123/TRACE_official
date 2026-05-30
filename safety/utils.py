"""
Utility helpers for semantic gate routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Iterable, List, Optional, Union
import os
from safety.angle_gate import AngleGate, AngleSignal
from safety.embeddings import HashEmbedder, HFEmbedder


def _normalize_text(value: Optional[str]) -> str:
    return value.strip() if isinstance(value, str) else ""


def format_turns(turns: Optional[Iterable[dict]]) -> str:
    """Flatten the agent/tool interaction turns into a single text block."""
    if not turns:
        return ""
    parts: List[str] = []
    for turn in turns:
        agent_thought = _normalize_text(turn.get("agent_thought"))
        action = _normalize_text(turn.get("action"))
        tool_output = _normalize_text(turn.get("tool_output"))

        if agent_thought:
            parts.append(f"agent_thought: {agent_thought}")
        if action:
            parts.append(f"action: {action}")
        if tool_output:
            parts.append(f"tool_output: {tool_output}")
    return "\n".join(parts).strip()


def text_entropy(text: str) -> float:
    """Compute simple Shannon entropy over whitespace tokens."""
    tokens = [t for t in text.split() if t]
    if not tokens:
        return 0.0

    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    total = float(len(tokens))
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * log2(p)
    return entropy

import re
from typing import List

_TURN_SPLIT_RE = re.compile(r"(?m)^\s*Turn\s+\d+\s*:\s*$")

def split_turns_text(turns_text: str) -> List[str]:
    """
    Split a single big turns string into per-turn chunks.

    Input example:
      "Turn 1:\n... \n\nTurn 2:\n...\n"
    Output:
      ["Turn 1:\n...", "Turn 2:\n..."]
    """
    if not turns_text:
        return []

    # Find all "Turn k:" headers and split by their positions
    matches = list(_TURN_SPLIT_RE.finditer(turns_text))
    if not matches:
        # If format doesn't contain headers, treat as single chunk
        return [turns_text.strip()] if turns_text.strip() else []

    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(turns_text)
        chunk = turns_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

@dataclass
class SemanticGateSignal:
    angle: AngleSignal                      
    entropy: float
    triggered: bool
    adjacent_triggered: bool = False        
    adjacent_max_angle_deg: Optional[float] = None
    adjacent_max_pair: Optional[tuple] = None  

class SemanticGate:
    """
    Semantic trigger that combines embedding angle and entropy heuristics.

    Triggered if:
      - angle(user_question, whole_turn_text) >= threshold
        OR
      - (optional) any angle(turn_i, turn_{i+1}) >= adjacent_threshold
    """
    def __init__(
        self,
        embedder,
        angle_threshold_deg: float = 90.0,
        entropy_threshold: float = 3.5,
        hard_reject_deg: Optional[float] = None,
        enable_adjacent_turn_gate: bool = False,
        adjacent_threshold_deg: Optional[float] = None,  # if None, reuse angle_threshold_deg
    ):
        self.angle_gate = AngleGate(
            embedder=embedder,
            threshold_deg=angle_threshold_deg,
            hard_reject_deg=hard_reject_deg,
        )

        # Adjacent gate uses the same embedder but can have different threshold/hard reject if you want
        self.enable_adjacent_turn_gate = bool(enable_adjacent_turn_gate)
        adj_th = angle_threshold_deg if adjacent_threshold_deg is None else float(adjacent_threshold_deg)
        self.adjacent_gate = AngleGate(
            embedder=embedder,
            threshold_deg=adj_th,
            hard_reject_deg=hard_reject_deg,
        )

        self.entropy_threshold = float(entropy_threshold)

    def _normalize_turns(self, turns: Union[str, Iterable[dict], None]) -> str:
        """
        Turns can be:
          - a big formatted string (your example)
          - an iterable of dict (if you later switch to structured turns)
        """
        if turns is None:
            return ""
        if isinstance(turns, str):
            return turns
        # fallback: if it's structured, join it into text
        parts = []
        for t in turns:
            if isinstance(t, dict):
                # Customize this if you have a canonical schema
                parts.append(str(t))
            else:
                parts.append(str(t))
        return "\n".join(parts)

    def _adjacent_turn_trigger(self, turns_text: str):
        """
        Return: (adjacent_triggered, max_angle, max_pair)
        """
        chunks = split_turns_text(turns_text)
        if len(chunks) < 2:
            return False, None, None

        max_angle = -1.0
        max_pair = None
        any_trig = False

        for i in range(len(chunks) - 1):
            sig = self.adjacent_gate(chunks[i], chunks[i + 1])
            if sig.angle_deg > max_angle:
                max_angle = sig.angle_deg
                max_pair = (i, i + 1)
            if sig.triggered or sig.hard_reject:
                any_trig = True

        return any_trig, float(max_angle), max_pair

    def __call__(self, user_question: str, turns: Union[str, Iterable[dict], None]) -> SemanticGateSignal:
        turns_text = self._normalize_turns(turns)

        # 1) overall: user_question vs whole turns
        angle_signal = self.angle_gate(user_question, turns_text)

        # 2) optional: adjacent turn angle jumps
        adjacent_triggered = False
        adjacent_max_angle = None
        adjacent_max_pair = None
        if self.enable_adjacent_turn_gate:
            adjacent_triggered, adjacent_max_angle, adjacent_max_pair = self._adjacent_turn_trigger(turns_text)

        # 3) final trigger (entropy currently disabled in your code)
        triggered = bool(angle_signal.triggered or angle_signal.hard_reject or adjacent_triggered)

        return SemanticGateSignal(
            angle=angle_signal,
            entropy=0.0,
            triggered=triggered,
            adjacent_triggered=adjacent_triggered,
            adjacent_max_angle_deg=adjacent_max_angle,
            adjacent_max_pair=adjacent_max_pair,
        )


def build_embedder(cfg: dict):
    embedder_type = cfg.get("embedder_type", "hash")
    if embedder_type == "hf":
        model_name = cfg.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
        # if model_name.startswith("meta-llama/"):
        #     model_name = "Qwen/Qwen2.5-7B-Instruct"
        max_length = cfg.get("max_length", 1024)
        return HFEmbedder(model_name=model_name, max_length=max_length)
    return HashEmbedder(dim=cfg.get("dim", 384))


def build_semantic_gate(cfg: dict) -> SemanticGate:
    embedder = build_embedder(cfg)
    return SemanticGate(
        embedder=embedder,
        angle_threshold_deg=cfg.get("angle_threshold_deg", 0.0),
        entropy_threshold=cfg.get("entropy_threshold", 10.0),
        hard_reject_deg=cfg.get("hard_reject_deg"),
        enable_adjacent_turn_gate=cfg.get("enable_adjacent_turn_gate", False),
        adjacent_threshold_deg=cfg.get("adjacent_threshold_deg"),  # None -> reuse angle_threshold_deg
    )

def print_trainable_modules(model, max_lines=300, only_lora=False):
    total = 0
    trainable = 0
    rows = []

    for n, p in model.named_parameters():
        numel = p.numel()
        total += numel
        if p.requires_grad:
            if only_lora and ("lora" not in n.lower()):
                continue
            trainable += numel
            rows.append((n, tuple(p.shape), str(p.dtype), str(p.device), numel))

    print(f"[total params]     {total:,}")
    print(f"[trainable params] {trainable:,} ({(trainable/total*100 if total else 0):.6f}%)")
    print("-" * 100)
    for i, (n, shape, dtype, device, numel) in enumerate(rows[:max_lines]):
        print(f"{i:04d} | {n:70s} | {shape!s:18s} | {dtype:10s} | {device:12s} | {numel:,}")
    if len(rows) > max_lines:
        print(f"... truncated {len(rows)-max_lines} lines")