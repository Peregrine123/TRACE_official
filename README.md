# TRACE: Trajectory Risk-Aware Compression for Long-Horizon Agent Safety

This is the official implementation of **TRACE**, a Compressor-Reader framework for long-horizon LLM agent safety detection. TRACE reframes agent safety as trajectory-level evidence compression: the Compressor encodes the full trajectory into a compact latent evidence state, and the Reader judges the raw trajectory with this latent state as a safety reference.

## Framework Overview

```
Trajectory τ = (x₁, x₂, ..., xₗ)
        │
        ▼
┌─────────────────┐
│   Compressor     │  ← Learnable query tokens + LoRA
│   C_φ(τ) → S    │  ← Aggregates dispersed risk cues
└────────┬────────┘
         │ latent evidence state S
         ▼
┌─────────────────┐
│     Reader       │  ← Frozen backbone + classification head
│  [E_τ; W(S)]    │  ← Raw trajectory + safety reference
│   → p̂ (unsafe)  │
└─────────────────┘
```

TRACE addresses three evidence patterns in long-horizon trajectories:
- **Sparse evidence**: only a few steps carry risk signals
- **Delayed evidence**: risk consequences surface after many steps
- **Compositional evidence**: individually safe steps become threatening when combined

## Setup

```bash
conda create -n trace python=3.12
conda activate trace
pip install -r requirements.txt
```

## Data

Place safety trajectory data in `data/agent_safety/`. The data format is JSONL with fields:
- `prompt`: the trajectory text (concatenated turns)
- `completion`: ground-truth label (`"0"` for safe, `"1"` for unsafe)

## Training

```bash
bash scripts/trace_safety_train.sh
```

Key training configurations:
- `model.max_prompt_aug_num`: number of prompt augmentation points (default: 1)
- `model.max_inference_aug_num`: number of inference augmentation points (default: 5)
- `model.weaver.prompt_latents_len`: number of Compressor query tokens for prompt (default: 16)
- `model.weaver.inference_latents_len`: number of Compressor query tokens for inference (default: 8)

## Evaluation

Modify the checkpoint path in the script, then run:

```bash
bash scripts/trace_safety_test.sh
```

## Angle Search (Semantic Gate Tuning)

```bash
bash scripts/angle_search.sh
```

This sweeps over angle thresholds to find the optimal semantic gate configuration.

## Project Structure

```
main_safety.py              # Entry point
common/                     # Config and logging utilities
configs/                    # YAML configurations
data/
  agent_safety/             # Safety trajectory data
  base_builder.py           # Dataset builder base class
  base_env.py               # Environment base class
  utils/safe_utils.py       # Scoring utilities
safety/
  model/
    modeling_memgen.py       # TRACE model (Compressor + Reader)
    modeling_utils.py        # Generation and LoRA mixins
    weaver.py                # Compressor module
    trigger.py               # Trigger module
    configuration_memgen.py  # Model configuration
  runner.py                  # Training and evaluation runner
  runner_merge.py            # Runner with semantic gate routing
  utils.py                   # Semantic gate utilities
  angle_gate.py              # Angle-based semantic gate
  embeddings.py              # Embedding utilities
memgen/
  utils.py                   # Shared utilities
scripts/                     # Shell scripts for training/evaluation
```

## Acknowledgements

This codebase builds upon the [MemGen](https://github.com/NL2G/MemGen) framework.
