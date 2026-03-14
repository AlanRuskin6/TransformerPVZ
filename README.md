# pvz-agent

Reinforcement learning agent for Plants vs. Zombies, built on Stable Baselines3 with a custom attention-based feature extractor.

## Overview

The agent reads live game state through memory injection (DLL hook), processes it through a multi-modal attention network, and outputs planting decisions in real-time.

## Architecture

```
Game Process (PVZ)
  │
  ├── DLL Hook (C++) ──→ shared memory bridge
  │
  └── Memory Reader (Python) ──→ game state extraction
          │
          ▼
    PVZ Gym Environment
          │
          ▼
    Attention Feature Extractor
      ├── Grid encoder (5×9×13 spatial state)
      ├── Threat-gated injection
      ├── Multi-scale zone tokens (front/mid/back)
      ├── Cross-modal attention (grid ↔ global + cards)
      ├── Short-term memory (LSTM)
      └── Long-term memory bank (retrievable)
          │
          ▼
    PPO Policy ──→ plant/skip actions
```

## Key Components

| Module | Description |
|---|---|
| `models/attention_extractor.py` | Multi-modal attention network with memory augmentation |
| `envs/pvz_env.py` | Gymnasium environment wrapping live game state |
| `game/` | Game state readers (grid, plants, zombies, projectiles) |
| `memory/` | Process memory read/write and DLL injection |
| `hook/` | C++ DLL for in-process game state bridge |
| `engine/action.py` | Action execution (planting, sun collection) |
| `data/` | Game constants, memory offsets, entity definitions |
| `utils/` | Damage calculation, spawn analysis, timing |

## Features

- **Attention-based perception**: custom Transformer encoder over the 5×9 grid with positional embeddings and threat gating
- **Memory augmentation**: LSTM short-term + retrievable long-term memory bank for temporal reasoning
- **Cross-modal fusion**: grid spatial features attend to global game state and card availability
- **Multi-scale spatial awareness**: separate zone tokens for front/mid/back row strategy
- **Live game integration**: reads game memory directly, no screenshot-based approach
- **Spawn prediction encoding**: dedicated encoder for upcoming zombie wave information

## Requirements

- Windows
- Plants vs. Zombies (GOTY edition)
- Python 3.8+
- PyTorch, Stable Baselines3

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train from scratch
python train.py

# Train from checkpoint
python train.py --load models/latest_model.zip
```

## License

MIT
