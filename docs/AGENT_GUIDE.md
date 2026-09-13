# FlySweeper: AI Agent & Developer Operating Guide

This guide provides context, architectural invariants, code pathways, and operational workflows for autonomous AI agents and human engineers working on FlySweeper.

---

## 1. System Overview

FlySweeper couples a biologically measured Drosophila melanogaster (fruit fly) connectome subcircuit with Microsoft Minesweeper. 

### Key Concept: Biological Reservoir Computing
Rather than training all synaptic connections in a neural network from scratch:
1. The biological connectome graph ($W_{connectome}$, 80 neurons, 1,296 synapses) remains **frozen**. It acts as a recurrent continuous-time dynamical reservoir.
2. A linear sensory projection layer ($W_{in}$) maps Minesweeper board states to 32 visual input neurons.
3. A linear motor readout layer ($W_{out}, b_{out}$) maps 16 descending motor neurons to action logits across the board tiles.
4. Only $W_{in}$, $W_{out}$, and $b_{out}$ (1,764 total float32 parameters) are optimized using the Cross-Entropy Method (CEM).

---

## 2. Fast Command Reference

All commands must be executed from the repository root:

| Objective | Command |
|---|---|
| Run Headless Simulation Test | `python -c "from minesweeper_env import MinesweeperEnv; env = MinesweeperEnv(); print(env.reset())"` |
| Evaluate Existing Checkpoint | `python train.py --eval-only --model-path checkpoints/best_fly_model.pt --eval-episodes 20` |
| Continue / Fine-Tune Checkpoint | `python train.py --resume checkpoints/best_fly_model.pt --generations 30 --pop-size 30 --elites 5 --init-std 0.15` |
| Train New Model from Scratch | `python train.py --generations 50 --pop-size 40 --elites 6 --model-path checkpoints/best_fly_model.pt` |
| Export PyTorch Checkpoint to WebGL | `python export_weights.py --model checkpoints/best_fly_model.pt --output fly_model_web.json` |
| Launch 3D WebGL HUD Daemon | `python serve.py` (Spectator UI: http://localhost:8000/web_visualizer.html) |
| Launch Native Pygame HUD | `python visualizer.py` |

---

## 3. Codebase Map & Responsibilities

```text
fly-brain-minesweeper/
├── checkpoints/
│   └── best_fly_model.pt       # Production PyTorch weights checkpoint
├── data/
│   └── malecns_circuit.json    # MaleCNS v1.0 biological graph (80 cells, 1,296 synapses)
├── docs/
│   ├── AGENT_GUIDE.md          # This file: context, workflows, invariants
│   ├── CHECKPOINT_SPEC.md      # PyTorch and WebGL serialization specifications
│   └── TRAINING_AND_IMPROVEMENT.md # CEM optimization, curriculum, fine-tuning guide
├── convert_circuit.py          # Script converting raw MaleCNS coordinates to normalized space
├── export_weights.py           # CLI tool exporting PyTorch checkpoints to WebGL JSON format
├── fly_brain.py                # Biological neural network model and recurrent dynamics
├── fly_model_web.json          # WebGL deployment weights loaded by web_visualizer.html
├── minesweeper_env.py          # Fast, headless gym-compatible Minesweeper environment
├── serve.py                    # Static HTTP server running on port 8000
├── train.py                    # Cross-Entropy Method (CEM) evolutionary trainer
├── visualizer.py               # Native Pygame desktop visualizer
└── web_visualizer.html         # High-density Three.js 3D neural activity spectator HUD
```

---

## 4. Architectural Invariants (Must Not Break)

When making modifications or building on top of this repository, ensure all five invariants are preserved:

1. **Connectome Topology Immutability**:
   The internal synaptic connectivity matrix $W_{connectome}$ in `data/malecns_circuit.json` represents physical electron-microscopy contacts. Do not add random weights or shuffle synapses during standard training runs. Learning happens strictly at sensory projection ($W_{in}$) and motor readout ($W_{out}, b_{out}$).

2. **Neurotransmitter Sign Conservation**:
   Synaptic polarity is constrained by biological neurotransmitter identity:
   * Acetylcholine (ACh): $+1$ (excitatory)
   * GABA / Glutamate (GABA/GLUT): $-1$ (inhibitory)
   Any addition of biological neurons must assign valid biological neurotransmitter signs.

3. **Valid Action Masking**:
   Minesweeper decisions must always apply `env.get_valid_action_mask()` prior to tile selection. Revealed tiles must be masked with large negative values ($-\infty$) to prevent the agent from repeatedly selecting already exposed cells.

4. **Observation Space Format**:
   The board state is fed to the model as a flattened tensor of size `(rows * cols,)`:
   * Unrevealed tile: `-1.0`
   * Revealed tile with 0 adjacent mines: `0.0`
   * Revealed tile with $k$ adjacent mines ($1 \le k \le 8$): `$k / 8.0$` (normalized between 0.0 and 1.0)
   * Flagged tile: `-2.0`

5. **Path Portability**:
   Always use relative paths or `os.path.join(basedir, ...)` anchored to the file location. Never hardcode absolute user directory paths.

---

## 5. Standard Agent Tasks

### Task A: Benchmark an Existing Checkpoint
An agent should run evaluation across at least 20 to 50 independent board seeds to calculate mean reward, win rate, and safe reveal percentage:

```bash
python train.py --eval-only --model-path checkpoints/best_fly_model.pt --eval-episodes 50
```

### Task B: Fine-Tune / Improve a Checkpoint
To push performance beyond the current 96.4% safe reveal rate:
1. Load existing weights using `--resume`.
2. Use a tighter exploration variance (`--init-std 0.15` or `0.10`).
3. Run 30 to 50 generations with an elite pool of 4 to 6:

```bash
python train.py --resume checkpoints/best_fly_model.pt --generations 30 --pop-size 40 --elites 6 --init-std 0.15
```

### Task C: Export Weights to Web Visualizer
Whenever a better checkpoint is found:
1. Export the `.pt` file to `fly_model_web.json`:
   ```bash
   python export_weights.py --model checkpoints/best_fly_model.pt --output fly_model_web.json
   ```
2. Verify the WebGL HUD loads cleanly by checking `fly_model_web.json` size and syntax:
   ```bash
   python -c "import json; data = json.load(open('fly_model_web.json')); print('Web model nodes:', len(data['nodes']), 'edges:', len(data['edges']))"
   ```

### Task D: Scale to Larger Boards (Curriculum Training)
To adapt FlySweeper to intermediate grids (e.g. 8x8 with 10 mines):
1. Initialize with larger dimensions:
   ```bash
   python train.py --rows 8 --cols 8 --mines 10 --generations 50 --pop-size 50 --elites 8 --model-path checkpoints/best_fly_8x8.pt
   ```
2. Re-export weights specifying the new dimensions.
