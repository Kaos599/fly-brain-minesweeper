# FlySweeper Checkpoint Specification

This document details the serialization formats, tensor geometries, and parameter mapping for FlySweeper model checkpoints.

---

## 1. Checkpoint Formats Overview

FlySweeper uses two checkpoint representations:
1. **PyTorch State File (`.pt`)**: Used during training, evaluation, and fine-tuning via PyTorch.
2. **WebGL JSON File (`.json`)**: Used for browser execution in Three.js and real-time 3D HUD rendering.

---

## 2. PyTorch State Format (`checkpoints/best_fly_model.pt`)

Saved using `torch.save()` as a standard Python dictionary.

### Tensor Anatomy

| Key | Type | Shape | Parameter Count | Description |
|---|---|---|---|---|
| `W_in` | `torch.FloatTensor` | `[32, num_board_cells]` | $32 	imes 36 = 1,152$ | Linear mapping from board cell inputs to 32 visual input neurons (LC4, LC11, LC15, LPLC2) |
| `W_out` | `torch.FloatTensor` | `[num_board_cells, 16]` | $36 	imes 16 = 576$ | Linear readout mapping from 16 descending motor neurons (DNp01, pIP1) to tile logits |
| `b_out` | `torch.FloatTensor` | `[num_board_cells]` | $36$ | Bias vector for tile selection logits |
| `num_board_cells` | `int` | Scalar | - | Total board cells (e.g. 36 for 6x6, 64 for 8x8) |
| `metadata` | `dict` | - | - | Optional performance metadata (best reward, win rate, generation) |

**Total Trainable Parameters**: 1,764 float32 values ($7,056$ bytes).

### Serialization Code

```python
import torch

checkpoint_data = {
    "W_in": brain.W_in.cpu(),
    "W_out": brain.W_out.cpu(),
    "b_out": brain.b_out.cpu(),
    "num_board_cells": brain.num_board_cells,
    "metadata": {
        "win_rate": 0.50,
        "safe_reveals": 96.4,
        "algorithm": "CEM"
    }
}
torch.save(checkpoint_data, "checkpoints/best_fly_model.pt")
```

### Loading & Deserialization Code

```python
import torch

checkpoint = torch.load("checkpoints/best_fly_model.pt", map_location="cpu", weights_only=False)
W_in = checkpoint["W_in"]       # Shape: [32, 36]
W_out = checkpoint["W_out"]     # Shape: [36, 16]
b_out = checkpoint["b_out"]     # Shape: [36]
board_cells = checkpoint.get("num_board_cells", 36)
```

---

## 3. Flat Parameter Vector Mapping

The Cross-Entropy Method (CEM) optimizes a single 1D flat vector $	heta \in \mathbb{R}^{1764}$.

### Flattening Layout
The 1D vector is concatenated in this exact order:
1. `W_in.flatten()`: elements $0 \dots 1151$ ($32 	imes 36$)
2. `W_out.flatten()`: elements $1152 \dots 1727$ ($36 	imes 16$)
3. `b_out.flatten()`: elements $1728 \dots 1763$ ($36$)

### Extraction & Injection Functions
In `fly_brain.py`:
* `brain.get_parameters_flat()`: returns `np.ndarray` of shape `(1764,)`.
* `brain.set_parameters_flat(flat_vector)`: unpacks slices back into `W_in`, `W_out`, and `b_out`.

---

## 4. WebGL JSON Format (`fly_model_web.json`)

To run client-side in the browser without a Python runtime, `export_weights.py` combines the biological circuit graph with the trained readout parameters.

### JSON Structure

```json
{
  "version": "malecns-v1.0",
  "num_board_cells": 36,
  "nodes": [
    {
      "id": 0,
      "name": "LC4_R_01",
      "type": "sensory",
      "x": -21.4,
      "y": 14.8,
      "z": 5.2,
      "neurotransmitter": "ACH",
      "sign": 1
    }
  ],
  "edges": [
    {
      "source": 0,
      "target": 35,
      "weight": 0.42,
      "sign": 1
    }
  ],
  "W_in": [[0.012, -0.045, ...], ...],
  "W_out": [[0.104, -0.032, ...], ...],
  "b_out": [-0.015, 0.042, ...]
}
```

### Validation Check

```bash
python -c "import json; d = json.load(open('fly_model_web.json')); assert len(d['W_in']) == 32; assert len(d['W_out']) == 36; assert len(d['b_out']) == 36; print('WebGL checkpoint valid!')"
```
