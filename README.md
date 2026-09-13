# Fly-Brain-Minesweeper: Drosophila Connectome Simulation (MaleCNS v1.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Connectome: MaleCNS v1.0](https://img.shields.io/badge/Connectome-MaleCNS_v1.0-orange.svg)](https://male-cns.janelia.org/)
[![Renderer: Three.js WebGL](https://img.shields.io/badge/Renderer-Three.js_WebGL-purple.svg)](https://threejs.org/)

FlySweeper is a bio-computational neuro-AI system where a biological fruit fly (*Drosophila melanogaster*) connectome controls Microsoft Minesweeper, paired with a real-time 3D neural activity HUD.

---

## Overview

Fly-Brain-Minesweeper executes game decision-making through a measured biological neural graph rather than an artificial deep network. The model uses an extracted subcircuit from the adult male fruit fly central nervous system (MaleCNS v1.0, published by Google Research and HHMI Janelia):

* **80 Biological Neurons**: 32 Lobula Columnar visual inputs (LC4, LC11, LC15, LPLC2), 32 central neuropil interneurons (PVLP, AVLP), and 16 descending motor neurons (DNp01, pIP1).
* **1,296 Directed Synaptic Contacts**: Preserves physical contact densities and biological neurotransmitter signs (Acetylcholine: +1, GABA/Glutamate: -1).
* **High-Density 3D Brain HUD**: Real-time WebGL and Three.js visualization with a translucent neuropil silhouette, recursive dendritic arborizations, and action potential pulse propagation.
* **Evolutionary Readout Optimization**: Trained via Cross-Entropy Method (CEM) on the biological reservoir, reaching 96.4% safe reveals and a 50.0% win rate on beginner boards.

---

## Benchmarks

| Metric | Value | Description |
|---|---|---|
| Safe Reveal Accuracy | 96.4% | Frequency of selecting safe tiles over mine tiles |
| Game Win Rate | 50.0% | Complete board clearance rate on beginner grids (6x6, 4 mines) |
| Active Neurons | 80 Cells | 32 sensory inputs, 32 interneurons, 16 motor descending outputs |
| Synaptic Edges | 1,296 Directed Edges | Measured electron-microscopy contacts with signed weights |
| Training Duration | 13.72 Seconds | 50 generations with population 40 using CEM |
| Inference Latency | < 1.0 ms / step | Forward step time on modern CPU and GPU runtimes |

---

## System Architecture

```text
+-------------------------------------------------------------+
|                  Minesweeper Grid State                     |
|            (-1: unrevealed, 0-8: revealed counts)           |
+-------------------------------------------------------------+
                               |
                               | Sensory Projection (W_in)
                               v
+-------------------------------------------------------------+
|         32 Visual Input Neurons (LC4, LC11, LPLC2)          |
+-------------------------------------------------------------+
                               |
                               | Synaptic Contacts (W_connectome)
                               v
+-------------------------------------------------------------+
|         32 Central Neuropil Interneurons (PVLP, AVLP)       |
|    Recurrent Dynamics: h_{t+1} = (1-a)h_t + a*tanh(u + g*W*h)|
+-------------------------------------------------------------+
                               |
                               | Motor Readout (W_out)
                               v
+-------------------------------------------------------------+
|          16 Descending Motor Neurons (DNp01, pIP1)          |
+-------------------------------------------------------------+
                               |
                               | Action Selection
                               v
+-------------------------------------------------------------+
|               Selected Tile: Reveal or Flag                 |
+-------------------------------------------------------------+
```

---

## Mathematical Formulation

The recurrent network updates via continuous leaky rate dynamics:

$$
h_{t+1}[i] = (1 - \alpha) h_t[i] + \alpha \tanh\left(u_t[i] + \gamma \sum_j W_{ji} h_t[j]\right)
$$

Where:
* $W_{ji} = \frac{c_{ji} s_j}{\sum_k c_{ki} |s_k|}$ is the normalized synaptic weight derived from physical contact count $c_{ji}$ and neurotransmitter sign $s_j \in \{+1, -1\}$.
* $u_t[i]$ is the sensory stimulus projected from board states onto visual input cells.
* $\alpha = 0.7$ is the leak coefficient, and $\gamma = 1.4$ is the recurrent synaptic gain.

---

## Connectome Selection: Why a Subcircuit Instead of the Full Brain?

The complete MaleCNS v1.0 connectome contains 166,700 neurons and over 124 million synapses. FlySweeper intentionally isolates an 80-neuron functional visual-motor circuit for four concrete technical reasons:

1. **Computational Tractability and Real-Time Latency**:
   Simulating 166,700 non-linear continuous differential equations across 124 million connections drops simulation speed from 100,000 steps per second down to approximately 1 step per second on consumer hardware. Training with evolutionary algorithms (CEM) across thousands of board rollouts finishes in 13.7 seconds with the subcircuit, but would require weeks of compute on the full graph.

2. **Biological Functional Modularity**:
   An adult fruit fly does not recruit its entire nervous system (such as olfactory glomeruli, gustatory pathways, circadian clocks, or mating circuits) to perform visual avoidance and motor escape. The Lobula Columnar (LC/LPLC) to central neuropil (PVLP/AVLP) to descending motor neuron (DN) axis is the exact pathway Drosophila evolved for spatial visual threat detection and directional steering.

3. **Signal-to-Noise Ratio in Reservoir Dynamics**:
   Feeding a 36-dimensional sensory observation (a 6x6 grid) into a 166,000-dimensional recurrent dynamical reservoir induces extreme signal dilution. Without extensive task-specific synaptic pruning, the sensory input vanishes into biological background noise. The 80-neuron circuit concentrates stimulus propagation through high-affinity synaptic pathways.

4. **WebGL Client-Side Rendering Limits**:
   Rendering 166,700 full dendritic morphology meshes and millions of dynamic action potential pulses crashes consumer browser WebGL contexts. The 80-neuron circuit with 1,296 synapses and recursive fractal arborizations renders smoothly at 60 FPS in Three.js with full post-processing bloom.

---

## Quick Start

### 1. WebGL 3D Brain Visualizer
Start the local server and open the browser interface:

```bash
python serve.py
```

Open http://localhost:8000/web_visualizer.html in Chrome, Edge, or Brave.
* Drag to rotate camera in 3D.
* Scroll to zoom.
* Toggle between auto-play AI, step mode, or manual board clicks.

### 2. Desktop Pygame Interface
Run the native desktop window:

```bash
python visualizer.py
```

### 3. Model Training
Train readout parameters using CEM:

```bash
python train.py --generations 50 --pop-size 40 --elites 6
```

---

## Frequently Asked Questions

### Is this a living organism?
No. FlySweeper is an in-silico simulation using the mapped electron-microscopy wiring diagram of an adult male fruit fly to compute game controls.

### What dataset is used?
The network uses the MaleCNS v1.0 connectome published by Google Research, HHMI Janelia, and Cambridge University, containing 166,700 neurons and 124 million synapses across the male nervous system.

### How does learning occur on a fixed connectome?
The biological connectome matrix $W$ remains frozen to maintain biological connectivity. Linear projection layers at the sensory inputs ($W_{in}$) and motor outputs ($W_{out}$) are trained using evolutionary strategies guided by game rewards and dopamine penalties.

---

## Attribution & Data Provenance

* **Connectome Data**: Derived from the FlyEM MaleCNS v1.0 dataset by HHMI Janelia, Google Research, and Cambridge University (CC BY 4.0).
* **Research Reference**: *The complete connectome of an adult male fruit fly central nervous system* (Janelia, Google Research, Cambridge).
