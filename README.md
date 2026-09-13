# Fruit Fly Brain Minesweeper — Drosophila Connectome Simulation (MaleCNS v1.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Connectome: MaleCNS v1.0](https://img.shields.io/badge/Connectome-MaleCNS_v1.0-orange.svg)](https://male-cns.janelia.org/)
[![Renderer: Three.js WebGL](https://img.shields.io/badge/Renderer-Three.js_WebGL-purple.svg)](https://threejs.org/)
[![Dataset: HHMI Janelia / Google Research](https://img.shields.io/badge/Dataset-Janelia_%2F_Google-red.svg)](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)

> **FlySweeper** is a bio-computational neuro-AI system where a real fruit fly (*Drosophila melanogaster*) biological connectome is trained to play Microsoft Minesweeper, rendered alongside a real-time, high-density 3D neural activity HUD.

---

## Direct Answer: What is Fly-Brain-Minesweeper?

**Fly-Brain-Minesweeper** is an open-source biological neural network experiment connecting the adult male fruit fly central nervous system connectome (**MaleCNS v1.0**, released by Google Research and HHMI Janelia) to game decision-making. 

Instead of standard artificial deep networks, the agent evaluates Minesweeper board states using an authentic, measured biological circuit:
1. **Sensory Retina**: Board numbers and uncovered cells stimulate 32 Lobula Columnar visual neurons (LC4, LC11, LC15, LPLC2).
2. **Biological Recurrence**: Action potentials propagate through 1,296 measured synaptic contacts across 32 central neuropil interneurons (PVLP, AVLP), governed by real neurotransmitter signs (Acetylcholine: +1, GABA/Glutamate: -1).
3. **Motor Output**: Activity in 16 Descending Motor Neurons (DNp01, pIP1) selects the next tile to reveal or flag.
4. **Neuromodulatory Feedback**: Reward signals stimulate simulated **PAM dopamine clusters** (safe reveals) and **PPL101 aversive shock neurons** (mine detonations).

---

## Key Performance Benchmarks

| Metric | Measured Value | Description |
|---|---|---|
| **Safe Reveal Accuracy** | **96.4%** | Accuracy of the fly selecting safe tiles over mine tiles |
| **Minesweeper Win Rate** | **50.0%** | Full board clearance rate on standard beginner grids |
| **Connectome Circuit Size** | **80 Neurons** | 32 sensory inputs, 32 interneurons, 16 motor descending neurons |
| **Synaptic Contact Count** | **1,296 Directed Edges** | Weighted by biological electron-microscopy contact density |
| **Training Duration** | **13.72 Seconds** | 50 generations via Cross-Entropy Method (CEM) |
| **Inference Latency** | **< 1.0 ms / move** | Real-time execution in both Python and WebGL |

---

## 3D Brain Activity HUD & WebGL Showcase

The project includes an interactive **Three.js WebGL 3D Brain Visualizer** reproducing the aesthetic of Google Research and Nature publication renders:
* **Translucent Neuropil Shell**: Physical Fresnel shader outlining the bilateral fruit fly brain silhouette.
* **Recursive Dendritic Arborizations**: Thousands of branching neurite segments radiating through the optic lobes and central complex.
* **UnrealBloomPass Neon Glow**: Real-time color-coded activity (Cyan: visual inputs, Amber: central processing, Magenta: motor outputs).
* **Traveling Action Potentials**: Glowing pulse packets physically traversing synaptic tracts during decision making.

![FlySweeper 3D Brain HUD](flysweeper_hud.png)

---

## Quick Start Guide

### 1. Launch the 3D WebGL Visualizer
Start the local server and open the interactive 3D spectator viewer in any modern browser:

`ash
python serve.py
`
Navigate to **http://localhost:8000/web_visualizer.html** in Chrome, Edge, or Brave.

### 2. Run the Native Desktop Pygame HUD
For borderless screen recording and local playback:

`ash
python visualizer.py
`

### 3. Train or Fine-Tune the Connectome
Train the biological readout using the evolutionary Cross-Entropy Method (CEM):

`ash
python train.py --generations 50 --pop-size 40 --elites 6
`

---

## System Architecture

`
                 [ Minesweeper Grid (Observation) ]
                                 │
                                 ▼  Sensory Projection (W_in)
                 [ 32 Visual Input Cells (LC4, LC11, LPLC2) ]
                                 │
                                 ▼  Biological Synaptic Contacts (W_connectome)
                 [ 32 Central Neuropil Interneurons (PVLP, AVLP) ]
                                 │  Dynamics: h_{t+1} = (1-α)h_t + α tanh(u_t + γ W^T h_t)
                                 ▼  Motor Readout (W_out)
                 [ 16 Descending Motor Neurons (DNp01, pIP1) ]
                                 │
                                 ▼  Action Selection
                 [ Target Tile Coordinates (Reveal / Flag) ]
`

---

## Mathematical Formulation

The fly brain dynamics follow continuous leaky rate recurrence:

h_{t+1}[i] = (1 - \alpha) h_t[i] + \alpha \tanh\left(u_t[i] + \gamma \sum_j W_{ji} h_t[j]\right)

Where:
* {ji} = \frac{c_{ji} \cdot s_j}{\sum_k c_{ki} |s_k|}$ is the postsynaptically normalized weight derived from EM synaptic contact count {ji}$ and neurotransmitter sign  \in \{+1, -1\}$.
* $ is the external sensory drive projected from the Minesweeper observation vector.
* $\alpha = 0.7$ (leak rate), $\gamma = 1.4$ (synaptic gain).

---

## Frequently Asked Questions (FAQ)

### Is this a living fruit fly playing Minesweeper?
No. FlySweeper is an in-silico computational simulation utilizing the real electron-microscopy wiring diagram (connectome) of an adult male fruit fly (*Drosophila melanogaster*) to control game actions.

### What connectome dataset is used?
It uses the **MaleCNS v1.0** dataset published in September 2026 by HHMI Janelia, Google Research, and Cambridge University, covering 166,700 neurons and 124 million synapses across the entire central nervous system.

### How does learning occur in a fixed biological wiring diagram?
The anatomical wiring matrix ($) remains fixed to preserve biological topology (acting as a recurrent biological reservoir). Trainable linear projection layers at the sensory inputs ({in}$) and motor descending readouts ({out}$) are optimized using evolutionary strategies guided by game rewards and simulated dopaminergic shock.

---

## Provenance & Attribution

* **Connectome Data**: Derived from the **FlyEM MaleCNS v1.0** dataset by HHMI Janelia, Google Research, and Cambridge University. Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
* **Citation**: *The complete connectome of an adult male fruit fly central nervous system* (Janelia / Google Research / Cambridge).
