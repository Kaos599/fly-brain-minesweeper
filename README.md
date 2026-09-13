# FlySweeper (fly-brain-minesweeper)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Connectome: MaleCNS v1.0](https://img.shields.io/badge/Connectome-MaleCNS_v1.0-orange.svg)](https://male-cns.janelia.org/)
[![Renderer: Three.js WebGL](https://img.shields.io/badge/Renderer-Three.js_WebGL-purple.svg)](https://threejs.org/)

> **A real fruit fly (*Drosophila melanogaster*) connectome playing Microsoft Minesweeper with a real-time, high-density 3D neural activity HUD.**

Powered by the **MaleCNS v1.0** central nervous system connectome published by Google Research and HHMI Janelia.

---

## Overview

FlySweeper connects biological connectomics to classic game puzzle solving. Rather than using an artificial black-box deep network, the agent’s sensory perception, recurrence, and motor commands are mediated through an authentic biological connectome subgraph extracted from the adult male fruit fly:

* **80 Biological Neurons**: Includes classified Lobula Columnar visual neurons (LC4, LC11, LC15, LPLC2), central neuropil interneurons (PVLP, AVLP), and motor Descending Neurons (DNp01, pIP1).
* **1,296 Measured Synaptic Contacts**: Preserves directional edge contact counts and true neurotransmitter signs (Acetylcholine: +1, GABA/Glutamate: -1).
* **High-Density 3D Brain Visualizer**: WebGL/Three.js interface with volumetric neuropil silhouette shell, recursive dendritic arborizations, traveling action potentials, and cinematic neon bloom.
* **Rapid Evolutionary Learning**: Trained via Cross-Entropy Method (CEM) on the biological reservoir to achieve **96.4% safe reveals** and a **50% win rate** on beginner boards in seconds.

---

## Quick Start

### 1. Launch the Cinematic 3D WebGL Visualizer
Start the local server and open the interactive 3D spectator viewer in your browser:

`ash
python serve.py
`
Open **http://localhost:8000/web_visualizer.html**

* Drag mouse to orbit in 3D, scroll to zoom.
* Toggle between auto-play AI, step-by-step evaluation, or manual play.

### 2. Desktop Pygame Visualizer
For native borderless window capture:

`ash
python visualizer.py
`

### 3. Train or Fine-Tune
Run evolutionary reinforcement learning on the connectome:

`ash
python train.py --generations 50 --pop-size 40 --elites 6
`

---

## Architecture

`
[ Minesweeper Grid ]
       │
       ▼  Sensory Projection (W_in)
[ 32 Visual Input Neurons (LC4, LC11, LPLC2) ]
       │
       ▼  Recurrent Biological Synaptic Dynamics (W_connectome)
[ 32 Central Neuropil Interneurons (PVLP, AVLP) ]
       │
       ▼  Motor Readout (W_out)
[ 16 Descending Motor Neurons (DNp01, pIP1) ]
       │
       ▼  Action Selection
[ Reveal / Flag Target Tile ]
`

---

## Provenance & Attribution

* **Connectome Data**: Derived from the **FlyEM MaleCNS v1.0** dataset by HHMI Janelia, Google Research, and Cambridge University. Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Research Paper**: *The complete connectome of an adult male fruit fly central nervous system* (Cell / Janelia / Google Research).
