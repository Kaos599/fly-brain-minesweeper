# Checkpoint Training & Improvement Guide

This guide details how to fine-tune existing FlySweeper weights, optimize hyperparameters, apply curriculum learning, and push agent win rates.

---

## 1. Why Evolutionary Strategies (CEM) for Biological Reservoirs?

Minesweeper involves discrete tile clicks, non-differentiable game outcomes (win, mine explosion, safe reveal), and a continuous recurrent biological dynamics loop:

$$
h_{t+1}[i] = (1 - lpha) h_t[i] + lpha 	anh\left(u_t[i] + \gamma \sum_j W_{ji} h_t[j]ight)
$$

Standard backpropagation through time (BPTT) across this recurrent differential reservoir suffers from vanishing and exploding gradients over long episode trajectories.

The Cross-Entropy Method (CEM) treats the biological reservoir as a black-box dynamical system and samples readout parameters $	heta \sim \mathcal{N}(\mu, \Sigma)$. It selects top-performing candidates ($K$ elites) and updates the parameter distribution towards the elite mean.

---

## 2. Continuing Training from an Existing Checkpoint

To improve a model without starting from scratch:

```bash
python train.py \
    --resume checkpoints/best_fly_model.pt \
    --generations 30 \
    --pop-size 30 \
    --elites 5 \
    --init-std 0.15 \
    --eval-episodes 5 \
    --model-path checkpoints/best_fly_model.pt
```

### Key Parameter Considerations for Fine-Tuning:
1. **`--init-std` (Exploration Noise)**:
   * Scratch training uses $\sigma = 0.40$ to explore the parameter space widely.
   * Fine-tuning should use $\sigma = 0.15$ or $\sigma = 0.10$ to make localized refinements around the proven elite weights.
2. **`--elites`**:
   * Set to approximately $15\%$ to $20\%$ of `--pop-size` (e.g. 5 elites for pop size 30).
3. **Elitism Invariant**:
   * Candidate 0 in every generation is set directly to the current distribution mean $\mu$. This guarantees that a generation can never regress below the prior best mean.

---

## 3. Hyperparameter Reference Table

| Parameter | Scratch Value | Fine-Tune Value | Description |
|---|---|---|---|
| `--generations` | 50 | 25 to 40 | Total optimization epochs |
| `--pop-size` | 40 | 30 | Candidate networks evaluated per epoch |
| `--elites` | 6 | 5 | Top candidates selected to compute the new mean |
| `--init-std` | 0.40 | 0.15 | Initial standard deviation for perturbation noise |
| `--eval-episodes` | 4 | 6 to 8 | Episodes per candidate (higher = less evaluation noise) |
| `--rows` | 6 | 6 (or 8) | Grid height |
| `--cols` | 6 | 6 (or 8) | Grid width |
| `--mines` | 4 | 4 (or 10) | Total hidden mines |

---

## 4. Reward Function Engineering

The environment reward in `minesweeper_env.py` is shaped to guide biological dynamics:

* **Safe Reveal**: $+1.0$ per safe tile opened.
* **First Action Safety**: Mines are generated only after the first click, ensuring the agent is never penalized for an unlucky first guess.
* **Game Won Bonus**: $+20.0$ for clearing all non-mine tiles.
* **Mine Detonation (Dopaminergic Punishment)**: $-10.0$ penalty immediately ending the episode.
* **Illegal Action Penalty**: Selecting an already revealed tile yields $-2.0$ (masked during normal inference).

### Suggested Reward Experiments to Boost Win Rate:
* **Chain-Reaction Bonus**: Provide an extra $+0.5$ bonus when a click triggers a cascade of zero-reveals, encouraging the fly to target interior cluster openings.
* **Perimeter Caution**: Scale safe reveal rewards higher when adjacent unrevealed neighbor counts are low.

---

## 5. Curriculum Learning Protocol

To transition the fly from beginner grids to standard intermediate boards:

```text
Stage 1: Beginner Mini      -> 6x6 grid, 2 mines   (Target: >98% safe reveal, >80% win rate)
Stage 2: Standard Beginner  -> 6x6 grid, 4 mines   (Current: 96.4% safe reveal, 50% win rate)
Stage 3: Intermediate Mini  -> 8x8 grid, 6 mines   (Target: >95% safe reveal, >60% win rate)
Stage 4: Standard Intermediate -> 8x8 grid, 10 mines (Target: >90% safe reveal, >40% win rate)
```

### Procedure:
1. Train Stage 1 for 30 generations.
2. Resume into Stage 2 using `--resume` for 30 generations.
3. For Stage 3, instantiate an 8x8 brain ($W_{in} \in \mathbb{R}^{32 	imes 64}$, $W_{out} \in \mathbb{R}^{64 	imes 16}$) and train using Stage 2 parameters as an initialization prior.

---

## 6. Verification and Export Checklist

After completing any training run:

1. **Quantitatively Benchmark**:
   ```bash
   python train.py --eval-only --model-path checkpoints/best_fly_model.pt --eval-episodes 50
   ```
2. **Export to WebGL JSON**:
   ```bash
   python export_weights.py --model checkpoints/best_fly_model.pt --output fly_model_web.json
   ```
3. **Verify Web Model Size**:
   Confirm that `fly_model_web.json` is between 80 KB and 150 KB.
4. **Visual Inspection**:
   Run `python serve.py` and inspect action choices and neural pulses in `http://localhost:8000/web_visualizer.html`.
