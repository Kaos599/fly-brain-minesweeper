import os
import time
import argparse
import numpy as np
import torch
from minesweeper_env import MinesweeperEnv
from fly_brain import FlyBrain

SCRATCH_DIR = "checkpoints"
DEFAULT_MODEL_PATH = os.path.join(SCRATCH_DIR, "best_fly_model.pt")

def evaluate_brain(brain, env, episodes=5, deterministic=True):
    """Evaluate a single fly brain across multiple episodes."""
    total_reward = 0.0
    wins = 0
    safe_reveals_total = 0

    for ep in range(episodes):
        obs = env.reset(seed=1000 + ep)
        brain.reset_state()
        ep_reward = 0.0
        done = False
        steps = 0
        max_steps = env.total_cells * 2

        while not done and steps < max_steps:
            steps += 1
            valid_mask = env.get_valid_action_mask()
            flat_obs = env.get_flat_observation()
            action = brain.pick_action(flat_obs, valid_mask=valid_mask, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            ep_reward += reward

        if env.won:
            wins += 1
        safe_reveals_total += env.safe_reveals
        total_reward += ep_reward

    mean_reward = total_reward / episodes
    win_rate = wins / episodes
    safe_pct = (safe_reveals_total / (env.safe_cells_total * episodes)) * 100.0
    return mean_reward, win_rate, safe_pct

def run_eval_only(args):
    print("=" * 70)
    print("  FLYSWEEPER: Model Evaluation Mode")
    print("=" * 70)
    env = MinesweeperEnv(rows=args.rows, cols=args.cols, num_mines=args.mines)
    brain = FlyBrain(num_board_cells=args.rows * args.cols)
    
    target_model = args.resume or args.model_path
    if not os.path.exists(target_model):
        raise FileNotFoundError(f"Checkpoint not found at: {target_model}")

    print(f"Loading weights from: {target_model}")
    brain.load_weights(target_model)
    print(brain.summary())
    print(f"Grid: {args.rows}x{args.cols} ({args.mines} mines) | Test Episodes: {args.eval_episodes}")
    print("-" * 70)

    mean_rew, win_rate, safe_pct = evaluate_brain(brain, env, episodes=args.eval_episodes, deterministic=True)
    print(f"Results across {args.eval_episodes} episodes:")
    print(f"  * Safe Reveal Accuracy: {safe_pct:5.2f}%")
    print(f"  * Game Win Rate:        {win_rate * 100:5.2f}%")
    print(f"  * Mean Reward:          {mean_rew:6.2f}")
    print("=" * 70)

def train_cem(args):
    if args.eval_only:
        run_eval_only(args)
        return

    print("=" * 70)
    print("  FLYSWEEPER: MaleCNS v1.0 Drosophila Connectome Training")
    print("=" * 70)

    env = MinesweeperEnv(rows=args.rows, cols=args.cols, num_mines=args.mines)
    brain = FlyBrain(num_board_cells=args.rows * args.cols)
    print(brain.summary())
    print(f"Grid: {args.rows}x{args.cols} with {args.mines} mines | Total cells: {env.total_cells}")
    print(f"Algorithm: Cross-Entropy Method (CEM) | Pop Size: {args.pop_size} | Elites: {args.elites}")
    print("-" * 70)

    # Initial parameter distribution
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        print(f"Resuming training from checkpoint: {args.resume}")
        brain.load_weights(args.resume)
        init_std_val = args.init_std if args.init_std is not None else 0.15
    else:
        init_std_val = args.init_std if args.init_std is not None else 0.40

    init_params = brain.get_parameters_flat()
    param_dim = len(init_params)
    mean = np.copy(init_params)
    std = np.ones(param_dim, dtype=np.float32) * init_std_val

    # Initial baseline evaluation
    baseline_rew, baseline_win, baseline_safe = evaluate_brain(brain, env, episodes=args.eval_episodes, deterministic=True)
    print(f"Starting Baseline -> Rew: {baseline_rew:.2f} | Win Rate: {baseline_win * 100:.1f}% | Safe: {baseline_safe:.1f}% | Init Std: {init_std_val:.3f}")
    print("-" * 70)

    best_overall_reward = baseline_rew
    best_overall_params = np.copy(mean)
    best_win_rate = baseline_win

    start_time = time.time()

    for gen in range(1, args.generations + 1):
        # Sample population
        population = np.random.randn(args.pop_size, param_dim).astype(np.float32) * std + mean
        # Always include current mean as candidate 0 (elitism)
        population[0] = mean

        scores = []
        win_rates = []
        safe_pcts = []

        for candidate in population:
            brain.set_parameters_flat(candidate)
            m_rew, w_rate, s_pct = evaluate_brain(brain, env, episodes=args.eval_episodes, deterministic=True)
            scores.append(m_rew)
            win_rates.append(w_rate)
            safe_pcts.append(s_pct)

        scores = np.array(scores)
        elite_indices = np.argsort(scores)[::-1][:args.elites]
        elites = population[elite_indices]

        # Update distribution towards elite population
        new_mean = np.mean(elites, axis=0)
        new_std = np.std(elites, axis=0) + 0.05  # minimum exploration noise floor

        mean = 0.8 * mean + 0.2 * new_mean
        std = 0.8 * std + 0.2 * new_std

        gen_best_score = scores[elite_indices[0]]
        gen_best_win = win_rates[elite_indices[0]]
        gen_best_safe = safe_pcts[elite_indices[0]]

        if gen_best_score > best_overall_reward:
            best_overall_reward = gen_best_score
            best_overall_params = np.copy(elites[0])
            best_win_rate = gen_best_win

            # Save checkpoint
            brain.set_parameters_flat(best_overall_params)
            os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
            brain.save_weights(args.model_path)
            saved_tag = " [SAVED CHECKPOINT]"
        else:
            saved_tag = ""

        elapsed = time.time() - start_time
        print(f"Gen {gen:03d}/{args.generations:03d} | "
              f"Best Rew: {gen_best_score:6.2f} | "
              f"Win Rate: {gen_best_win * 100:4.1f}% | "
              f"Safe Reveals: {gen_best_safe:5.1f}% | "
              f"Elapsed: {elapsed:5.1f}s{saved_tag}")

    print("-" * 70)
    print(f"Training Complete in {time.time() - start_time:.2f}s!")
    print(f"Best Reward: {best_overall_reward:.2f} | Best Win Rate: {best_win_rate * 100:.1f}%")
    print(f"Model saved to: {args.model_path}")

    # Final verification run
    brain.set_parameters_flat(best_overall_params)
    final_rew, final_win, final_safe = evaluate_brain(brain, env, episodes=10, deterministic=True)
    print(f"Final 10-Episode Test -> Win Rate: {final_win * 100:.1f}%, Safe: {final_safe:.1f}%, Mean Reward: {final_rew:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fly Connectome Minesweeper Agent")
    parser.add_argument("--generations", type=int, default=30, help="Number of training generations")
    parser.add_argument("--pop-size", type=int, default=20, help="Population size per generation")
    parser.add_argument("--elites", type=int, default=4, help="Number of elite candidates to select")
    parser.add_argument("--eval-episodes", type=int, default=4, help="Episodes per candidate evaluation")
    parser.add_argument("--rows", type=int, default=6, help="Board rows")
    parser.add_argument("--cols", type=int, default=6, help="Board cols")
    parser.add_argument("--mines", type=int, default=4, help="Number of mines")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to save best weights")
    parser.add_argument("--resume", type=str, default=None, help="Path to existing checkpoint to resume training from")
    parser.add_argument("--init-std", type=float, default=None, help="Initial noise std (default: 0.40 for scratch, 0.15 for resume)")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only without training")
    args = parser.parse_args()

    train_cem(args)
