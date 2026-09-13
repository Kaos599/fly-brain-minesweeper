import random
import numpy as np

class MinesweeperEnv:
    """
    Lightweight, fast Gymnasium-style Minesweeper environment.
    State conventions:
      -1 : Unrevealed tile
       0-8: Revealed tile with number of neighboring mines
      -2 : Flagged tile (optional)
    """
    def __init__(self, rows=6, cols=6, num_mines=4, seed=None):
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.total_cells = rows * cols
        self.safe_cells_total = self.total_cells - num_mines
        
        self.rng = random.Random(seed)
        self.board = np.zeros((rows, cols), dtype=int)  # 1 if mine, 0 otherwise
        self.revealed = np.zeros((rows, cols), dtype=bool)
        self.flagged = np.zeros((rows, cols), dtype=bool)
        self.counts = np.zeros((rows, cols), dtype=int)
        
        self.game_over = False
        self.won = False
        self.first_click = True
        self.safe_reveals = 0
        self.step_count = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.board.fill(0)
        self.revealed.fill(False)
        self.flagged.fill(False)
        self.counts.fill(0)
        self.game_over = False
        self.won = False
        self.first_click = True
        self.safe_reveals = 0
        self.step_count = 0
        return self.get_observation()

    def _place_mines(self, first_r, first_c):
        """Place mines randomly avoiding the first clicked cell and its immediate neighbors."""
        forbidden = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = first_r + dr, first_c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    forbidden.add((nr, nc))
        
        candidates = [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in forbidden]
        # If candidate pool is too small, just exclude first_r, first_c
        if len(candidates) < self.num_mines:
            candidates = [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) != (first_r, first_c)]
            
        chosen = self.rng.sample(candidates, self.num_mines)
        for r, c in chosen:
            self.board[r, c] = 1
            
        # Compute adjacent counts
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r, c] == 1:
                    continue
                cnt = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            cnt += self.board[nr, nc]
                self.counts[r, c] = cnt

    def _cascade_reveal(self, r, c):
        """Reveal empty cell and cascade adjacent cells."""
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if self.revealed[cr, cc]:
                continue
            self.revealed[cr, cc] = True
            self.safe_reveals += 1
            
            if self.counts[cr, cc] == 0:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            if not self.revealed[nr, nc] and not self.flagged[nr, nc]:
                                stack.append((nr, nc))

    def step(self, action):
        """
        action: integer in [0, total_cells - 1] representing cell index (r = action // cols, c = action % cols)
        Returns: observation, reward, done, info
        """
        self.step_count += 1
        r = action // self.cols
        c = action % self.cols

        info = {
            "action": (r, c),
            "hit_mine": False,
            "already_revealed": False,
            "won": False,
            "safe_reveals": self.safe_reveals
        }

        if self.game_over:
            return self.get_observation(), 0.0, True, info

        # Repeated click on already revealed cell
        if self.revealed[r, c]:
            info["already_revealed"] = True
            return self.get_observation(), -0.1, False, info

        # First click safety
        if self.first_click:
            self._place_mines(r, c)
            self.first_click = False

        # Hit mine!
        if self.board[r, c] == 1:
            self.game_over = True
            self.revealed[r, c] = True
            info["hit_mine"] = True
            reward = -5.0
            return self.get_observation(), reward, True, info

        # Safe reveal
        prev_safe = self.safe_reveals
        if self.counts[r, c] == 0:
            self._cascade_reveal(r, c)
        else:
            self.revealed[r, c] = True
            self.safe_reveals += 1

        new_reveals = self.safe_reveals - prev_safe
        reward = 0.5 * new_reveals

        info["safe_reveals"] = self.safe_reveals
        # Check win condition
        if self.safe_reveals >= self.safe_cells_total:
            self.game_over = True
            self.won = True
            info["won"] = True
            reward += 10.0
            return self.get_observation(), reward, True, info

        return self.get_observation(), reward, False, info

    def get_observation(self):
        """
        Returns observation grid:
        -1 for unrevealed
        0-8 for revealed counts
        """
        obs = np.full((self.rows, self.cols), -1.0, dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.revealed[r, c]:
                    obs[r, c] = float(self.counts[r, c])
        return obs

    def get_flat_observation(self):
        """Normalized 1D observation for neural networks (values in [-1.0, 1.0])."""
        obs = self.get_observation().flatten()
        # Normalize: -1 -> -1.0, 0 -> 0.0, 1..8 -> count / 8.0
        norm_obs = np.zeros_like(obs)
        for i, val in enumerate(obs):
            if val < 0:
                norm_obs[i] = -1.0
            else:
                norm_obs[i] = val / 8.0
        return norm_obs

    def get_valid_action_mask(self):
        """Boolean mask: True if cell is unrevealed, False if already revealed."""
        return (~self.revealed).flatten()
