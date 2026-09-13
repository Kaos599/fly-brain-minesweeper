import os
import sys
import math
import time
import numpy as np
import pygame
import torch

from minesweeper_env import MinesweeperEnv
from fly_brain import FlyBrain

SCRATCH_DIR = r"checkpoints"
DEFAULT_MODEL_PATH = os.path.join(SCRATCH_DIR, "best_fly_model.pt")

# Color Palette
COLOR_BG = (24, 26, 32)
COLOR_PANEL = (34, 38, 48)
COLOR_PANEL_BORDER = (55, 62, 78)
COLOR_TEXT = (230, 235, 245)
COLOR_TEXT_DIM = (140, 150, 170)
COLOR_ACCENT = (66, 153, 225)

# Minesweeper Tile Colors
TILE_UNREVEALED = (185, 192, 205)
TILE_HIGHLIGHT = (215, 222, 235)
TILE_SHADOW = (130, 138, 152)
TILE_REVEALED = (210, 215, 225)
TILE_MINE = (220, 50, 50)
TILE_FLAG = (240, 140, 30)

NUM_COLORS = {
    1: (25, 75, 220),    # Blue
    2: (25, 140, 45),    # Green
    3: (215, 40, 40),    # Red
    4: (15, 25, 130),    # Dark Blue
    5: (120, 20, 20),    # Dark Red
    6: (20, 140, 140),   # Cyan
    7: (10, 10, 10),     # Black
    8: (120, 120, 120),  # Grey
}

class FlyVisualizer:
    def __init__(self, rows=6, cols=6, num_mines=4, model_path=DEFAULT_MODEL_PATH, headless=False):
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.headless = headless

        # Initialize Environment and Brain
        self.env = MinesweeperEnv(rows=rows, cols=cols, num_mines=num_mines)
        self.brain = FlyBrain(num_board_cells=rows * cols)

        if os.path.exists(model_path):
            print(f"Loading trained weights from {model_path}")
            self.brain.load_weights(model_path)
        else:
            print("Warning: No trained model checkpoint found, running with initial weights.")

        # Window dimensions
        self.width = 1100
        self.height = 680
        
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        pygame.init()
        pygame.display.set_caption("FlySweeper — Drosophila Connectome Brain HUD & Minesweeper")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        self.font_sm = pygame.font.SysFont("Segoe UI, Arial", 14)
        self.font_md = pygame.font.SysFont("Segoe UI, Arial", 18, bold=True)
        self.font_lg = pygame.font.SysFont("Segoe UI, Arial", 24, bold=True)
        self.font_num = pygame.font.SysFont("Consolas, Courier", 22, bold=True)

        # 3D Brain Projection Variables
        self.angle_y = 0.0
        self.angle_x = 0.35
        self.zoom = 180.0
        self.brain_center = (800, 320)

        # Simulation state
        self.auto_play = True
        self.step_delay = 0.5  # seconds per move
        self.last_step_time = time.time()
        self.last_action = None
        self.wins = 0
        self.games = 0

        self.reset_game()

    def reset_game(self):
        self.env.reset()
        self.brain.reset_state()
        self.last_action = None
        self.games += 1

    def step_fly(self):
        if self.env.game_over:
            return

        valid_mask = self.env.get_valid_action_mask()
        if not np.any(valid_mask):
            return

        flat_obs = self.env.get_flat_observation()
        # Pick action deterministically
        action = self.brain.pick_action(flat_obs, valid_mask=valid_mask, deterministic=True)
        self.last_action = (action // self.cols, action % self.cols)
        obs, reward, done, info = self.env.step(action)

        if self.env.won:
            self.wins += 1

    def project_3d_point(self, x, y, z):
        """Projects 3D normalized coordinate [-1, 1] to 2D screen coordinate."""
        # Yaw rotation (around Y axis)
        cos_y, sin_y = math.cos(self.angle_y), math.sin(self.angle_y)
        x1 = x * cos_y + z * sin_y
        z1 = -x * sin_y + z * cos_y

        # Pitch rotation (around X axis)
        cos_x, sin_x = math.cos(self.angle_x), math.sin(self.angle_x)
        y2 = y * cos_x - z1 * sin_x
        z2 = y * sin_x + z1 * cos_x

        # Perspective projection
        distance = 3.2
        fov = distance / (distance + z2 + 1e-4)
        px = self.brain_center[0] + x1 * self.zoom * fov
        py = self.brain_center[1] + y2 * self.zoom * fov
        return int(px), int(py), z2

    def draw_brain_hud(self):
        # Background box for Brain HUD
        hud_rect = pygame.Rect(540, 20, 530, 630)
        pygame.draw.rect(self.screen, COLOR_PANEL, hud_rect, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, hud_rect, width=2, border_radius=12)

        # Header Title
        title = self.font_lg.render("MaleCNS v1.0 Biological Neural HUD", True, COLOR_TEXT)
        self.screen.blit(title, (565, 38))
        subtitle = self.font_sm.render("80 Neurons | 1,296 Synaptic Contacts | Janelia EM Soma Atlas", True, COLOR_TEXT_DIM)
        self.screen.blit(subtitle, (565, 68))

        # Rotate brain slightly
        self.angle_y += 0.012

        neural_act = self.brain.get_neural_activity()
        norm_positions = self.brain.norm_positions

        # Precompute projected points
        projected = []
        for i, pos in enumerate(norm_positions):
            px, py, depth = self.project_3d_point(pos[0], -pos[1], pos[2])
            projected.append((px, py, depth, i))

        # Sort by depth (painter's algorithm)
        projected.sort(key=lambda item: item[2])

        # 1. Draw Synaptic Edges (sample prominent edges for visual clarity)
        edges = self.brain.edges
        edge_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for idx, edge in enumerate(edges):
            if idx % 3 != 0:
                continue
            s_id = edge[0] if isinstance(edge, list) else edge.get("src", edge.get("source"))
            d_id = edge[1] if isinstance(edge, list) else edge.get("dst", edge.get("target"))
            if s_id in self.brain.node_id_to_idx and d_id in self.brain.node_id_to_idx:
                s_idx = self.brain.node_id_to_idx[s_id]
                d_idx = self.brain.node_id_to_idx[d_id]
                spx, spy, _ = self.project_3d_point(norm_positions[s_idx][0], -norm_positions[s_idx][1], norm_positions[s_idx][2])
                dpx, dpy, _ = self.project_3d_point(norm_positions[d_idx][0], -norm_positions[d_idx][1], norm_positions[d_idx][2])
                
                # Activity along edge
                act_sum = abs(neural_act[s_idx]) + abs(neural_act[d_idx])
                alpha = int(min(120, 25 + act_sum * 50))
                pygame.draw.line(edge_surface, (70, 110, 160, alpha), (spx, spy), (dpx, dpy), 1)

        self.screen.blit(edge_surface, (0, 0))

        # 2. Draw Neurons as glowing spheres
        for px, py, depth, i in projected:
            role = self.brain.nodes[i].get("role", "interneuron")
            h = neural_act[i]

            # Radius based on depth
            base_r = 5.0 + (depth * 1.5)
            r = max(3, int(base_r + abs(h) * 4))

            # Color calculation
            if role == "input":
                # Green/Yellow for sensory
                val = int(np.clip((h + 1) * 127, 40, 255))
                color = (val, 220, 80)
            elif role == "output":
                # Magenta/Gold for descending motor
                val = int(np.clip(abs(h) * 255, 60, 255))
                color = (255, 70, val)
            else:
                # Interneurons: Blue for inhibited, Red for excited
                if h > 0.05:
                    intensity = int(min(255, 90 + h * 165))
                    color = (intensity, 60, 40)
                elif h < -0.05:
                    intensity = int(min(255, 90 + abs(h) * 165))
                    color = (40, 110, intensity)
                else:
                    color = (90, 100, 120)

            # Glow
            if abs(h) > 0.25:
                pygame.draw.circle(self.screen, (*color[:3], 80), (px, py), r + 4, width=1)
            pygame.draw.circle(self.screen, color, (px, py), r)

        # Telemetry info at bottom of HUD
        mean_act = float(np.mean(np.abs(neural_act)))
        max_act = float(np.max(np.abs(neural_act)))
        stats_text1 = self.font_sm.render(f"Brain Mean Rate: {mean_act:.3f} | Max Peak: {max_act:.3f}", True, COLOR_TEXT)
        self.screen.blit(stats_text1, (565, 555))

        dopamine_status = "PPL101 Aversive Shock: ACTIVE" if self.env.game_over and not self.env.won else "PAM Dopamine: Baseline"
        dop_color = (240, 80, 80) if self.env.game_over and not self.env.won else (80, 220, 120)
        stats_text2 = self.font_sm.render(dopamine_status, True, dop_color)
        self.screen.blit(stats_text2, (565, 580))

        # Legend
        leg_input = self.font_sm.render("● Inputs (LC4/LPLC)", True, (120, 220, 80))
        leg_inter = self.font_sm.render("● Interneurons", True, (200, 80, 80))
        leg_output = self.font_sm.render("● Motor DNs", True, (255, 70, 180))
        self.screen.blit(leg_input, (565, 610))
        self.screen.blit(leg_inter, (740, 610))
        self.screen.blit(leg_output, (890, 610))

    def draw_minesweeper_board(self):
        # Board Panel
        board_panel = pygame.Rect(30, 20, 480, 630)
        pygame.draw.rect(self.screen, COLOR_PANEL, board_panel, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, board_panel, width=2, border_radius=12)

        # Title
        title = self.font_lg.render("Minesweeper Agent", True, COLOR_TEXT)
        self.screen.blit(title, (50, 38))

        status_str = f"Win Rate: {(self.wins / max(1, self.games - 1)) * 100:.1f}% ({self.wins}/{max(0, self.games - 1)})"
        status = self.font_sm.render(status_str, True, COLOR_ACCENT)
        self.screen.blit(status, (50, 68))

        # Header Bar (Smiley & Mine counter)
        header_rect = pygame.Rect(50, 100, 440, 55)
        pygame.draw.rect(self.screen, (20, 22, 28), header_rect, border_radius=8)

        mines_left = self.num_mines
        mine_text = self.font_num.render(f"MINES: {mines_left:02d}", True, (240, 60, 60))
        self.screen.blit(mine_text, (65, 115))

        # Status Header Indicator
        if self.env.won:
            face = "[ WIN ]"
            face_col = (80, 220, 120)
        elif self.env.game_over:
            face = "[ BOOM ]"
            face_col = (240, 70, 70)
        else:
            face = "[ PLAYING ]"
            face_col = (240, 220, 70)
        face_surf = self.font_md.render(face, True, face_col)
        self.screen.blit(face_surf, (220, 117))

        safe_text = self.font_num.render(f"SAFE: {self.env.safe_reveals:02d}", True, (80, 200, 240))
        self.screen.blit(safe_text, (380, 115))

        # Grid Calculation
        grid_start_x = 70
        grid_start_y = 180
        tile_size = 65

        for r in range(self.rows):
            for c in range(self.cols):
                rect = pygame.Rect(grid_start_x + c * tile_size, grid_start_y + r * tile_size, tile_size - 4, tile_size - 4)

                # Draw Tile
                if not self.env.revealed[r, c]:
                    # Unrevealed Tile with 3D bevel
                    pygame.draw.rect(self.screen, TILE_UNREVEALED, rect, border_radius=4)
                    pygame.draw.line(self.screen, TILE_HIGHLIGHT, rect.topleft, rect.topright, 3)
                    pygame.draw.line(self.screen, TILE_HIGHLIGHT, rect.topleft, rect.bottomleft, 3)
                    pygame.draw.line(self.screen, TILE_SHADOW, rect.bottomleft, rect.bottomright, 3)
                    pygame.draw.line(self.screen, TILE_SHADOW, rect.topright, rect.bottomright, 3)
                else:
                    # Revealed Tile
                    pygame.draw.rect(self.screen, TILE_REVEALED, rect, border_radius=4)
                    pygame.draw.rect(self.screen, (170, 175, 185), rect, width=1, border_radius=4)

                    if self.env.board[r, c] == 1:
                        # Mine
                        pygame.draw.circle(self.screen, TILE_MINE, rect.center, tile_size // 4)
                        m_txt = self.font_sm.render("*", True, (255, 255, 255))
                        self.screen.blit(m_txt, m_txt.get_rect(center=rect.center))
                    else:
                        cnt = self.env.counts[r, c]
                        if cnt > 0:
                            num_color = NUM_COLORS.get(cnt, (0, 0, 0))
                            txt_surf = self.font_num.render(str(cnt), True, num_color)
                            self.screen.blit(txt_surf, txt_surf.get_rect(center=rect.center))

                # Highlight last action
                if self.last_action == (r, c):
                    pygame.draw.rect(self.screen, (255, 215, 0), rect, width=3, border_radius=4)

        # Controls instructions
        ctrl1 = self.font_sm.render("[SPACE] Pause/Play | [R] Restart Game | [N] Single Step", True, COLOR_TEXT_DIM)
        ctrl2 = self.font_sm.render("[UP/DOWN] Adjust Speed | [M] Manual Click", True, COLOR_TEXT_DIM)
        self.screen.blit(ctrl1, (50, 595))
        self.screen.blit(ctrl2, (50, 620))

    def run(self, max_frames=None):
        frame = 0
        running = True
        auto_restart_timer = 0

        while running:
            dt = self.clock.tick(60) / 1000.0
            frame += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.auto_play = not self.auto_play
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_n:
                        self.step_fly()
                    elif event.key == pygame.K_UP:
                        self.step_delay = max(0.05, self.step_delay - 0.1)
                    elif event.key == pygame.K_DOWN:
                        self.step_delay = min(2.0, self.step_delay + 0.1)

            # Auto-play loop
            if self.auto_play and not self.env.game_over:
                if time.time() - self.last_step_time >= self.step_delay:
                    self.step_fly()
                    self.last_step_time = time.time()
            elif self.auto_play and self.env.game_over:
                # Wait 1.5 seconds then auto-restart
                auto_restart_timer += dt
                if auto_restart_timer >= 1.5:
                    self.reset_game()
                    auto_restart_timer = 0

            # Render
            self.screen.fill(COLOR_BG)
            self.draw_minesweeper_board()
            self.draw_brain_hud()
            pygame.display.flip()

            if max_frames and frame >= max_frames:
                break

        pygame.quit()

if __name__ == "__main__":
    vis = FlyVisualizer()
    vis.run()
