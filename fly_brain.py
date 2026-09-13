import os
import json
import numpy as np
import torch
import torch.nn as nn

class FlyBrain:
    """
    Biological Connectome Neural Controller.
    Uses the MaleCNS v1.0 80-neuron visual-to-motor subgraph from HHMI Janelia.
    Integrates real synaptic contact counts, neurotransmitter signs (ACh: +1, GABA: -1),
    continuous leaky rate dynamics, and trainable sensory-motor readout interfaces.
    """
    def __init__(self, num_board_cells=36, device=None, circuit_path=None):
        self.num_board_cells = num_board_cells
        
        # Determine device (RTX 3060 CUDA if safe, else CPU)
        if device is None:
            if torch.cuda.is_available():
                # Check free memory
                try:
                    free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 2)
                    self.device = torch.device("cuda" if free_mem > 300 else "cpu")
                except Exception:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Locate circuit JSON
        if circuit_path is None:
            candidate_paths = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "malecns_circuit.json"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "malecns_circuit.json"),
                r"data/malecns_circuit.json"
            ]
            for cp in candidate_paths:
                if os.path.exists(cp):
                    circuit_path = cp
                    break

        if circuit_path is None or not os.path.exists(circuit_path):
            raise FileNotFoundError(f"Could not locate malecns_circuit.json in candidate paths: {candidate_paths}")

        self.circuit_path = circuit_path
        self._load_circuit(circuit_path)
        self._build_connectome_matrix()
        self._init_trainable_parameters()
        self.reset_state()

    def _load_circuit(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.nodes = data["nodes"]
        self.edges = data.get("edges", [])
        self.num_nodes = len(self.nodes)  # 80

        self.node_id_to_idx = {node["id"]: i for i, node in enumerate(self.nodes)}
        self.input_indices = [i for i, n in enumerate(self.nodes) if n.get("role") == "input"]
        self.inter_indices = [i for i, n in enumerate(self.nodes) if n.get("role") == "interneuron"]
        self.output_indices = [i for i, n in enumerate(self.nodes) if n.get("role") == "output"]

        self.num_inputs = len(self.input_indices)    # 32
        self.num_outputs = len(self.output_indices)  # 16

        # 3D spatial coordinates of soma for Brain HUD visualizer
        self.positions = np.array([n["position"] for n in self.nodes], dtype=np.float32)
        # Normalize positions to [-1, 1] bounding box for 3D rendering
        p_min = self.positions.min(axis=0)
        p_max = self.positions.max(axis=0)
        self.norm_positions = 2.0 * (self.positions - p_min) / (p_max - p_min + 1e-6) - 1.0

    def _build_connectome_matrix(self):
        """Build normalized synaptic weight matrix W from biological contacts and transmitter signs."""
        raw_W = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)

        for edge in self.edges:
            # Handles both [src, dst, contacts] list or dict {"source": .., "target": .., "contacts": ..}
            if isinstance(edge, list):
                src_id, dst_id, contacts = edge[0], edge[1], edge[2]
            else:
                src_id = edge.get("source", edge.get("src"))
                dst_id = edge.get("target", edge.get("dst"))
                contacts = edge.get("contacts", edge.get("weight", 1))

            if src_id in self.node_id_to_idx and dst_id in self.node_id_to_idx:
                s_idx = self.node_id_to_idx[src_id]
                d_idx = self.node_id_to_idx[dst_id]
                sign = self.nodes[s_idx].get("sign", 1)
                raw_W[s_idx, d_idx] = contacts * sign

        # Post-synaptic normalization
        W = np.zeros_like(raw_W)
        for d in range(self.num_nodes):
            col_abs_sum = np.sum(np.abs(raw_W[:, d]))
            if col_abs_sum > 0:
                W[:, d] = raw_W[:, d] / col_abs_sum

        self.W_connectome = torch.tensor(W, dtype=torch.float32, device=self.device)

    def _init_trainable_parameters(self):
        """
        Trainable interfaces:
        1. W_in: Sensory projection (Board state -> 32 Input Visual Neurons)
        2. W_out: Motor readout (16 Descending Neurons -> Board Action Logits)
        """
        # Small random initialization
        W_in = np.random.randn(self.num_inputs, self.num_board_cells).astype(np.float32) * 0.1
        W_out = np.random.randn(self.num_board_cells, self.num_outputs).astype(np.float32) * 0.1
        b_out = np.zeros(self.num_board_cells, dtype=np.float32)

        self.W_in = torch.tensor(W_in, dtype=torch.float32, device=self.device)
        self.W_out = torch.tensor(W_out, dtype=torch.float32, device=self.device)
        self.b_out = torch.tensor(b_out, dtype=torch.float32, device=self.device)

    def reset_state(self):
        """Reset internal membrane/firing rate state h."""
        self.h = torch.zeros(self.num_nodes, dtype=torch.float32, device=self.device)

    def forward_step(self, obs_flat, valid_mask=None, internal_steps=2):
        """
        Runs neural dynamics for internal_steps simulation ticks:
        h_{t+1} = (1 - alpha) * h_t + alpha * tanh(u_t + gamma * W^T h_t)
        
        obs_flat: 1D numpy array or torch tensor of normalized board state
        valid_mask: boolean array (True if tile can be clicked)
        """
        if not isinstance(obs_flat, torch.Tensor):
            x = torch.tensor(obs_flat, dtype=torch.float32, device=self.device)
        else:
            x = obs_flat.to(self.device)

        # 1. Sensory drive onto visual input neurons
        u_sensory = torch.matmul(self.W_in, x)  # Shape: [32]

        u_full = torch.zeros(self.num_nodes, dtype=torch.float32, device=self.device)
        u_full[self.input_indices] = u_sensory

        # 2. Recurrent leaky rate dynamics
        alpha = 0.7
        gamma = 1.4
        for _ in range(internal_steps):
            recurrent_input = torch.matmul(self.W_connectome.T, self.h)
            h_target = torch.tanh(u_full + gamma * recurrent_input)
            self.h = (1.0 - alpha) * self.h + alpha * h_target

        # 3. Readout from Descending Neurons (DNs)
        dn_activity = self.h[self.output_indices]  # Shape: [16]
        logits = torch.matmul(self.W_out, dn_activity) + self.b_out  # Shape: [num_board_cells]

        # 4. Mask invalid actions (already revealed tiles)
        if valid_mask is not None:
            mask_tensor = torch.tensor(valid_mask, dtype=torch.bool, device=self.device)
            logits = torch.where(mask_tensor, logits, torch.tensor(-1e9, device=self.device))

        return logits

    def pick_action(self, obs_flat, valid_mask=None, deterministic=False, temperature=1.0):
        logits = self.forward_step(obs_flat, valid_mask)
        if deterministic:
            action = int(torch.argmax(logits).item())
        else:
            probs = torch.softmax(logits / max(temperature, 1e-4), dim=0)
            dist = torch.distributions.Categorical(probs)
            action = int(dist.sample().item())
        return action

    def get_neural_activity(self):
        """Returns numpy copy of current neural activation vector h for HUD visualizer."""
        return self.h.detach().cpu().numpy()

    def get_parameters_flat(self):
        """Flattens all trainable parameters into a single 1D numpy array."""
        w_in_flat = self.W_in.detach().cpu().numpy().flatten()
        w_out_flat = self.W_out.detach().cpu().numpy().flatten()
        b_out_flat = self.b_out.detach().cpu().numpy().flatten()
        return np.concatenate([w_in_flat, w_out_flat, b_out_flat])

    def set_parameters_flat(self, flat_params):
        """Sets trainable parameters from a 1D numpy array."""
        idx = 0
        in_size = self.num_inputs * self.num_board_cells
        w_in_data = flat_params[idx:idx + in_size].reshape(self.num_inputs, self.num_board_cells)
        self.W_in.copy_(torch.tensor(w_in_data, dtype=torch.float32, device=self.device))
        idx += in_size

        out_size = self.num_board_cells * self.num_outputs
        w_out_data = flat_params[idx:idx + out_size].reshape(self.num_board_cells, self.num_outputs)
        self.W_out.copy_(torch.tensor(w_out_data, dtype=torch.float32, device=self.device))
        idx += out_size

        b_size = self.num_board_cells
        b_out_data = flat_params[idx:idx + b_size]
        self.b_out.copy_(torch.tensor(b_out_data, dtype=torch.float32, device=self.device))

    def save_weights(self, path):
        """Saves trainable weights to path."""
        weights = {
            "W_in": self.W_in.detach().cpu().numpy(),
            "W_out": self.W_out.detach().cpu().numpy(),
            "b_out": self.b_out.detach().cpu().numpy(),
            "num_board_cells": self.num_board_cells
        }
        torch.save(weights, path)

    def load_weights(self, path):
        """Loads trainable weights from path."""
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.W_in.copy_(torch.tensor(data["W_in"], dtype=torch.float32, device=self.device))
        self.W_out.copy_(torch.tensor(data["W_out"], dtype=torch.float32, device=self.device))
        self.b_out.copy_(torch.tensor(data["b_out"], dtype=torch.float32, device=self.device))

    def summary(self):
        return (f"FlyBrain (MaleCNS v1.0) on [{self.device}]\n"
                f"  Total Neurons: {self.num_nodes} (32 Inputs, 32 Interneurons, 16 Descending Outputs)\n"
                f"  Synaptic Edges: {len(self.edges)}\n"
                f"  Trainable Parameters: {len(self.get_parameters_flat())}")
