import os
import json
import torch
import numpy as np

SCRATCH_DIR = r"checkpoints"
MODEL_PATH = os.path.join(SCRATCH_DIR, "best_fly_model.pt")
CIRCUIT_PATH = os.path.join(SCRATCH_DIR, "malecns_circuit.json")
OUT_PATH = os.path.join(SCRATCH_DIR, "fly_model_web.json")

print("Exporting PyTorch FlyBrain weights and circuit to web JSON...")

with open(CIRCUIT_PATH, "r", encoding="utf-8") as f:
    circuit = json.load(f)

checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

export_data = {
    "version": circuit.get("version", "malecns-v1.0"),
    "nodes": circuit["nodes"],
    "edges": circuit["edges"],
    "W_in": checkpoint["W_in"].tolist(),
    "W_out": checkpoint["W_out"].tolist(),
    "b_out": checkpoint["b_out"].tolist(),
    "num_board_cells": checkpoint.get("num_board_cells", 36)
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(export_data, f)

print(f"Successfully exported web model to: {OUT_PATH} ({os.path.getsize(OUT_PATH) // 1024} KB)")
