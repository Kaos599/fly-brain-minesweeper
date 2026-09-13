import os
import json
import argparse
import torch
import numpy as np

def export_weights(model_path, circuit_path, out_path):
    print(f"Exporting PyTorch weights: {model_path}")
    print(f"Biological circuit:       {circuit_path}")
    print(f"Target web JSON:          {out_path}")

    if not os.path.exists(circuit_path):
        raise FileNotFoundError(f"Circuit file not found: {circuit_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(circuit_path, "r", encoding="utf-8") as f:
        circuit = json.load(f)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    w_in = checkpoint["W_in"]
    w_out = checkpoint["W_out"]
    b_out = checkpoint["b_out"]

    if isinstance(w_in, np.ndarray):
        w_in = w_in.tolist()
    elif isinstance(w_in, torch.Tensor):
        w_in = w_in.tolist()

    if isinstance(w_out, np.ndarray):
        w_out = w_out.tolist()
    elif isinstance(w_out, torch.Tensor):
        w_out = w_out.tolist()

    if isinstance(b_out, np.ndarray):
        b_out = b_out.tolist()
    elif isinstance(b_out, torch.Tensor):
        b_out = b_out.tolist()

    export_data = {
        "version": circuit.get("version", "malecns-v1.0"),
        "nodes": circuit["nodes"],
        "edges": circuit["edges"],
        "W_in": w_in,
        "W_out": w_out,
        "b_out": b_out,
        "num_board_cells": checkpoint.get("num_board_cells", len(b_out))
    }

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f)

    file_size_kb = os.path.getsize(out_path) // 1024
    print(f"Successfully exported web model to: {out_path} ({file_size_kb} KB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch FlyBrain weights to WebGL JSON")
    parser.add_argument("--model", type=str, default="checkpoints/best_fly_model.pt", help="Path to PyTorch checkpoint")
    parser.add_argument("--circuit", type=str, default="data/malecns_circuit.json", help="Path to circuit JSON")
    parser.add_argument("--output", type=str, default="fly_model_web.json", help="Path to output web JSON")
    args = parser.parse_args()

    export_weights(args.model, args.circuit, args.output)
