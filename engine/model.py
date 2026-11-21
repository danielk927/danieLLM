import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

def load_weights(model_id): # Returns dict[str, torch.Tensor]
    path = Path(snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"]))
    weights = {}
    for file in sorted(path.glob("*.safetensors")):
        with safe_open(file, "pt") as f:
            for k in f.keys(): 
                weights[k] = f.get_tensor(k).float()
    return weights

def load_config(model_id): # Returns contents of config.json as a Python dict
    path = Path(snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"]))
    with open(path / "config.json") as f: 
        return json.load(f)

def embed(ids, table): # [1, 5] token IDs -> [1, 5, 1024] embedding, 1 vector per token
    return table[ids]

def rms_norm(x, weight, eps): # Keeps every token's vector at a steady scale before attention/MLP 
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight