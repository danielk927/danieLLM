from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors import safe_open

path = Path(snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=["*.safetensors", "*.json"]))
for file in sorted(path.glob("*.safetensors")):
    with safe_open(file, "pt") as f:
        for k in f.keys():
            print(k, f.get_slice(k).get_shape(), f.get_slice(k).get_dtype())