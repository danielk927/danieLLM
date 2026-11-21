import torch
from engine.model import load_weights, embed

def test_embedding():
    ref = torch.load("reference.pt")
    w = load_weights("Qwen/Qwen3-0.6B")
    out = embed(ref["ids"], w["model.embed_tokens.weight"])
    torch.testing.assert_close(out, ref["hidden"][0], rtol=0, atol=0)
