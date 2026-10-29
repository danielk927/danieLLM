import torch
from transformers import AutoTokenizer

ref = torch.load("reference.pt")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

print(tok.convert_ids_to_tokens(ref["ids"][0]))
print(len(ref["hidden"]), ref["hidden"][0].shape, ref["logits"].shape)
print(repr(tok.decode(ref["logits"][0, -1].argmax())))

# ['The', 'Ġcapital', 'Ġof', 'ĠFrance', 'Ġis']
# 29 torch.Size([1, 5, 1024]) torch.Size([1, 5, 151936])
# ' Paris'