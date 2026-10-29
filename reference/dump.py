import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer 

name, dev = "Qwen/Qwen3-0.6B", "cuda" if torch.cuda.is_available() else "cpu" #every tensor the model is fed must live in the same place as the weights
tok = AutoTokenizer.from_pretrained(name) #qwen3's tokenizer, string -> integer for each token
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(dev).eval()
ids = tok("The capital of France is", return_tensors="pt").input_ids.to(dev) #runs the tokenizer
with torch.no_grad(): #turns off autograd (since we're not training the model, saves memory)
    out = model(ids, output_hidden_states=True) #runs full transformer + lm_head
torch.save({"ids": ids.cpu(), "logits": out.logits.cpu(), "hidden": [h.cpu() for h in out.hidden_states]}, "reference.pt") #logits are the raw predictions for the next token
          