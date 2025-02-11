from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_models_tokenizer(checkpoint_path):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
       #torch_device='cpu',
        torch_dtype=torch.float16
    )
    # return the result
    return model
checkpoint_path = ''
model = load_models_tokenizer(checkpoint_path)
out_dir = ''
model.save_pretrained(out_dir, use_safetensors=True)
