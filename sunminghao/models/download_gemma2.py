import os
import subprocess
import sys

# Ensure transformers and accelerate are up to date
def install_package(package):
    print(f"Installing/Updating {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])

try:
    import transformers
    import accelerate
except ImportError:
    install_package("transformers")
    install_package("accelerate")
    import transformers

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model ID on HuggingFace
model_name = "google/gemma-2-9b-it"
# Local path to save
base_model_path = "./models/gemma2/gemma-2-9b-it"

print(f"Downloading {model_name} to {base_model_path}...")
os.makedirs(base_model_path, exist_ok=True)

try:
    # Gemma 2 often requires an access token (gated model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Remove device_map='auto' and low_cpu_mem_usage=True to avoid needing 'accelerate'
    # We only need to download, so loading to CPU is fine.
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.float16,
                                                 use_cache=False)
    
    print("Saving model and tokenizer...")
    model.save_pretrained(base_model_path, from_pt=True)
    tokenizer.save_pretrained(base_model_path, from_pt=True)
    print("Download complete.")
except Exception as e:
    print(f"Error downloading model: {e}")
    print("Please ensure you have access to the model on HuggingFace and are logged in (huggingface-cli login) or have HF_TOKEN set.")
