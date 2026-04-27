import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-chat-hf"   # 远端仓库名
model_path = "/gemini/space/sunminghao/AutoDAN-main-10G/models/llama2/llama-2-7b-chat-hf"  # 本地缓存目录

# 先从远端加载（需要已登录、且有访问权限）
# 如果没有 GPU 或没装 accelerate，先别用 device_map="auto"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False,
    # use_auth_token=True,   # 旧写法；如果没用 CLI 登录，可以解开这行
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",          # 或者 torch.float16 / torch.bfloat16（视硬件而定）
    low_cpu_mem_usage=True,
    use_safetensors=True,
    # device_map="auto",        # 如未安装 accelerate，请先注释掉这行
    # use_auth_token=True,      # 同上
)

# 保存到你想要的本地目录（会自动创建目录）
tokenizer.save_pretrained(model_path)
model.save_pretrained(model_path)
print(f"✅ 已将模型与分词器保存到: {model_path}")






