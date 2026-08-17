import torch
from safetensors.torch import load_file, save_file
import os

# 读取有问题的 checkpoint
checkpoint_path = "/mnt/petrelfs/zhiyuxu/LTX-2/outputs/ltx2_full_finetune/checkpoints/model_weights_step_00001.safetensors"

try:
    # 尝试读取，如果失败，可能需要用其他方式
    state_dict = load_file(checkpoint_path)
except Exception as e:
    print(f"Cannot read directly: {e}")
    # 如果有备份，用备份；否则需要重新训练
    exit(1)

# 重新保存，不包含 metadata
new_path = checkpoint_path.replace(".safetensors", "_fixed.safetensors")
save_file(state_dict, new_path, metadata=None)
print(f"Saved fixed checkpoint to {new_path}")