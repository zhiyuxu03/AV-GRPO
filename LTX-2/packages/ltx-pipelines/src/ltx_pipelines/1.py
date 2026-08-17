import torch
from safetensors.torch import load_file, save_file

# 你的输入路径（FSDP训练的坏模型）
INPUT_CKPT = "/mnt/petrelfs/zhiyuxu/LTX-2/outputs/ltx2_full_finetune/checkpoints/model_weights_step_00001.safetensors"
# 输出路径（清洗后的好模型）
OUTPUT_CKPT = "/mnt/petrelfs/zhiyuxu/LTX-2/clean_model.safetensors"

print("Loading broken FSDP checkpoint...")
sd = load_file(INPUT_CKPT)

print("Cleaning FSDP garbage keys...")
clean_sd = {}
for k, v in sd.items():
    # 删掉所有 FSDP 垃圾 key（这些是炸 header 的元凶）
    if k.startswith("_") or "fsdp" in k or "checkpoint" in k:
        continue
    clean_sd[k] = v.to(torch.bfloat16)

print(f"Saving clean checkpoint to: {OUTPUT_CKPT}")
# 保存极简 header，绝对不炸
save_file(clean_sd, OUTPUT_CKPT, metadata={"model_type": "ltx"})

print("✅ DONE! 干净模型已生成，可以直接推理！")