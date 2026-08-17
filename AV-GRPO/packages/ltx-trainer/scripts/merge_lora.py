import re
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from tqdm import tqdm

# ===================== Configuration =====================
INPUT_PATH    = ".../AV-GRPO/outputs/ltx2_lora/checkpoints/model_weights_step_000xx.safetensors"
CLEANED_OUT   = ".../AV-GRPO/outputs/ltx2_lora/checkpoints/xx.safetensors"
OFFICIAL_BASE = ".../AV-GRPO/LTX-2.3/ltx-2.3-22b-dev.safetensors"
MERGED_OUT    = ".../AV-GRPO/outputs/ltx2_lora/checkpoints/xx_lora.safetensors"

# LoRA hyperparameters — must match training exactly
LORA_RANK  = 32
LORA_ALPHA = 32
STRENGTH   = 1.0
SCALE      = LORA_ALPHA / LORA_RANK * STRENGTH

# Prefix for the transformer part in the base model
BASE_PREFIX = "model.diffusion_model."
# =========================================================


def clean_fsdp_prefix(key: str) -> str:
    """Remove prefixes introduced by FSDP + PEFT nesting."""
    key = re.sub(r'^(base_model\.model\.)+', '', key)
    key = key.replace('._fsdp_wrapped_module.', '.')
    key = re.sub(r'^_fsdp_wrapped_module\.', '', key)
    return key


def clean_all_prefix(key: str) -> str:
    """Clean all wrapper prefixes: _orig_mod / base_model.model / _fsdp_wrapped_module."""
    key = key.replace("_orig_mod.", "")
    key = clean_fsdp_prefix(key)
    return key


def extract_and_clean_lora(input_path: str, output_path: str):
    """Step 1: Extract and clean LoRA weights from the training checkpoint."""
    print("=" * 80)
    print("Step 1/2: Extracting and cleaning LoRA weights...")
    print("=" * 80)

    sd = load_file(input_path)
    lora_sd = {}

    for k, v in sd.items():
        if 'lora_' not in k:
            continue

        # 1. Remove _orig_mod. prefix from torch.compile
        new_k = k.replace('_orig_mod.', '')

        # 2. Remove PEFT + FSDP nesting prefixes
        new_k = clean_fsdp_prefix(new_k)

        # 3. Remove .default suffix (some frameworks add this)
        new_k = new_k.replace('.default.weight', '.weight')

        lora_sd[new_k] = v

    if len(lora_sd) == 0:
        raise ValueError("No LoRA parameters extracted. Please check the input checkpoint!")

    save_file(lora_sd, output_path)
    print(f"Extracted and cleaned {len(lora_sd)} LoRA weights, saved to: {output_path}")
    return lora_sd


def merge_lora_to_base(cleaned_path: str, base_path: str, merged_out: str):
    """Step 2: Merge cleaned LoRA weights into the base model."""
    print("\n" + "=" * 80)
    print("Step 2/2: Merging LoRA into base model...")
    print("=" * 80)

    # Load base model weights
    official_sd = load_file(base_path, device="cpu")
    print(f"Base weights loaded: {len(official_sd)} keys")

    # Read full metadata from the official base
    with safe_open(base_path, framework="pt", device="cpu") as f:
        official_metadata = f.metadata()
    print(f"Base metadata loaded. Fields: {list(official_metadata.keys()) if official_metadata else 'empty'}")

    # Load cleaned LoRA checkpoint
    trained_sd = load_file(cleaned_path, device="cpu")
    print(f"Cleaned LoRA checkpoint loaded: {len(trained_sd)} keys")

    # Extract and clean all LoRA params (second pass as a safety net)
    print("\nExtracting and cleaning LoRA params...")
    lora_params = {}
    for k, v in trained_sd.items():
        if "lora_" not in k:
            continue
        clean_k = clean_all_prefix(k)
        clean_k = clean_k.replace(".default.", ".")
        lora_params[clean_k] = v

    print(f"Extracted {len(lora_params)} LoRA params")

    # Print sample keys for verification
    print("\nSample cleaned LoRA keys:")
    for i, k in enumerate(lora_params.keys()):
        if i >= 3:
            break
        print(f"   {k} | shape: {tuple(lora_params[k].shape)}")

    # Pair lora_A and lora_B
    print("\nPairing LoRA modules...")
    lora_pairs = {}
    for k, v in lora_params.items():
        if ".lora_A." in k:
            base_name = k.replace(".lora_A.", ".")
            lora_pairs.setdefault(base_name, [None, None])[0] = v
        elif ".lora_B." in k:
            base_name = k.replace(".lora_B.", ".")
            lora_pairs.setdefault(base_name, [None, None])[1] = v

    complete_pairs = {k: v for k, v in lora_pairs.items() if v[0] is not None and v[1] is not None}
    broken_pairs = {k: v for k, v in lora_pairs.items() if v[0] is None or v[1] is None}
    print(f"Complete pairs: {len(complete_pairs)}")
    if broken_pairs:
        print(f"Incomplete pairs: {len(broken_pairs)}")

    # Perform merging
    print("\nMerging weights...")
    merged_sd = official_sd.copy()
    fused_count = 0
    not_found = 0

    for base_name, (lora_A, lora_B) in tqdm(complete_pairs.items(), desc="Fusing"):
        target_key = BASE_PREFIX + base_name

        if target_key not in merged_sd:
            not_found += 1
            continue

        base_weight = merged_sd[target_key]
        delta = torch.matmul(lora_B.float(), lora_A.float()) * SCALE
        merged_sd[target_key] = base_weight + delta.to(base_weight.dtype)
        fused_count += 1

    print(f"\nSuccessfully fused: {fused_count} modules")
    print(f"Not found in base: {not_found} modules")

    if fused_count == 0:
        raise RuntimeError("No modules were fused. Please check prefixes and path matching!")

    # Save with official metadata
    print(f"\nSaving merged weights to: {merged_out}")
    save_file(merged_sd, merged_out, metadata=official_metadata)

    print("Merging complete. Metadata fully preserved!")

    # Quick verification
    print("\nVerification:")
    sample_key = BASE_PREFIX + list(complete_pairs.keys())[0]
    if sample_key in official_sd and sample_key in merged_sd:
        diff = (merged_sd[sample_key].float() - official_sd[sample_key].float()).abs().mean()
        print(f"   Mean absolute weight change on sample layer: {diff.item():.6f}")

    with safe_open(merged_out, framework="pt", device="cpu") as f:
        out_meta = f.metadata()
    print(f"   Output metadata intact: {out_meta is not None and len(out_meta) > 0}")


# ===================== Main =====================
if __name__ == "__main__":
    # Step 1: Extract and clean LoRA weights
    extract_and_clean_lora(INPUT_PATH, CLEANED_OUT)

    # Step 2: Merge into base model
    merge_lora_to_base(CLEANED_OUT, OFFICIAL_BASE, MERGED_OUT)