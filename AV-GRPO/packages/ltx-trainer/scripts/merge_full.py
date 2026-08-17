import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from tqdm import tqdm
import time

def merge_weights_fast(original_ckpt_path, trained_ckpt_path, output_ckpt_path):
    """
    Quickly merge weights while preserving the original file's metadata.
    Fix: Automatically clean up the _fsdp_wrapped_module. prefix left over from FSDP export.
    """
    
    print(f"Loading original checkpoint from: {original_ckpt_path}")
    start_time = time.time()
    
    # Load original weights and metadata
    if original_ckpt_path.endswith('.safetensors'):
        original_state = load_file(original_ckpt_path, device='cpu')
        with safe_open(original_ckpt_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
    else:
        original_state = torch.load(original_ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(original_state, dict) and 'state_dict' in original_state:
            original_state = original_state['state_dict']
        metadata = None
    
    print(f"✓ Original loaded: {len(original_state)} keys, metadata: {bool(metadata)}")
    
    # Load trained weights
    print(f"Loading trained checkpoint from: {trained_ckpt_path}")
    if trained_ckpt_path.endswith('.safetensors'):
        trained_state = load_file(trained_ckpt_path, device='cpu')
    else:
        trained_state = torch.load(trained_ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(trained_state, dict) and 'state_dict' in trained_state:
            trained_state = trained_state['state_dict']
    
    print(f"✓ Trained loaded: {len(trained_state)} keys")
    
    # Create a new dictionary
    merged_state = {}
    for k, v in original_state.items():
        merged_state[k] = v.clone() if isinstance(v, torch.Tensor) else v
    
    # Rule-based matching
    PREFIX = 'model.diffusion_model.'
    CLEAN_FSDP_PREFIX = True  # Enable for legacy FSDP-exported weights, disable for new-style unwrapped exports
    
    matched_count = 0
    for trained_key, trained_value in tqdm(trained_state.items(), desc="Merging"):
        if CLEAN_FSDP_PREFIX:
            clean_trained_key = trained_key.replace("_fsdp_wrapped_module.", "")
        else:
            clean_trained_key = trained_key

        original_key = PREFIX + clean_trained_key
        if original_key in merged_state:
            if trained_value.shape == merged_state[original_key].shape:
                merged_state[original_key] = trained_value.clone()
                matched_count += 1
    
    print(f"\n✓ Replaced {matched_count} keys")
    
    # Save with metadata
    print(f"Saving to: {output_ckpt_path}")
    if output_ckpt_path.endswith('.safetensors'):
        save_file(merged_state, output_ckpt_path, metadata=metadata)
    else:
        torch.save(merged_state, output_ckpt_path)
    
    print(f"✓ Done! Metadata preserved: {bool(metadata)}")

if __name__ == "__main__":
    original_ckpt_path = ".../AV-GRPO/LTX-2.3/ltx-2.3-22b-dev.safetensors"
    trained_ckpt_path = ".../AV-GRPO/outputs/ltx2_full/checkpoints/model_weights_step_000xxx.safetensors"
    output_ckpt_path = ".../AV-GRPO/outputs/ltx2_full/checkpoints/xxx.safetensors"
    
    merge_weights_fast(original_ckpt_path, trained_ckpt_path, output_ckpt_path)