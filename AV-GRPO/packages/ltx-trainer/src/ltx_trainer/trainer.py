import os
import time
import warnings
import gc
import numpy as np
warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is recommended to use sourceTensor.clone")
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.fx")
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple, Any, Callable
from ltx_trainer.training_strategies.base_strategy import ModelInputs
from ltx_trainer.reward_computation import _compute_vq_batch, _compute_other_metrics
import torch
import wandb
import math
import json
import yaml
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper
from pydantic import BaseModel
from safetensors.torch import load_file, save_file
from ltx_core.quantization import QuantizationPolicy
quant_policy = QuantizationPolicy.fp8_cast()
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    StepLR,
)
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F  # noqa: N812

from ltx_core.model.transformer.model import X0Model
from ltx_core.text_encoders.gemma import convert_to_additive_mask
from ltx_trainer import logger
from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.config_display import print_config
from ltx_trainer.datasets import PrecomputedDataset
from ltx_trainer.gpu_utils import free_gpu_memory, free_gpu_memory_context, get_gpu_memory_gb
from ltx_trainer.hf_hub_utils import push_to_hub
from ltx_trainer.model_loader import load_embeddings_processor, load_text_encoder
from ltx_trainer.model_loader import load_model as load_ltx_model
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.quantization import quantize_model
from ltx_trainer.timestep_samplers import SAMPLERS
from ltx_trainer.training_strategies import get_training_strategy
from ltx_trainer.utils import open_image_as_srgb, save_image
from ltx_trainer.validation_sampler import CachedPromptEmbeddings, GenerationConfig, ValidationSampler
from ltx_trainer.video_utils import read_video, save_video

# Disable irrelevant warnings from transformers
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Silence bitsandbytes warnings about casting
warnings.filterwarnings(
    "ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)

# Disable progress bars if not main process
IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "0") == "0"
if not IS_MAIN_PROCESS:
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()

StepCallback = Callable[[int, int, list[Path]], None]  # (step, total, list[sampled_video_path]) -> None

MEMORY_CHECK_INTERVAL = 200


class TrainingStats(BaseModel):
    """Statistics collected during training"""

    total_time_seconds: float
    steps_per_second: float
    samples_per_second: float
    peak_gpu_memory_gb: float
    global_batch_size: int
    num_processes: int


class LtxvTrainer:
    def __init__(self, trainer_config: LtxTrainerConfig) -> None:
        self._config = trainer_config
        if IS_MAIN_PROCESS:
            print_config(trainer_config)
        self._training_strategy = get_training_strategy(self._config.training_strategy)
        self._cached_validation_embeddings = self._load_text_encoder_and_cache_embeddings()
        self._load_models()
        self._setup_accelerator()
        self._load_checkpoint()
        self._prepare_models_for_training()
        self._dataset = None
        self._global_step = -1
        self._checkpoint_paths = []
        self._init_wandb()
        self._trajectory_cache = None   
        self._video_advantages = None          
        self._audio_advantages = None          
        self._init_reference_model()
        self._beta_kl = getattr(trainer_config.optimization, 'beta_kl', 0.04)  # KL coefficient
        self._current_freeze_modality = getattr(self._config.validation, 'freeze_modality', 'video')
        # Note: freeze_modality='video' means freezing video → training audio
        
        if IS_MAIN_PROCESS:
            print(f"Initial freeze_modality = {self._current_freeze_modality}")
        self._reinit_optimizer_for_modality(self._current_freeze_modality)
        self._validation_count = 0
        self._reward_history = []   

    def _reinit_optimizer_for_modality(self, freeze_modality: str) -> None:
        if freeze_modality == 'video':
            train_mode = 'audio'
        elif freeze_modality == 'audio':
            train_mode = 'video'
        else:
            raise ValueError(...)

        self._collect_trainable_params(train_mode=train_mode)

        for p in self._transformer.parameters():
            p.requires_grad = True

        if hasattr(self, '_optimizer') and self._optimizer is not None:
            
            self._optimizer.state.clear()
            self._optimizer.param_groups[0]['params'] = self._trainable_params
            if IS_MAIN_PROCESS:
                print(f"[Switch] Optimizer params replaced and state cleared for {freeze_modality}")
        else:
            self._init_optimizer(params_override=self._trainable_params)

    def _generate_trajectories(self, freeze_modality: str) -> None:
        
        import torch.distributed as dist
        from pathlib import Path

        rank = dist.get_rank() if dist.is_initialized() else 0
        
        old_freeze = self._config.validation.freeze_modality
        self._config.validation.freeze_modality = freeze_modality

        
        cached_embeddings = (
            self._cached_validation_embeddings[rank % len(self._cached_validation_embeddings)]
            if self._cached_validation_embeddings is not None and len(self._cached_validation_embeddings) > 0
            else None
        )

        
        from ltx_trainer.validation_sampler import GenerationConfig
        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(exist_ok=True, parents=True)

     
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        samples_per_rank = 4   

        gen_config = GenerationConfig(
            prompt=self._config.validation.prompts[0] if self._config.validation.prompts else "A person walking",
            negative_prompt=self._config.validation.negative_prompt,
            height=self._config.validation.video_dims[1],
            width=self._config.validation.video_dims[0],
            num_frames=self._config.validation.video_dims[2],
            frame_rate=self._config.validation.frame_rate,
            num_inference_steps=self._config.validation.inference_steps,
            guidance_scale=self._config.validation.guidance_scale,
            seed=self._config.validation.seed,
            generate_audio=self._config.validation.generate_audio,
            cached_embeddings=cached_embeddings,
            stg_scale=self._config.validation.stg_scale,
            stg_blocks=self._config.validation.stg_blocks,
            stg_mode=self._config.validation.stg_mode,
            num_samples=samples_per_rank * world_size,
            output_dir=output_dir,
            output_prefix=f"traj_gen_{freeze_modality}",
            freeze_modality=freeze_modality,
            enable_sde=False,          
            sde_noise_level=0.0,
            skip_reward_processing=True,   
        )

        self._sample_videos(progress=None, skip_reward_processing=True, freeze_modality_override=freeze_modality)

        #
        self._config.validation.freeze_modality = old_freeze

    def _init_reference_model(self):
        """Load the reference model θ_ref, used to compute the reference velocity field in the KL divergence"""
        from safetensors.torch import load_file
        from ltx_trainer.model_loader import load_model
        
        model_path = self._config.model.model_path
        
        if IS_MAIN_PROCESS:
            print(f"[GRPO] Loading reference model from {model_path}...")
        
        
        components = load_model(
            checkpoint_path=model_path,
            device="cpu",  
            dtype=torch.bfloat16,
            with_video_vae_encoder=False,
            with_video_vae_decoder=False,
            with_audio_vae_decoder=False,
            with_vocoder=False,
            with_text_encoder=False, 
            text_encoder_path=None,
        )
        
        self._ref_transformer = components.transformer
        
        
        self._ref_transformer.to(self._accelerator.device)
        self._ref_transformer.eval()
        for param in self._ref_transformer.parameters():
            param.requires_grad = False
        
     
        self._ref_transformer = self._accelerator.prepare(self._ref_transformer)
        
        if IS_MAIN_PROCESS:
            print(f"[GRPO] ✅ Reference model loaded and frozen")

    def _load_advantages(self):
        """Load the most recently saved advantage file."""
        adv_dir = Path(self._config.output_dir) / "advantages"
        if not adv_dir.exists():
            if self._accelerator.is_main_process:
                logger.warning(f"Advantage directory does not exist: {adv_dir}")
            self._video_advantages = None
            self._audio_advantages = None
            return

        adv_files = sorted(adv_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split('_')[1]))
        if not adv_files:
            if self._accelerator.is_main_process:
                logger.warning(f"Advantage directory is empty: {adv_dir}")
            self._video_advantages = None
            self._audio_advantages = None
            return

        latest = adv_files[-1]
        data = torch.load(latest, map_location="cpu")
        self._video_advantages = data.get('video_advantage', None)
        self._audio_advantages = data.get('audio_advantage', None)
        if self._accelerator.is_main_process:
            logger.info(f"📥 Loading advantage values: {latest} (number of video advantages: {len(self._video_advantages) if self._video_advantages is not None else 0})")

    
    def _load_trajectory_cache(self):
        """Load pre-saved trajectory cache file"""
        cache_path = str(Path(self._config.output_dir).parent / "trajectory" / "validation_trajectory_cached.pt")
        if not os.path.exists(cache_path):
            self._trajectory_cache = None
            return

        cache_data = torch.load(cache_path, map_location="cpu")
        data = cache_data["data"]
        
 
        num_steps = len(data["video_sigma"])
        
  
        def get_field(field_name):
            if field_name in data:
                return [data[field_name][i] for i in range(num_steps)]
            else:
                return [None] * num_steps

        self._trajectory_cache = {
            # video
            "video_latent": get_field("video_latent"),
            "next_video_latent": get_field("next_video_latent"),
            "video_log_prob": get_field("video_log_prob"),
            "video_x0_pred": get_field("video_x0_pred"),
            "video_sigma": get_field("video_sigma"),
            "video_dt_abs": get_field("video_dt_abs"),
            "video_sigma_t_eff": get_field("video_sigma_t_eff"),
            "video_noise": get_field("video_noise"),
            # audio
            "audio_latent": get_field("audio_latent"),
            "next_audio_latent": get_field("next_audio_latent"),
            "audio_log_prob": get_field("audio_log_prob"),
            "audio_x0_pred": get_field("audio_x0_pred"),
            "audio_sigma": get_field("audio_sigma"),
            "audio_dt_abs": get_field("audio_dt_abs"),
            "audio_sigma_t_eff": get_field("audio_sigma_t_eff"),
            "audio_noise": get_field("audio_noise"),
            "sigma": get_field("video_sigma"),   
        }
        
        if self._accelerator.is_main_process:
            logger.info(f"✅ Trajectory cache loaded, containing {num_steps} steps.")

    def train(
            self,
            disable_progress_bars: bool = False,
            step_callback: StepCallback | None = None,
        ) -> tuple[Path, TrainingStats]:
        device = self._accelerator.device
        start_mem = get_gpu_memory_gb(device)
        train_start_time = time.time()

        set_seed(self._config.seed)
        logger.debug(f"Process {self._accelerator.process_index} using seed: {self._config.seed}")

        self._init_optimizer()
        self._init_dataloader()
        self._load_trajectory_cache()
        data_iter = iter(self._dataloader)
        self._init_timestep_sampler()
        self._load_advantages()
        if not hasattr(self, '_optimizer') or self._optimizer is None:
            self._reinit_optimizer_for_modality(self._current_freeze_modality)
        self._accelerator.wait_for_everyone()
        
       
        import torch.distributed as dist
        adv_dir = Path(self._config.output_dir) / "advantages"
        if dist.is_initialized():
            if dist.get_rank() == 0:
                if adv_dir.exists():
                    for f in adv_dir.glob("step_*_rank*.pt"):
                        f.unlink()
                    print(f"[INIT] Cleared old advantage files in {adv_dir}")
            dist.barrier()
        else:
            if adv_dir.exists():
                for f in adv_dir.glob("step_*_rank*.pt"):
                    f.unlink()

        reward_history_path = Path(self._config.output_dir) / "reward_history.json"
        if dist.is_initialized():
            if dist.get_rank() == 0:
                if reward_history_path.exists():
                    reward_history_path.unlink()
                    print(f"[INIT] Deleted old reward history: {reward_history_path}")
            dist.barrier()
        else:
            if reward_history_path.exists():
                reward_history_path.unlink()

        Path(self._config.output_dir).mkdir(parents=True, exist_ok=True)
        self._save_config()

        logger.info("🚀 Starting training...")
        if IS_MAIN_PROCESS:
            logger.info(f"GPU memory: {get_gpu_memory_gb(device):.2f} GB")

        
        self._wandb_run = None
        import wandb

       
        if self._accelerator.is_main_process:
            print(f"[WANDB] Main process starting initialization")
            
           
            wandb.login(
                key='',
                relogin=False
            )
            

            self._wandb_run = wandb.init(
                project="AV-GRPO",
                entity="",
                mode="offline",
                dir = str(Path(self._config.output_dir).parent)
                name="AV-GRPO-training",
                config=self._config.__dict__,
                save_code=True
            )
            
            if self._wandb_run is not None:
                print(f"✅ wandb initialized successfully")
                print(f"📁 Log save path: {self._wandb_run.dir}")


        progress_enabled = IS_MAIN_PROCESS and not disable_progress_bars
        progress = TrainingProgress(
            enabled=progress_enabled,
            total_steps=self._config.optimization.steps,
        )
        if IS_MAIN_PROCESS and disable_progress_bars:
            logger.warning("Progress bars disabled.")

        self._transformer.train()
        self._global_step = 0
        peak_mem_during_training = start_mem
        sampled_videos_paths = None

        
        BLOCK_CONFIG = {
            'audio': (4, 1),   # Freeze audio (training video): 4 consecutive blocks, 1 step per block
            'video': (4, 1),   # Freeze video (training audio): 4 consecutive blocks, 1 step per block
        }
        # ───────────────────────────────────────────

        total_grpo_steps = self._config.optimization.steps
        if IS_MAIN_PROCESS:
            print(f"\n{'='*60}")
            print(f"[GRPO] Alternating Training Configuration:")
            print(f"[GRPO]   Total GRPO steps: {total_grpo_steps}")
            print(f"[GRPO]   Block config: {BLOCK_CONFIG}")
            print(f"[GRPO]   Current freeze modality (initial): {self._current_freeze_modality}")
            print(f"{'='*60}\n")

        
        self._current_cycle_idx = 0          # Which cycle is currently being used (32 prompts make up one cycle)
        self._current_modality_step = 0      # Step count within the current modality (0-3)
     
        current_freeze = self._current_freeze_modality
        blocks, steps_per_block = BLOCK_CONFIG[current_freeze]
        remaining_blocks = blocks
        block_steps_left = steps_per_block

        with progress:
            if self._config.validation.interval and not self._config.validation.skip_initial_validation:
                sampled_videos_paths = self._sample_videos(progress)
                # if IS_MAIN_PROCESS and sampled_videos_paths and self._config.wandb.log_validation_videos:
                #     self._log_validation_samples(sampled_videos_paths, self._config.validation.prompts)
                self._accelerator.wait_for_everyone()
                data_iter = iter(self._dataloader)

            self._accelerator.wait_for_everyone()

            for grpo_step in range(total_grpo_steps):
                step_start_time = time.time()

                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self._dataloader)
                    batch = next(data_iter)

                
                loss, policy_loss, kl_loss = self._training_step(batch)

                
                if getattr(self, '_skip_modality', False):
                    self._skip_modality = False
                    if IS_MAIN_PROCESS:
                        print("[WARN] NaN advantage detected, switching to the other modality")

                    new_freeze = 'audio' if current_freeze == 'video' else 'video'
                    if IS_MAIN_PROCESS:
                        print(f"[Switch] Forced switch from {current_freeze} to {new_freeze} due to NaN (global_step={self._global_step})")

                    self._current_freeze_modality = new_freeze
                    self._config.validation.freeze_modality = new_freeze
                    current_freeze = new_freeze
                    blocks, steps_per_block = BLOCK_CONFIG[current_freeze]
                    remaining_blocks = blocks
                    block_steps_left = steps_per_block
                    self._current_modality_step = 0 

                    self._reinit_optimizer_for_modality(new_freeze)
                   

                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(device=device)
                    gc.collect()
                    if dist.is_initialized():
                        dist.barrier()
                    self._sample_videos(
                        progress=progress,
                        skip_reward_processing=False,
                        freeze_modality_override=new_freeze,
                        force_refresh_cache=True
                    )
                    self._accelerator.wait_for_everyone()
                    data_iter = iter(self._dataloader)
                    continue

                self._global_step += 1
                total = self._config.optimization.steps

                
                start_lr = 1e-5          
                peak_lr = 1e-5          
                final_lr = 0          
                warmup_steps = 1       

                if self._global_step <= warmup_steps:
                    warmup_ratio = self._global_step / warmup_steps   
                    new_lr = start_lr + (peak_lr - start_lr) * warmup_ratio
                else:
                    decay_steps = total - warmup_steps
                    step_in_decay = self._global_step - warmup_steps
                    cosine_progress = step_in_decay / decay_steps
                    new_lr = final_lr + 0.5 * (peak_lr - final_lr) * (1 + math.cos(math.pi * cosine_progress))

                for param_group in self._optimizer.param_groups:
                    param_group['lr'] = new_lr

                if torch.is_tensor(loss):
                    loss_value = loss.item()
                else:
                    loss_value = float(loss) if loss is not None else 0.0

                if torch.distributed.is_initialized():
                    loss_tensor = torch.tensor([loss_value], device=device)
                    torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
                    global_avg_loss = loss_tensor.item()
                else:
                    global_avg_loss = loss_value

                step_time = time.time() - step_start_time

                progress.update_training(
                    loss=global_avg_loss,
                    lr=self._optimizer.param_groups[0]["lr"],
                    step_time=step_time,
                    advance=True,
                )

                if IS_MAIN_PROCESS:
                    print(f"\n{'='*60}")
                    print(f"[GRPO] ✅ Step {self._global_step}/{total_grpo_steps} completed")
                    print(f"[GRPO] 📊 Global Loss: {global_avg_loss:.6f}")
                    print(f"[GRPO] 📊 Policy Loss: {policy_loss:.6f}")
                    print(f"[GRPO] 📊 KL Loss: {kl_loss:.6f}")
                    print(f"[GRPO] 📊 Learning Rate: {self._optimizer.param_groups[0]['lr']:.2e}")
                    print(f"[GRPO] 📊 Step Time: {step_time:.2f}s")
                    print(f"[GRPO] 📊 Current Modality: {'Video' if current_freeze == 'audio' else 'Audio'}")
                    print(f"{'='*60}\n")
                    
                  
                    if self._wandb_run is not None:
                        wandb.log({
                            "train/loss": global_avg_loss,
                            "train/policy_loss": policy_loss,
                            "train/kl_loss": kl_loss,
                            "train/learning_rate": self._optimizer.param_groups[0]["lr"],
                            "train/step_time": step_time,
                            "train/freeze_modality": current_freeze,
                            "train/cycle_idx": self._current_cycle_idx,
                            "train/modality_step": self._current_modality_step
                        }, step=self._global_step)

                
                block_steps_left -= 1
                if block_steps_left == 0:
                    remaining_blocks -= 1
                    self._current_modality_step += 1  
                    
                    if remaining_blocks > 0:
                        # Continue with the current modality, start the next block (using the next group of 8 prompts from the same cycle)
                        if IS_MAIN_PROCESS:
                            print(f"\n[Block] Starting new block for current modality {current_freeze}")
                            print(f"[Block] Cycle {self._current_cycle_idx} | Modality Step {self._current_modality_step}/{self._steps_per_modality}")
                        
                        self._sample_videos(
                            progress=progress,
                            skip_reward_processing=False,
                            freeze_modality_override=current_freeze,
                            force_refresh_cache=True
                        )
                        self._accelerator.wait_for_everyone()
                        data_iter = iter(self._dataloader)
                        block_steps_left = steps_per_block
                    else:
                        # All 4 steps of the current modality are completed
                        if current_freeze == 'audio':
                            # ========== Video modality completed, switching to audio modality (using prompts from the same cycle) ==========
                            new_freeze = 'video'
                            if IS_MAIN_PROCESS:
                                print(f"\n{'='*60}")
                                print(f"[CYCLE] ✅ Video modality completed cycle {self._current_cycle_idx}")
                                print(f"[CYCLE] Switching to Audio modality, using SAME prompts (cycle {self._current_cycle_idx})")
                                print(f"{'='*60}\n")
                            
                            # Update modality state
                            self._current_freeze_modality = new_freeze
                            self._config.validation.freeze_modality = new_freeze
                            current_freeze = new_freeze
                            blocks, steps_per_block = BLOCK_CONFIG[current_freeze]
                            remaining_blocks = blocks
                            block_steps_left = steps_per_block
                            self._current_modality_step = 0  # Reset step count within the modality
                            
                            # Reinitialize optimizer and sample
                            self._reinit_optimizer_for_modality(new_freeze)
                
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize(device=device)
                            gc.collect()
                            if dist.is_initialized():
                                dist.barrier()
                            self._sample_videos(
                                progress=progress,
                                skip_reward_processing=False,
                                freeze_modality_override=new_freeze,
                                force_refresh_cache=True
                            )
                            self._accelerator.wait_for_everyone()
                            data_iter = iter(self._dataloader)
                        else:
                            # ========== Audio modality completed, switching back to video modality (using prompts from the next cycle) ==========
                            new_freeze = 'audio'
                            self._current_cycle_idx += 1  # Switch to the next cycle
                            if IS_MAIN_PROCESS:
                                print(f"\n{'='*60}")
                                print(f"[CYCLE] ✅ Audio modality completed cycle {self._current_cycle_idx - 1}")
                                print(f"[CYCLE] Full cycle completed! Switching to Video modality, next cycle {self._current_cycle_idx}")
                                print(f"[CYCLE] Next prompts: {self._current_cycle_idx * self._prompts_per_cycle} - {(self._current_cycle_idx + 1) * self._prompts_per_cycle - 1}")
                                print(f"{'='*60}\n")
                            
                            # Update modality state
                            self._current_freeze_modality = new_freeze
                            self._config.validation.freeze_modality = new_freeze
                            current_freeze = new_freeze
                            blocks, steps_per_block = BLOCK_CONFIG[current_freeze]
                            remaining_blocks = blocks
                            block_steps_left = steps_per_block
                            self._current_modality_step = 0  # Reset step count within the modality
                            
                            # Reinitialize optimizer and sample
                            self._reinit_optimizer_for_modality(new_freeze)
                           
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize(device=device)
                            gc.collect()
                            if dist.is_initialized():
                                dist.barrier()
                            self._sample_videos(
                                progress=progress,
                                skip_reward_processing=False,
                                freeze_modality_override=new_freeze,
                                force_refresh_cache=True
                            )
                            self._accelerator.wait_for_everyone()
                            data_iter = iter(self._dataloader)


                # checkpoint
                if self._global_step > 0 and self._global_step % self._config.checkpoints.interval == 0:
                    self._save_checkpoint()

                if self._global_step > 0 and self._global_step % 4 == 0:
                    self._cleanup_samples_folder(keep_steps=2)

                if step_callback:
                    step_callback(self._global_step, self._config.optimization.steps, sampled_videos_paths)

                if grpo_step % MEMORY_CHECK_INTERVAL == 0:
                    current_mem = get_gpu_memory_gb(device)
                    peak_mem_during_training = max(peak_mem_during_training, current_mem)
                    if IS_MAIN_PROCESS:
                        logger.info(f"Step {self._global_step} GPU memory: {current_mem:.2f} GB (peak: {peak_mem_during_training:.2f} GB)")

                if grpo_step % 1 == 0:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    if dist.is_initialized():
                        torch.cuda.ipc_collect()
                    gc.collect()

           
            train_end_time = time.time()
            end_mem = get_gpu_memory_gb(device)
            peak_mem = max(start_mem, end_mem, peak_mem_during_training)

            total_time_seconds = train_end_time - train_start_time
            steps_per_second = total_grpo_steps / total_time_seconds if total_time_seconds > 0 else 0
            samples_per_second = steps_per_second * self._accelerator.num_processes * self._config.optimization.batch_size

            stats = TrainingStats(
                total_time_seconds=total_time_seconds,
                steps_per_second=steps_per_second,
                samples_per_second=samples_per_second,
                peak_gpu_memory_gb=peak_mem,
                num_processes=self._accelerator.num_processes,
                global_batch_size=self._config.optimization.batch_size * self._accelerator.num_processes,
            )

            saved_path = self._save_checkpoint()

            if IS_MAIN_PROCESS:
                self._log_training_stats(stats)
                if self._config.hub.push_to_hub:
                    push_to_hub(saved_path, sampled_videos_paths, self._config)
                if self._wandb_run is not None:
                    wandb.log({
                        "stats/total_time_minutes": stats.total_time_seconds / 60,
                        "stats/steps_per_second": stats.steps_per_second,
                        "stats/samples_per_second": stats.samples_per_second,
                        "stats/peak_gpu_memory_gb": stats.peak_gpu_memory_gb,
                    }, step=self._global_step)
                    self._wandb_run.finish()

            self._accelerator.wait_for_everyone()
            self._accelerator.end_training()

            return saved_path, stats

    def _compute_new_log_prob_and_velocity(
        self,
        x_t: Tensor,                
        x0_pred: Tensor,            
        sigma: Tensor,              
        dt_abs: Tensor,             
        sigma_t_eff: Tensor,        
        noise: Tensor,              
        next_latent: Tensor,        
    ) -> tuple[Tensor, Tensor]:
        
        train_len = x0_pred.shape[1]
        cache_len = x_t.shape[1]
        min_len = min(train_len, cache_len)

      
        x_t = x_t[:, :min_len, :]
        x0_pred = x0_pred[:, :min_len, :]
        noise = noise[:, :min_len, :]
        next_latent = next_latent[:, :min_len, :]

        t = sigma.clamp(min=1e-5)
        v_new = (x_t - x0_pred) / t

        inner = x_t + (1 - t) * v_new
        correction_coeff = (sigma_t_eff ** 2) / (2 * t)
        score_correction = - correction_coeff * inner * dt_abs

        prev_sample_mean = x_t + v_new * (-dt_abs) + score_correction
        variance = (sigma_t_eff * torch.sqrt(dt_abs)) ** 2

        log_prob = -((next_latent - prev_sample_mean) ** 2) / (2 * variance) \
                - 0.5 * torch.log(2 * torch.pi * variance)
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return log_prob, v_new

    def _compute_lambda_policy(self, t: Tensor, dt_abs: Tensor) -> Tensor:
        return torch.sqrt(t / (dt_abs * (1 - t).clamp(min=1e-5)))

    def _compute_lambda_kl(self, t: Tensor, dt_abs: Tensor) -> Tensor:
        return t / (dt_abs * (1 - t).clamp(min=1e-5))

    def _load_advantages(self):
        """Load the most recently saved advantage file."""
        adv_dir = Path(self._config.output_dir) / "advantages"
        if not adv_dir.exists():
            self._video_advantages = None
            return
       
        adv_files = sorted(adv_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split('_')[1]))
        if not adv_files:
            self._video_advantages = None
            return
        latest_file = adv_files[-1]
        data = torch.load(latest_file, map_location="cpu")
        self._video_advantages = data.get('video_advantage', None)
        if self._accelerator.is_main_process:
            logger.info(f"📥 Loading advantage values: {latest_file}")

    def _training_step(self, batch: dict[str, dict[str, Tensor]]) -> tuple[Tensor, float, float]:
        import torch.distributed as dist
        from pathlib import Path
        from ltx_core.model.transformer.modality import Modality
        import random
        import numpy as np

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        device = self._accelerator.device

        
        freeze_modality = self._current_freeze_modality
        if freeze_modality == "audio":
            beta_kl = 0.01 # KL coefficient for training video. Note that only the parameters here are effective.
        else:
            beta_kl = 0.002 # KL coefficient for training audio. Note that only the parameters here are effective.

        clip_range = getattr(self._config.optimization, 'clip_range', 0.2)
        samples_per_step = getattr(self._config.optimization, 'grpo_samples_per_step', 1)
        timesteps_per_sample = getattr(self._config.optimization, 'grpo_timesteps_per_sample', 4)
        if freeze_modality == "audio":
            a_noise = 0.02
        else:
            a_noise = 0.8
       
        adv_dir = Path(self._config.output_dir) / "advantages"
        if hasattr(self, '_last_advantage_step') and self._last_advantage_step is not None:
            adv_path = adv_dir / f"step_{self._last_advantage_step:06d}_rank{rank}.pt"
            if adv_path.exists():
                if rank == 0:
                    print(f"[DEBUG] Using matched advantage file: {adv_path}")
                adv_data = torch.load(adv_path, map_location='cpu')
            else:
                if rank == 0:
                    print(f"[GRPO] ⚠️ Matched advantage file not found: {adv_path}, falling back to latest file")
                adv_files = sorted(adv_dir.glob(f"step_*_rank{rank}.pt"))
                if not adv_files:
                    if rank == 0:
                        print(f"[GRPO] ❌ No advantage file found for rank {rank}!")
                   
                    return torch.tensor(0.0, device=device), 0.0, 0.0
                latest_adv = adv_files[-1]
                adv_data = torch.load(latest_adv, map_location='cpu')
        else:
            adv_files = sorted(adv_dir.glob(f"step_*_rank{rank}.pt"))
            if not adv_files:
                if rank == 0:
                    print(f"[GRPO] ❌ No advantage file found for rank {rank}!")
                
                return torch.tensor(0.0, device=device), 0.0, 0.0
            latest_adv = adv_files[-1]
            if rank == 0:
                print(f"[DEBUG] Loading latest advantage file: {latest_adv}")
            adv_data = torch.load(latest_adv, map_location='cpu')

        video_advantages = adv_data.get('video_advantage')
        audio_advantages = adv_data.get('audio_advantage')

        if rank == 0:
            print(f"[DEBUG] video_advantages: {video_advantages}")
            print(f"[DEBUG] audio_advantages: {audio_advantages}")
            if video_advantages is not None:
                print(f"[DEBUG] video_advantages has nan: {torch.isnan(video_advantages).any().item()}")
            if audio_advantages is not None:
                print(f"[DEBUG] audio_advantages has nan: {torch.isnan(audio_advantages).any().item()}")

        
        traj_dir = Path(self._config.output_dir).parent / "sample_trajectory_train"
        traj_files = sorted(traj_dir.glob(f"traj_rank{rank}_sample*.pt"))
        if not traj_files:
            if rank == 0:
                print(f"[GRPO] ❌ No trajectory files found for rank {rank}!")
            
            return torch.tensor(0.0, device=device), 0.0, 0.0

        if len(traj_files) > samples_per_step:
            sampled_traj_files = random.sample(traj_files, samples_per_step)
        else:
            sampled_traj_files = traj_files

        if rank == 0:
            print(f"[DEBUG] All traj files: {[f.name for f in traj_files]}")
            print(f"[DEBUG] Sampled traj files: {[f.name for f in sampled_traj_files]}")

        accum_steps = self._config.optimization.gradient_accumulation_steps
        step_count = 0
        total_loss = 0.0
        total_steps = 0
        
        total_policy_loss = 0.0
        total_kl_loss = 0.0

       
        train_video = (freeze_modality == "audio")
        train_audio = (freeze_modality == "video")
        if not train_video and not train_audio:
            train_video = True
        modality_name = "Video" if train_video else "Audio"

       
        local_has_nan = False
        if train_video:
            if video_advantages is None or torch.isnan(video_advantages).any():
                local_has_nan = True
        else:
            if audio_advantages is None or torch.isnan(audio_advantages).any():
                local_has_nan = True

        if dist.is_initialized():
            local_nan_tensor = torch.tensor([local_has_nan], dtype=torch.int, device=device)
            global_nan_tensor = local_nan_tensor.clone()
            dist.all_reduce(global_nan_tensor, op=dist.ReduceOp.MAX)
            global_has_nan = bool(global_nan_tensor.item())
        else:
            global_has_nan = local_has_nan

        if global_has_nan:
            if rank == 0:
                print(f"[WARN] NaN advantage detected across ranks, skipping this training step")
            self._skip_modality = True          
     
            return torch.tensor(0.0, device=device), 0.0, 0.0


    
        for sample_idx, traj_file in enumerate(sampled_traj_files):
            try:
                local_sample_idx = int(traj_file.stem.split('_sample')[-1])
            except:
                local_sample_idx = sample_idx

            if rank == 0:
                print(f"[DEBUG] Processing {traj_file.name}, local_sample_idx={local_sample_idx}")

            if train_video:
                if video_advantages is None:
                    if rank == 0:
                        print(f"[DEBUG] video_advantages is None, skip sample")
                    continue
                if local_sample_idx >= len(video_advantages):
                    if rank == 0:
                        print(f"[DEBUG] local_sample_idx {local_sample_idx} >= len(video_advantages)={len(video_advantages)}, skip")
                    continue
                advantage = video_advantages[local_sample_idx].to(device, dtype=torch.float32)
            else:
                if audio_advantages is None:
                    if rank == 0:
                        print(f"[DEBUG] audio_advantages is None, skip sample")
                    continue
                if local_sample_idx >= len(audio_advantages):
                    if rank == 0:
                        print(f"[DEBUG] local_sample_idx {local_sample_idx} >= len(audio_advantages)={len(audio_advantages)}, skip")
                    continue
                advantage = audio_advantages[local_sample_idx].to(device, dtype=torch.float32)

            if rank == 0:
                print(f"[DEBUG] advantage value: {advantage.item()} (is_nan: {torch.isnan(advantage).item()})")
            if torch.isnan(advantage):
                if rank == 0:
                    print(f"[WARN] advantage is nan for sample {local_sample_idx}, skipping this sample")
                continue

            traj = torch.load(traj_file, map_location='cpu')
            num_steps = len(traj['video_sigma'])
            if num_steps > timesteps_per_sample:
                step_indices = sorted(random.sample(range(num_steps), timesteps_per_sample))
            else:
                step_indices = list(range(num_steps))

            if rank == 0:
                print(f"[DEBUG] Using {len(step_indices)} timesteps: {step_indices}")

            for step_idx in step_indices:
                
                if train_video:
                    x_t = traj['video_latent'][step_idx].to(device, dtype=torch.bfloat16)
                    next_latent = traj['next_video_latent'][step_idx].to(device, dtype=torch.float32)
                    old_log_prob = traj['video_log_prob'][step_idx].to(device, dtype=torch.float32)
                    t_val = traj['video_sigma'][step_idx].to(device, dtype=torch.float32)
                    dt_abs = traj['video_dt_abs'][step_idx].to(device, dtype=torch.float32)
                    sigma_t_eff = traj['video_sigma_t_eff'][step_idx].to(device, dtype=torch.float32)
                    noise = traj['video_noise'][step_idx].to(device, dtype=torch.float32)
                    context = traj['video_context'][step_idx].to(device, dtype=torch.bfloat16)
                    positions = traj['video_positions'][step_idx].to(device, dtype=torch.bfloat16)
                else:
                    x_t = traj['audio_latent'][step_idx].to(device, dtype=torch.bfloat16)
                    next_latent = traj['next_audio_latent'][step_idx].to(device, dtype=torch.float32)
                    old_log_prob = traj['audio_log_prob'][step_idx].to(device, dtype=torch.float32)
                    t_val = traj['audio_sigma'][step_idx].to(device, dtype=torch.float32)
                    dt_abs = traj['audio_dt_abs'][step_idx].to(device, dtype=torch.float32)
                    sigma_t_eff = traj['audio_sigma_t_eff'][step_idx].to(device, dtype=torch.float32)
                    noise = traj['audio_noise'][step_idx].to(device, dtype=torch.float32)
                    context = traj['audio_context'][step_idx].to(device, dtype=torch.bfloat16)
                    positions = traj['audio_positions'][step_idx].to(device, dtype=torch.bfloat16)

                modality = Modality(
                    enabled=True,
                    latent=x_t,
                    sigma=t_val.expand(1),
                    timesteps=t_val.view(1, 1, 1).expand(-1, x_t.shape[1], -1),
                    positions=positions,
                    context=context,
                    context_mask=None,
                )
                if train_video:
                    video_modality, audio_modality = modality, None
                else:
                    video_modality, audio_modality = None, modality

                with self._accelerator.autocast():
                    vx, ax = self._transformer(video=video_modality, audio=audio_modality, perturbations=None)
                v_cur = vx if train_video else ax

                if beta_kl > 0:
                    with torch.no_grad(), self._accelerator.autocast():
                        vx_ref, ax_ref = self._ref_transformer(video=video_modality, audio=audio_modality, perturbations=None)
                    v_ref = vx_ref if train_video else ax_ref
                else:
                    v_ref = None

                x0_new = x_t.to(torch.float32) - t_val.to(torch.float32) * v_cur.to(torch.float32)
                new_log_prob, v_new = self._compute_new_log_prob_and_velocity(
                    x_t.to(torch.float32), x0_new, t_val, dt_abs, sigma_t_eff, noise, next_latent
                )

                ratio = torch.exp(new_log_prob - old_log_prob)

                # ==============================================
                t_val_clamped = t_val.clamp(min=1e-5, max=0.999)
                t = t_val_clamped       
                dt = dt_abs             

                denominator_policy = dt * (1.0 - t) + 1e-10
                lambda_policy = torch.sqrt(t / denominator_policy)

            
                unclipped = -advantage * ratio * lambda_policy
                clipped = -advantage * torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * lambda_policy

                policy_loss = torch.mean(torch.maximum(unclipped, clipped))
                step_loss = policy_loss

               
                kl_loss_val = 0.0
                if beta_kl > 0 and v_ref is not None:
                    
                    sigma_t = a_noise * torch.sqrt(t_val_clamped / (1 - t_val_clamped))
                    
                  
                    coef_1 = sigma_t * (1 - t_val_clamped) / (2 * t_val_clamped + 1e-8)
                    coef_2 = 1 / (sigma_t + 1e-8)
                    lambda_kl = (dt / 2) * ((coef_1 + coef_2) ** 2)

                    denominator_kl = dt * (1.0 - t) + 1e-10
                    if freeze_modality == "audio":
                        lambda_kl_reweight = t / denominator_kl
                    else:
                        lambda_kl_reweight=1
              
                    mse_v = torch.nn.functional.mse_loss(v_cur.to(torch.float32), v_ref.to(torch.float32))
                    
                  
                    raw_kl_div = lambda_kl * mse_v
                    reweighted_kl_div = lambda_kl_reweight * raw_kl_div

                  
                    kl_loss = beta_kl * reweighted_kl_div

                    step_loss = step_loss + kl_loss
                    kl_loss_val = kl_loss.item()

               
                if rank == 0 and step_count % 5 == 0:
                    v_cur_stats = f"min={v_cur.min().item():.4f} max={v_cur.max().item():.4f} mean={v_cur.mean().item():.4f} std={v_cur.std().item():.4f}"
                    if v_ref is not None:
                        v_ref_stats = f"min={v_ref.min().item():.4f} max={v_ref.max().item():.4f} mean={v_ref.mean().item():.4f} std={v_ref.std().item():.4f}"
                    else:
                        v_ref_stats = "N/A"
                    print(f"[DEBUG] v_cur stats: {v_cur_stats}")
                    print(f"[DEBUG] v_ref stats: {v_ref_stats}")

                    ratio_mean = ratio.mean().item()
                    ratio_std = ratio.std().item() if ratio.numel() > 1 else 0.0
                    ratio_min = ratio.min().item()
                    ratio_max = ratio.max().item()
                    unclipped_mean = unclipped.mean().item()
                    clipped_mean = clipped.mean().item()
                    clip_frac_low = ((ratio < 1 - clip_range).float().mean().item())
                    clip_frac_high = ((ratio > 1 + clip_range).float().mean().item())
                    clip_frac_total = clip_frac_low + clip_frac_high
                    print(f"[Policy-{modality_name}] t={t_val.item():.4f}, adv={advantage.item():.4f}, "
                        f"ratio: {ratio_mean:.4f}±{ratio_std:.4f} [{ratio_min:.4f}, {ratio_max:.4f}], "
                        f"unclipped: {unclipped_mean:.4f}, clipped: {clipped_mean:.4f}, "
                        f"clip_frac: {clip_frac_total:.2%}, policy_loss: {policy_loss.item():.6f}")
                    if beta_kl > 0 and v_ref is not None:
                        sigma_t_val = sigma_t.item() if hasattr(sigma_t, 'item') else sigma_t.mean().item()
                        print(f"[KL-{modality_name}] t={t_val.item():.4f}, σ_t={sigma_t_val:.4f}, "
                            f"λ_kl={lambda_kl.mean().item():.4f}, MSE={mse_v.item():.6f}, "
                            f"β*KL={kl_loss.item():.6f}")

                original_loss = step_loss.item()
                step_loss = step_loss / accum_steps
                self._accelerator.backward(step_loss)

                total_loss += original_loss
           
                total_policy_loss += policy_loss.item()
                total_kl_loss += kl_loss_val
                total_steps += 1
                step_count += 1

                if step_count % accum_steps == 0:
                    if self._accelerator.sync_gradients:
                        self._accelerator.clip_grad_norm_(self._trainable_params, self._config.optimization.max_grad_norm)
                    self._optimizer.step()
                    self._optimizer.zero_grad()

           
                del x_t, next_latent, old_log_prob, t_val, dt_abs, sigma_t_eff, noise
                del modality, v_cur, x0_new, new_log_prob, v_new
                if v_ref is not None:
                    del v_ref
                if train_video:
                    del vx
                else:
                    del ax
                if beta_kl > 0:
                    del vx_ref
                    if ax_ref is not None:
                        del ax_ref


            torch.cuda.empty_cache()

     
        if step_count % accum_steps != 0:
            if self._accelerator.sync_gradients:
                self._accelerator.clip_grad_norm_(self._trainable_params, self._config.optimization.max_grad_norm)
            self._optimizer.step()
            self._optimizer.zero_grad()

        if rank == 0:
            print(f"[DEBUG] total_steps processed in this training_step: {total_steps}")

        if total_steps == 0:
            if rank == 0:
                print("[WARN] No valid steps were processed, returning loss=0")
         
            return torch.tensor(0.0, device=device), 0.0, 0.0

     
        local_avg_loss = total_loss / total_steps
        local_avg_policy_loss = total_policy_loss / total_steps
        local_avg_kl_loss = total_kl_loss / total_steps

    
        if dist.is_initialized() and world_size > 1:
            loss_tensor = torch.tensor([local_avg_loss], device=device)
            policy_loss_tensor = torch.tensor([local_avg_policy_loss], device=device)
            kl_loss_tensor = torch.tensor([local_avg_kl_loss], device=device)
            
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(policy_loss_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(kl_loss_tensor, op=dist.ReduceOp.AVG)
            
            global_avg_loss = loss_tensor.item()
            global_avg_policy_loss = policy_loss_tensor.item()
            global_avg_kl_loss = kl_loss_tensor.item()
        else:
            global_avg_loss = local_avg_loss
            global_avg_policy_loss = local_avg_policy_loss
            global_avg_kl_loss = local_avg_kl_loss

        if rank == 0:
            print(f"[GRPO-{modality_name}] Step {self._global_step} | Global Loss: {global_avg_loss:.6f} | Policy Loss: {global_avg_policy_loss:.6f} | KL Loss: {global_avg_kl_loss:.6f}")

        del adv_data, video_advantages, audio_advantages
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

      
        return torch.tensor(global_avg_loss, device=device), global_avg_policy_loss, global_avg_kl_loss

    def _should_compute_grpo_loss(self) -> bool:
     

        grpo_enabled = getattr(self._config.optimization, 'grpo_enabled', False)
        if not grpo_enabled:
            return False
        

        grpo_interval = getattr(self._config.optimization, 'grpo_interval', 5)
        if self._global_step % grpo_interval != 0:
            return False
        

        import torch.distributed as dist
        from pathlib import Path
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        traj_dir = Path(self._config.output_dir).parent / "sample_trajectory"
        traj_files = list(traj_dir.glob(f"traj_rank{rank}_sample*.pt"))
        has_files = len(traj_files) > 0
        
  
        if dist.is_initialized():
         
            has_files_tensor = torch.tensor([has_files], dtype=torch.int, device=self.accelerator.device)
            all_has_files = [torch.zeros(1, dtype=torch.int, device=self.accelerator.device) for _ in range(dist.get_world_size())]
            dist.all_gather(all_has_files, has_files_tensor)
            

            all_have_files = all(h.item() == 1 for h in all_has_files)
            
            if rank == 0:
                print(f"[GRPO] File check: rank has files: {[h.item() for h in all_has_files]}")
            
            if not all_have_files:
                if rank == 0:
                    print(f"[GRPO] ⚠️ Not all ranks have trajectory files, skipping GRPO this step")
                return False
        
        return has_files

    def _contains_chinese(self, text: str) -> bool:
        import re
        return bool(re.search(r'[\u3400-\u4DBF\u4E00-\u9FFF]', text))

    @free_gpu_memory_context(after=True)
    def _load_text_encoder_and_cache_embeddings(self) -> list[CachedPromptEmbeddings] | None:
        
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)

        
        self._text_encoder = load_text_encoder(
            gemma_model_path=self._config.model.text_encoder_path,
            device="cuda",
            dtype=torch.bfloat16,
            load_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
        )

        
        logger.debug("Loading embeddings processor...")
        self._embeddings_processor = load_embeddings_processor(
            checkpoint_path=self._config.model.model_path,
            device="cuda",
            dtype=torch.bfloat16,
        )

        
        import json
        import random
        from collections import OrderedDict
        PROMPT_JSON_PATH = str(Path(self._config.output_dir).parent.parent / "Dataset" / "dataset.json")
        
        with open(PROMPT_JSON_PATH, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)
        
        
        self._all_prompts = []
        total_loaded = 0
        for item in prompt_data:
            # if item.get("t") == 2:
            #     continue
            if 'prompts' in item and isinstance(item['prompts'], list):
                total_loaded += len(item['prompts'])
                for prompt in item['prompts']:
                    if not self._contains_chinese(prompt):
                        self._all_prompts.append(prompt.strip())
        
        if IS_MAIN_PROCESS:
            filtered_count = total_loaded - len(self._all_prompts)
            print(f"[INIT] Loaded {len(self._all_prompts)} English prompts from {PROMPT_JSON_PATH}")
            print(f"[INIT] Filtered out {filtered_count} Chinese prompts")

      
        random.seed(self._config.seed)
        random.shuffle(self._all_prompts)
        if IS_MAIN_PROCESS:
            print(f"[INIT] ✅ All prompts shuffled globally with seed {self._config.seed}")
            print(f"[INIT] Total unique prompts: {len(self._all_prompts)}")

        
        self._prompts_per_step = 8          # Use 8 prompts per step
        self._steps_per_modality = 4        # Train 4 steps per modality
        self._prompts_per_cycle = self._prompts_per_step * self._steps_per_modality  # 32 prompts per cycle
        if IS_MAIN_PROCESS:
            print(f"[INIT] Training cycle config:")
            print(f"[INIT]   Prompts per step: {self._prompts_per_step}")
            print(f"[INIT]   Steps per modality: {self._steps_per_modality}")
            print(f"[INIT]   Prompts per full cycle (video+audio): {self._prompts_per_cycle}")

       
        self._embedding_cache = OrderedDict()
        self._max_cache_size = 24
        
   
        self._negative_prompt = self._config.validation.negative_prompt or ""
        with torch.inference_mode():
            neg_hs, neg_mask = self._text_encoder.encode(self._negative_prompt)
            self._neg_out = self._embeddings_processor.process_hidden_states(neg_hs, neg_mask)
        
        logger.debug("Text encoder and embeddings processor loaded. No pre-computation done.")
        
      
        return []

    def _get_prompt_embedding(self, prompt: str) -> CachedPromptEmbeddings:
      
     
        if prompt in self._embedding_cache:
          
            self._embedding_cache.move_to_end(prompt)
            return self._embedding_cache[prompt]
        
        
        with torch.inference_mode():
            pos_hs, pos_mask = self._text_encoder.encode(prompt)
            pos_out = self._embeddings_processor.process_hidden_states(pos_hs, pos_mask)
        
        cached_embedding = CachedPromptEmbeddings(
            video_context_positive=pos_out.video_encoding.cpu(),
            audio_context_positive=pos_out.audio_encoding.cpu(),
            video_context_negative=self._neg_out.video_encoding.cpu(),
            audio_context_negative=(
                self._neg_out.audio_encoding.cpu() if self._neg_out.audio_encoding is not None else None
            ),
        )
        
       
        self._embedding_cache[prompt] = cached_embedding
        
        
        if len(self._embedding_cache) > self._max_cache_size:
            self._embedding_cache.popitem(last=False)
        
        return cached_embedding

    def _load_models(self) -> None:
        """Load the LTX-2 model components."""
        # Load audio components if:
        # 1. Training strategy requires audio (training the audio branch), OR
        # 2. Validation is configured to generate audio (even if not training audio)
        load_audio = self._training_strategy.requires_audio or self._config.validation.generate_audio

        # Check if we need VAE encoder (for image or reference video conditioning)
        need_vae_encoder = (
            self._config.validation.images is not None or self._config.validation.reference_videos is not None
        )

        # Load all model components (except text encoder - already handled)
        components = load_ltx_model(
            checkpoint_path=self._config.model.model_path,
            device="cpu",
            dtype=torch.bfloat16,
            with_video_vae_encoder=need_vae_encoder,  # Needed for image conditioning
            with_video_vae_decoder=True,  # Needed for validation sampling
            with_audio_vae_decoder=load_audio,
            with_vocoder=load_audio,
            with_text_encoder=False,  # Text encoder handled separately
        )

        # Extract components
        self._transformer = components.transformer
        self._vae_decoder = components.video_vae_decoder.to(dtype=torch.bfloat16)
        self._vae_encoder = components.video_vae_encoder
        if self._vae_encoder is not None:
            self._vae_encoder = self._vae_encoder.to(dtype=torch.bfloat16)
        self._scheduler = components.scheduler
        self._audio_vae = components.audio_vae_decoder
        self._vocoder = components.vocoder
        # Note: self._embeddings_processor was set in _load_text_encoder_and_cache_embeddings

        # Determine initial dtype based on training mode.
        # Note: For FSDP + LoRA, we'll cast to FP32 later in _prepare_models_for_training()
        # after the accelerator is set up, and we can detect FSDP.
        transformer_dtype = torch.bfloat16 if self._config.model.training_mode == "lora" else torch.float32
        self._transformer = self._transformer.to(dtype=transformer_dtype)

        if self._config.acceleration.quantization is not None:
            if self._config.model.training_mode == "full":
                raise ValueError("Quantization is not supported in full training mode.")

            logger.info(f'Quantizing model with "{self._config.acceleration.quantization}". This may take a while...')
            self._transformer = quantize_model(
                self._transformer,
                precision=self._config.acceleration.quantization,
            )

        # Freeze all models. We later unfreeze the transformer based on training mode.
        # Note: embedding_connectors are already frozen (they come from the frozen text encoder)
        self._vae_decoder.requires_grad_(False)
        if self._vae_encoder is not None:
            self._vae_encoder.requires_grad_(False)
        self._transformer.requires_grad_(False)
        if self._audio_vae is not None:
            self._audio_vae.requires_grad_(False)
        if self._vocoder is not None:
            self._vocoder.requires_grad_(False)

        if self._config.model.training_mode == "lora":
            self._setup_lora()
            if IS_MAIN_PROCESS:
                print(f"[INIT] LoRA adapter added once in _load_models, rank={self._accelerator.process_index if hasattr(self, '_accelerator') else 0}")

    def _collect_trainable_params(self, train_mode: str = "audio") -> None:
        
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
   
        self._transformer.requires_grad_(True)
        if dist.is_initialized():
            dist.barrier()

        if dist.is_initialized():
            dist.barrier()

        trainable_names = set()
        
        for name, param in self._transformer.named_parameters():
            # ---------------- LoRA ----------------
            if self._config.model.training_mode == "lora":
                
                if "lora_" not in name:
                    continue
                
           
                if train_mode == "video":
                    
                    if any(key in name for key in ["audio_attn1", "audio_attn2", "video_to_audio_attn"]):
                        continue
                    if any(key in name for key in [
                        "audio_patchify_proj", "audio_adaln_single", 
                        "audio_scale_shift_table", "audio_proj_out",
                        "audio_prompt_adaln_single"
                    ]):
                        continue

                    if any(key in name for key in [
                        "av_ca_v2a_gate", "av_ca_audio_scale_shift"
                    ]):
                        continue

                elif train_mode == "audio":
                    
                    if any(key in name for key in [".attn1.", ".attn2.", "audio_to_video_attn"]):
                        continue
                   
                    if any(key in name for key in [
                        "patchify_proj", "adaln_single", 
                        "scale_shift_table", "proj_out",
                        "prompt_adaln_single"
                    ]) and "audio_" not in name and "av_ca_" not in name:
                        continue
       
                    if any(key in name for key in [
                        "av_ca_a2v_gate", "av_ca_video_scale_shift"
                    ]):
                        continue
            
          
            elif self._config.model.training_mode == "full":
                if train_mode == "audio":
                    if "audio_to_video_attn" in name:
                        continue
                elif train_mode == "video":
                    if "video_to_audio_attn" in name:
                        continue

             
                if train_mode == "audio":
                    keep = (
                        name.startswith("_fsdp_wrapped_module.audio_") or
                        ("transformer_blocks" in name and "audio_" in name) or
                        "video_to_audio_attn" in name or
                        "av_ca_v2a_gate" in name or
                        "av_ca_audio_scale_shift" in name
                    )
                    if not keep:
                        continue
                elif train_mode == "video":
                    keep = (
                        (name.startswith("_fsdp_wrapped_module.") and "audio_" not in name and "av_ca_" not in name) or
                        ("transformer_blocks" in name and "audio_" not in name) or
                        "audio_to_video_attn" in name or
                        "av_ca_a2v_gate" in name or
                        "av_ca_video_scale_shift" in name
                    )
                    if not keep:
                        continue
            
            else:
                raise ValueError(f"Unknown training mode: {self._config.model.training_mode}")
            
            trainable_names.add(name)
        
       
        self._trainable_params = [
            param for name, param in self._transformer.named_parameters()
            if name in trainable_names
        ]
        


        if rank == 0:
            print("\n" + "="*80)
            print("✅ All GPU freezing logic executed successfully!")
            print(f"✅ Training mode:: {train_mode} | {self._config.model.training_mode}")
            print(f"✅ Total GPU count: {world_size}")

        

    def _init_timestep_sampler(self) -> None:
        """Initialize the timestep sampler based on the config."""
        sampler_cls = SAMPLERS[self._config.flow_matching.timestep_sampling_mode]
        self._timestep_sampler = sampler_cls(**self._config.flow_matching.timestep_sampling_params)

    def _setup_lora(self) -> None:
        """Configure LoRA adapters for the transformer. Only called in LoRA training mode."""
        logger.debug(f"Adding LoRA adapter with rank {self._config.lora.rank}")
        lora_config = LoraConfig(
            r=self._config.lora.rank,
            lora_alpha=self._config.lora.alpha,
            target_modules=self._config.lora.target_modules,
            lora_dropout=self._config.lora.dropout,
            init_lora_weights=True,
        )
        # Wrap the transformer with PEFT to add LoRA layers
        # noinspection PyTypeChecker
        self._transformer = get_peft_model(self._transformer, lora_config)

    def _load_checkpoint(self) -> None:
        """Load checkpoint if specified in config."""
        if not self._config.model.load_checkpoint:
            return

        checkpoint_path = self._find_checkpoint(self._config.model.load_checkpoint)
        if not checkpoint_path:
            logger.warning(f"⚠️ Could not find checkpoint at {self._config.model.load_checkpoint}")
            return

        logger.info(f"📥 Loading checkpoint from {checkpoint_path}")

        if self._config.model.training_mode == "full":
            self._load_full_checkpoint(checkpoint_path)
        else:  # LoRA mode
            self._load_lora_checkpoint(checkpoint_path)

    def _load_full_checkpoint(self, checkpoint_path: Path) -> None:
        """Load full model checkpoint."""
        state_dict = load_file(checkpoint_path)
        self._transformer.load_state_dict(state_dict, strict=True)

        logger.info("✅ Full model checkpoint loaded successfully")

    def _load_lora_checkpoint(self, checkpoint_path: Path) -> None:
        """Load LoRA checkpoint with DDP/FSDP compatibility."""
        state_dict = load_file(checkpoint_path)

        # Adjust layer names to match internal format.
        # (Weights are saved in ComfyUI-compatible format, with "diffusion_model." prefix)
        state_dict = {k.replace("diffusion_model.", "", 1): v for k, v in state_dict.items()}

        # Load LoRA weights and verify all weights were loaded
        base_model = self._transformer.get_base_model()
        set_peft_model_state_dict(base_model, state_dict)

        logger.info("✅ LoRA checkpoint loaded successfully")

    def _prepare_models_for_training(self) -> None:
        """Prepare models for training with Accelerate."""

        # For FSDP + LoRA: Cast entire model to FP32.
        # FSDP requires uniform dtype across all parameters in wrapped modules.
        # In LoRA mode, PEFT creates LoRA params in FP32 while base model is BF16.
        # We cast the base model to FP32 to match the LoRA params.
        if self._accelerator.distributed_type == DistributedType.FSDP and self._config.model.training_mode == "lora":
            logger.debug("FSDP: casting transformer to FP32 for uniform dtype")
            self._transformer = self._transformer.to(dtype=torch.float32)

        # Enable gradient checkpointing if requested
        # For PeftModel, we need to access the underlying base model
        transformer = (
            self._transformer.get_base_model() if hasattr(self._transformer, "get_base_model") else self._transformer
        )

        transformer.set_gradient_checkpointing(self._config.optimization.enable_gradient_checkpointing)

        # Keep frozen models on CPU for memory efficiency
        self._vae_decoder = self._vae_decoder.to("cpu")
        if self._vae_encoder is not None:
            self._vae_encoder = self._vae_encoder.to("cpu")

        # Embedding connectors are already on GPU from _load_text_encoder_and_cache_embeddings

        # noinspection PyTypeChecker
        self._transformer = self._accelerator.prepare(self._transformer)

        # Log GPU memory usage after model preparation
        vram_usage_gb = torch.cuda.memory_allocated() / 1024**3
        logger.debug(f"GPU memory usage after models preparation: {vram_usage_gb:.2f} GB")

    @staticmethod
    def _find_checkpoint(checkpoint_path: str | Path) -> Path | None:
        """Find the checkpoint file to load, handling both file and directory paths."""
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_file():
            if not checkpoint_path.suffix == ".safetensors":
                raise ValueError(f"Checkpoint file must have a .safetensors extension: {checkpoint_path}")
            return checkpoint_path

        if checkpoint_path.is_dir():
            # Look for checkpoint files in the directory
            checkpoints = list(checkpoint_path.rglob("*step_*.safetensors"))

            if not checkpoints:
                return None

            # Sort by step number and return the latest
            def _get_step_num(p: Path) -> int:
                try:
                    return int(p.stem.split("step_")[1])
                except (IndexError, ValueError):
                    return -1

            latest = max(checkpoints, key=_get_step_num)
            return latest

        else:
            raise ValueError(f"Invalid checkpoint path: {checkpoint_path}. Must be a file or directory.")

    def _init_dataloader(self) -> None:
        """Initialize the training data loader using the strategy's data sources."""
        if self._dataset is None:
            # Get data sources from the training strategy
            data_sources = self._training_strategy.get_data_sources()

            self._dataset = PrecomputedDataset(self._config.data.preprocessed_data_root, data_sources=data_sources)
            logger.debug(f"Loaded dataset with {len(self._dataset):,} samples from sources: {list(data_sources)}")

        num_workers = self._config.data.num_dataloader_workers
        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
        )

        self._dataloader = self._accelerator.prepare(dataloader)

    def _init_lora_weights(self) -> None:
        """Initialize LoRA weights for the transformer."""
        logger.debug("Initializing LoRA weights...")
        for _, module in self._transformer.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.reset_lora_parameters(adapter_name="default", init_lora_weights=True)

    def _init_optimizer(self, params_override=None) -> None:
        opt_cfg = self._config.optimization
        params = params_override if params_override is not None else self._trainable_params
        lr = opt_cfg.learning_rate

        if opt_cfg.optimizer_type == "adamw":
            optimizer = AdamW(params, lr=lr)
        elif opt_cfg.optimizer_type == "adamw8bit":
            from bitsandbytes.optim import AdamW8bit
            optimizer = AdamW8bit(params, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_cfg.optimizer_type}")

        self._optimizer = self._accelerator.prepare(optimizer)
        if IS_MAIN_PROCESS:
            print(f"[INIT] Optimizer created, lr={self._optimizer.param_groups[0]['lr']:.10f}")

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler | None:
        """Create learning rate scheduler based on config."""
        scheduler_type = self._config.optimization.scheduler_type
        steps = self._config.optimization.steps
        params = self._config.optimization.scheduler_params or {}

        if scheduler_type is None:
            return None

        if scheduler_type == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=params.pop("start_factor", 1.0),
                end_factor=params.pop("end_factor", 0.1),
                total_iters=steps,
                **params,
            )
        elif scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=steps,
                eta_min=params.pop("eta_min", 0),
                **params,
            )
        elif scheduler_type == "cosine_with_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=params.pop("T_0", steps // 4),  # First restart cycle length
                T_mult=params.pop("T_mult", 1),  # Multiplicative factor for cycle lengths
                eta_min=params.pop("eta_min", 5e-5),
                **params,
            )
        elif scheduler_type == "polynomial":
            scheduler = PolynomialLR(
                optimizer,
                total_iters=steps,
                power=params.pop("power", 1.0),
                **params,
            )
        elif scheduler_type == "step":
            scheduler = StepLR(
                optimizer,
                step_size=params.pop("step_size", steps // 2),
                gamma=params.pop("gamma", 0.1),
                **params,
            )
        elif scheduler_type == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return scheduler

    def _setup_accelerator(self) -> None:
        """Initialize the Accelerator with the appropriate settings."""

        # All distributed setup (DDP/FSDP, number of processes, etc.) is controlled by
        # the user's Accelerate configuration (accelerate config / accelerate launch).
        self._accelerator = Accelerator(
            mixed_precision=self._config.acceleration.mixed_precision_mode,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
        )

        if self._accelerator.num_processes > 1:
            logger.info(
                f"{self._accelerator.distributed_type.value} distributed training enabled "
                f"with {self._accelerator.num_processes} processes"
            )

            local_batch = self._config.optimization.batch_size
            global_batch = self._config.optimization.batch_size * self._accelerator.num_processes
            logger.info(f"Local batch size: {local_batch}, global batch size: {global_batch}")

        # Log torch.compile status from Accelerate's dynamo plugin
        is_compile_enabled = (
            hasattr(self._accelerator.state, "dynamo_plugin") and self._accelerator.state.dynamo_plugin.backend != "NO"
        )
        if is_compile_enabled:
            plugin = self._accelerator.state.dynamo_plugin
            logger.info(f"🔥 torch.compile enabled via Accelerate: backend={plugin.backend}, mode={plugin.mode}")

            if self._accelerator.distributed_type == DistributedType.FSDP:
                logger.warning(
                    "⚠️ FSDP + torch.compile is experimental and may hang on the first training iteration. "
                    "If this occurs, disable torch.compile by removing dynamo_config from your Accelerate config."
                )

        if self._accelerator.distributed_type == DistributedType.FSDP and self._config.acceleration.quantization:
            logger.warning(
                f"FSDP with quantization ({self._config.acceleration.quantization}) may have compatibility issues."
                "Monitor training stability and consider disabling quantization if issues arise."
            )

    def _cleanup_samples_folder(self, keep_steps: int = 10) -> None:
        """
        Clean up video/audio files in the {output_dir}/samples/ directory that are older than keep_steps steps before the current step.
        Executed by the main process only.
        """
        if not self._accelerator.is_main_process:
            return

        import re
        from pathlib import Path

        samples_dir = Path(self._config.output_dir) / "samples"
        if not samples_dir.exists():
            return

       
        step_pattern = re.compile(r'(?:step_|ref_step_)(\d+)')
        current_step = self._global_step
        threshold_step = current_step - keep_steps

        deleted = 0
        for f in samples_dir.iterdir():
            if not f.is_file():
                continue
    
            if f.suffix.lower() not in ('.mp4', '.png', '.jpg', '.jpeg', '.wav'):
                continue
            match = step_pattern.search(f.name)
            if match:
                step = int(match.group(1))
                if step < threshold_step:
                    f.unlink()
                    deleted += 1
                 
                    if f.suffix.lower() == '.mp4':
                        wav_path = f.with_suffix('.wav')
                        if wav_path.exists():
                            wav_path.unlink()
                            deleted += 1

        if deleted:
            print(f"[Cleanup] Deleted {deleted} old sample files (step < {threshold_step}) from {samples_dir}")


    # Note: Use @torch.no_grad() instead of @torch.inference_mode() to avoid FSDP inplace update errors after validation
    @torch.no_grad()
    @free_gpu_memory_context(after=True)
    def _sample_videos(self, progress: TrainingProgress | None = None, 
                        skip_reward_processing: bool = False,
                        freeze_modality_override: str | None = None,force_refresh_cache: bool = False) -> list[Path] | None:
        """Run validation by generating videos from validation prompts."""
        import torch.distributed as dist
        import os
        import numpy as np
        import json
        from pathlib import Path

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        if dist.is_initialized():
            dist.barrier()

        # ========== [Modification: Select prompts based on training cycle + step within modality] ==========
        # Use 8 prompts per step. Video and audio within the same cycle use exactly the same prompts.
        num_prompts_per_step = self._prompts_per_step
        total_prompts = len(self._all_prompts)

        # Core formula: current cycle start position + current step within modality × number of prompts per step
        cycle_start_idx = self._current_cycle_idx * self._prompts_per_cycle
        step_in_modality = self._current_modality_step
        start_idx = cycle_start_idx + step_in_modality * num_prompts_per_step

        # Cycle through prompts (start over from the beginning when all prompts are used up)
        start_idx = start_idx % total_prompts

        # Take out the 8 prompts for the current step (if not enough, supplement by cycling)
        current_prompts = self._all_prompts[start_idx:start_idx + num_prompts_per_step]
        if len(current_prompts) < num_prompts_per_step:
            current_prompts += self._all_prompts[:num_prompts_per_step - len(current_prompts)]

        # Each rank takes one from the current 8
        my_prompt = current_prompts[rank % num_prompts_per_step]

        if IS_MAIN_PROCESS:
            print(f"\n[SAMPLING] Step {self._global_step} | Cycle {self._current_cycle_idx} | Modality Step {step_in_modality}/{self._steps_per_modality}")
            print(f"[SAMPLING] Using prompts {start_idx}-{start_idx+num_prompts_per_step-1}")
            print(f"[SAMPLING] Rank {rank} using prompt: '{my_prompt[:50]}...'")
        # ========== End of prompt index logic modification ==========

        use_images = self._config.validation.images is not None
        use_reference_videos = self._config.validation.reference_videos is not None
        generate_audio = self._config.validation.generate_audio
        inference_steps = self._config.validation.inference_steps

        self._optimizer.zero_grad(set_to_none=True)
        free_gpu_memory()

        # Number of samples per GPU: read from config, default is 8 (must be >1 to compute within-group normalization)
        samples_per_gpu = 8
        total_samples = samples_per_gpu * world_size

       
        if progress is not None:
            sampling_ctx = progress.start_sampling(
                num_prompts=1,
                num_steps=inference_steps,
            )
        else:
            sampling_ctx = None

        sampler = ValidationSampler(
            transformer=self._transformer,
            vae_decoder=self._vae_decoder,
            vae_encoder=self._vae_encoder,
            text_encoder=None,
            audio_decoder=self._audio_vae if generate_audio else None,
            vocoder=self._vocoder if generate_audio else None,
            sampling_context=sampling_ctx,
        )

        output_dir = Path(self._config.output_dir) / "samples"
        output_dir.mkdir(exist_ok=True, parents=True)

        video_paths = []
        width, height, num_frames = self._config.validation.video_dims

        if sampling_ctx:
            sampling_ctx.start_video(0)

        condition_image = None
        if use_images and len(self._config.validation.images) > 0:
            image_path = self._config.validation.images[0]
            image = open_image_as_srgb(image_path)
            condition_image = F.to_tensor(image)

        reference_video = None
        if use_reference_videos and len(self._config.validation.reference_videos) > 0:
            ref_video_path = self._config.validation.reference_videos[0]
            reference_video, _ = read_video(ref_video_path, max_frames=num_frames)

      
        cached_embeddings = self._get_prompt_embedding(my_prompt)

        # ========== Generate reference trajectory cache (forced regeneration conditions: modality switch or initial validation) ==========
        cache_dir = str(Path(self._config.output_dir).parent / "trajectory")
        cache_path = os.path.join(cache_dir, f"validation_trajectory_cached_rank{rank}.pt")

        # Force regeneration: modality switch (skip_reward_processing=True) or initial validation (global_step==0)
        force_regen = skip_reward_processing or force_refresh_cache or (self._global_step == 0)
        if force_regen and os.path.exists(cache_path):
            os.remove(cache_path)
            if IS_MAIN_PROCESS:
                print(f"Deleted stale reference cache for rank {rank} (force_regen)")

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        if dist.is_initialized():
            dist.barrier()
        if not os.path.exists(cache_path):
            if IS_MAIN_PROCESS or rank == 0:
                print(f"🔧 Rank {rank}: Generating reference trajectory cache (freeze_modality=None, SDE disabled)...")

            ref_config = GenerationConfig(
                prompt=my_prompt,
                negative_prompt=self._config.validation.negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=self._config.validation.frame_rate,
                num_inference_steps=inference_steps,
                guidance_scale=self._config.validation.guidance_scale,
                seed=self._config.validation.seed,
                condition_image=condition_image,
                reference_video=reference_video,
                reference_downscale_factor=self._config.validation.reference_downscale_factor,
                generate_audio=generate_audio,
                include_reference_in_output=False,
                cached_embeddings=cached_embeddings,
                stg_scale=self._config.validation.stg_scale,
                stg_blocks=self._config.validation.stg_blocks,
                stg_mode=self._config.validation.stg_mode,
                num_samples=world_size,        
                output_dir=output_dir,
                output_prefix=f"ref_step_{self._global_step:06d}_rank{rank}",
                freeze_modality=None,
                enable_sde=False,
                sde_noise_level=0.0,
                skip_reward_processing=True,
            )

            if sampling_ctx:
                sampling_ctx.start_video(0)
            ref_videos, ref_audios, _ = sampler.generate(config=ref_config, device=self._accelerator.device, skip_reward_processing=True)
            if sampling_ctx:
                sampling_ctx.cleanup()
            if IS_MAIN_PROCESS:
                print(f"✅ Rank {rank}: Reference trajectory cache has been generated")

        self._accelerator.wait_for_everyone()

        # ========== Determine the freeze modality used for the current validation ==========
        current_validate_freeze = freeze_modality_override if freeze_modality_override is not None else getattr(self._config.validation, 'freeze_modality', None)
        if IS_MAIN_PROCESS:
            print(f"[_sample_videos] Using freeze_modality = {current_validate_freeze}")

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        if dist.is_initialized():
            dist.barrier()
        
        
        if current_validate_freeze == 'video': 
        # ========== Formal sampling (using freezing and SDE) ==========
            gen_config = GenerationConfig(
                prompt=my_prompt,
                negative_prompt=self._config.validation.negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=self._config.validation.frame_rate,
                num_inference_steps=inference_steps,
                guidance_scale=self._config.validation.guidance_scale,
                seed=self._config.validation.seed,
                condition_image=condition_image,
                reference_video=reference_video,
                reference_downscale_factor=self._config.validation.reference_downscale_factor,
                generate_audio=generate_audio,
                include_reference_in_output=self._config.validation.include_reference_in_output,
                cached_embeddings=cached_embeddings,
                stg_scale=self._config.validation.stg_scale,
                stg_blocks=self._config.validation.stg_blocks,
                stg_mode=self._config.validation.stg_mode,
                num_samples=total_samples,
                output_dir=output_dir,
                output_prefix=f"step_{self._global_step:06d}_rank{rank}",
                freeze_modality=current_validate_freeze,   
                enable_sde=True,
                sde_noise_level=0.8,
            )
        else:
            gen_config = GenerationConfig(
                prompt=my_prompt,
                negative_prompt=self._config.validation.negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=self._config.validation.frame_rate,
                num_inference_steps=inference_steps,
                guidance_scale=self._config.validation.guidance_scale,
                seed=self._config.validation.seed,
                condition_image=condition_image,
                reference_video=reference_video,
                reference_downscale_factor=self._config.validation.reference_downscale_factor,
                generate_audio=generate_audio,
                include_reference_in_output=self._config.validation.include_reference_in_output,
                cached_embeddings=cached_embeddings,
                stg_scale=self._config.validation.stg_scale,
                stg_blocks=self._config.validation.stg_blocks,
                stg_mode=self._config.validation.stg_mode,
                num_samples=total_samples,
                output_dir=output_dir,
                output_prefix=f"step_{self._global_step:06d}_rank{rank}",
                freeze_modality=current_validate_freeze,  
                enable_sde=True,
                sde_noise_level=0.02,
            )
        videos, audios, sample_details = sampler.generate(config=gen_config, device=self._accelerator.device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        if dist.is_initialized():
            dist.barrier()
    
        ext = "png" if num_frames == 1 else "mp4"
        for local_idx in range(samples_per_gpu):
            output_path = output_dir / f"{gen_config.output_prefix}_{local_idx:02d}.{ext}"
            video_paths.append(output_path)

        if sampling_ctx:
            sampling_ctx.cleanup()
        rel_outputs_path = output_dir.relative_to(self._config.output_dir)
        logger.info(f"🎥 Validation samples for step {self._global_step} saved in {rel_outputs_path}")

        torch.cuda.synchronize()
        self._accelerator.wait_for_everyone()
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._config.seed + self._global_step)
  

        # ========== Save advantage values and statistical rewards (only in non-skip mode) ==========
        if not skip_reward_processing:
            # Save advantage file (each rank saves separately)
            if sample_details and len(sample_details) == samples_per_gpu:
                my_advantages = [d['advantage'] for d in sample_details]
                adv_save_dir = Path(self._config.output_dir) / "advantages"
                adv_save_dir.mkdir(exist_ok=True, parents=True)
                adv_path = adv_save_dir / f"step_{self._global_step:06d}_rank{rank}.pt"
                torch.save({
                    'video_advantage': torch.tensor(my_advantages),
                    'audio_advantage': torch.tensor(my_advantages)
                }, adv_path)
                if IS_MAIN_PROCESS:
                    logger.info(f"💾 Rank {rank} advantage values saved to: {adv_path}")
            else:
                logger.error(f"[Rank {rank}] No valid sample_details obtained，number of samples={len(sample_details) if sample_details else 0}，expected={samples_per_gpu}")

            # ================== Diagnostic print: entering reward statistics ==================
            if IS_MAIN_PROCESS:
                print(f"\n[DEBUG REWARD] skip=False, step={self._global_step}, freeze_override={freeze_modality_override}, current_validate_freeze={current_validate_freeze}")
                print(f"[DEBUG REWARD] sample_details is None: {sample_details is None}, len={len(sample_details) if sample_details is not None else 'N/A'}")

            if sample_details is not None and len(sample_details) > 0:
                    
                    if current_validate_freeze == 'video':
                        metric_names = ['audio_quality', 'clap', 'desync']
                    elif current_validate_freeze == 'audio':
                        metric_names = ['visual_quality', 'desync', 'clip','overexposure']
                    else:
                        metric_names = []
                        if IS_MAIN_PROCESS:
                            print(f"[DEBUG REWARD] Unknown freeze_modality, skip recording")

                    if metric_names:
                     
                        metric_sums = {name: 0.0 for name in metric_names}
                        metric_counts = {name: 0 for name in metric_names}
                        for r in sample_details:
                            for name in metric_names:
                                val = r.get(name, float('nan'))
                                if not np.isnan(val):
                                    metric_sums[name] += val
                                    metric_counts[name] += 1

                        my_metric_avgs = {}
                        for name in metric_names:
                            if metric_counts[name] > 0:
                                my_metric_avgs[name] = metric_sums[name] / metric_counts[name]
                            else:
                                my_metric_avgs[name] = float('nan')

                        
                        if dist.is_initialized() and world_size > 1:
                        
                            all_metrics = [None] * world_size if rank == 0 else None
                            dist.gather_object(my_metric_avgs, all_metrics, dst=0)
                            if rank == 0:
                               
                                global_avgs = {}
                                for name in metric_names:
                                    values = []
                                    for r_dict in all_metrics:
                                        v = r_dict.get(name, float('nan'))
                                        if not np.isnan(v):
                                            values.append(v)
                                    if values:
                                        global_avgs[name] = np.mean(values)
                                    else:
                                        global_avgs[name] = float('nan')
                        else:
                            
                            global_avgs = my_metric_avgs

                        
                        if IS_MAIN_PROCESS:
                            history_file = Path(self._config.output_dir) / "reward_history.json"
                            history = []
                            if history_file.exists() and history_file.stat().st_size > 0:
                                try:
                                    with open(history_file, 'r') as f:
                                        history = json.load(f)
                                except json.JSONDecodeError:
                                    print(f"Warning: {history_file} corrupted, resetting")
                                    history = []

                            record = {"step": self._global_step}
                            for name in metric_names:
                                record[name] = global_avgs[name]
                            history.append(record)

                            with open(history_file, 'w') as f:
                                json.dump(history, f, indent=2)
                            print(f"[DEBUG REWARD] Wrote record (global avg) to {history_file}")

                            # WandB
                            if self._wandb_run is not None:
                                
                                wandb_metrics = {}
                                for name in metric_names:
                                    wandb_metrics[f"reward_avg/{name}"] = global_avgs[name]
                                
                               
                                wandb_metrics["train/freeze_modality"] = current_validate_freeze
                                
                                
                                wandb.log(wandb_metrics, step=self._global_step)
                                
                                print(f"✅ WandB: Reward metrics have been logged to step {self._global_step}")

                           
                            self._plot_reward_curves()
                            print(f"[DEBUG REWARD] Plotted curves")

                            print(f"\n[Reward Record] Step {self._global_step} | {current_validate_freeze}")
                            for name in metric_names:
                                print(f"  {name}: {global_avgs[name]:.4f}")
                    else:
                        if IS_MAIN_PROCESS:
                            print("[DEBUG REWARD] metric_names is empty, no record saved")
            else:
                    if IS_MAIN_PROCESS:
                        print("[DEBUG REWARD] No sample_details, skip reward recording")

        return video_paths

    @staticmethod
    def _log_training_stats(stats: TrainingStats) -> None:
        """Log training statistics."""
        stats_str = (
            "📊 Training Statistics:\n"
            f" - Total time: {stats.total_time_seconds / 60:.1f} minutes\n"
            f" - Training speed: {stats.steps_per_second:.2f} steps/second\n"
            f" - Samples/second: {stats.samples_per_second:.2f}\n"
            f" - Peak GPU memory: {stats.peak_gpu_memory_gb:.2f} GB"
        )
        if stats.num_processes > 1:
            stats_str += f"\n - Number of processes: {stats.num_processes}\n"
            stats_str += f" - Global batch size: {stats.global_batch_size}"
        logger.info(stats_str)

    def _save_reward_history(self, step: int, video_avg: float, audio_avg: float) -> None:
       
        if not IS_MAIN_PROCESS:
            return
        history_file = Path(self._config.output_dir) / "reward_history.json"
        history = []
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
        history.append({
            "step": step,
            "video_avg": video_avg,
            "audio_avg": audio_avg
        })
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Reward history saved to {history_file}")

    def _plot_reward_curves(self) -> None:
        if not IS_MAIN_PROCESS:
            return
        history_file = Path(self._config.output_dir) / "reward_history.json"
        if not history_file.exists() or history_file.stat().st_size == 0:
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, cannot plot reward curves")
            return

        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {history_file} is corrupted, cannot plot")
            return

        if not history:
            return

        
        video_records = [r for r in history if 'visual_quality' in r]
        audio_records = [r for r in history if 'audio_quality' in r]

        def plot_curves(records, metrics, title, filename):
            if not records:
                return
            steps = [r['step'] for r in records]                    
            plt.figure(figsize=(10, 6))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            markers = ['o', 's', '^', 'D', 'v', 'p']
            linestyles = ['-', '--', '-.', ':', '-', '--']
            for i, metric in enumerate(metrics):
                values = [r.get(metric, float('nan')) for r in records]
                valid_steps = []
                valid_values = []
                for s, val in zip(steps, values):
                    if val is not None and not np.isnan(val):
                        valid_steps.append(s)
                        valid_values.append(val)
                if valid_steps:
                    plt.plot(valid_steps, valid_values,
                            label=metric,
                            color=colors[i % len(colors)],
                            marker=markers[i % len(markers)],
                            linestyle=linestyles[i % len(linestyles)],
                            linewidth=2, markersize=6)
            plt.xlabel('Training Step', fontsize=12)                 
            plt.ylabel('Average Reward', fontsize=12)
            plt.title(title, fontsize=14)
            plt.legend(loc='best', fontsize=9)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            save_path = Path(self._config.output_dir) / filename
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved {filename}")

        
        video_metrics = ['visual_quality', 'desync', 'clip']
        plot_curves(video_records, video_metrics, 'Video Modality Rewards', 'reward_curves_video.png')

        audio_metrics = ['audio_quality', 'clap', 'desync']
        plot_curves(audio_records, audio_metrics, 'Audio Modality Rewards', 'reward_curves_audio.png')

    def _save_checkpoint(self) -> Path | None:
        
        import torch
        from safetensors.torch import save_file
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        is_lora = self._config.model.training_mode == "lora"
        is_fsdp = self._accelerator.distributed_type == DistributedType.FSDP
        save_dir = Path(self._config.output_dir) / "checkpoints"
        filename = f"model_weights_step_{self._global_step:05d}.safetensors"
        saved_weights_path = save_dir / filename

        self._accelerator.wait_for_everyone()

        state_dict = None
        if is_fsdp:
            with FSDP.summon_full_params(self._transformer, writeback=False, offload_to_cpu=True):
                if IS_MAIN_PROCESS:
                    state_dict = {}
                   
                    unwrap_tfm = self._accelerator.unwrap_model(self._transformer)
                    for name, param in unwrap_tfm.named_parameters():
                        if any(x in name for x in ["video_connector", "audio_connector", "feature_extractor", "gemma"]):
                            continue
                        
                        state_dict[name] = param.detach().cpu().clone().to(torch.bfloat16)
                else:
                    state_dict = None
        else:
            if IS_MAIN_PROCESS:
                unwrap_tfm = self._accelerator.unwrap_model(self._transformer)
                state_dict = {}
                for name, param in unwrap_tfm.named_parameters():
                    if any(x in name for x in ["video_connector", "audio_connector", "feature_extractor", "gemma"]):
                        continue
                    state_dict[name] = param.detach().cpu().clone().to(torch.bfloat16)
            else:
                state_dict = None

        if not IS_MAIN_PROCESS:
            return None

        save_dir.mkdir(exist_ok=True, parents=True)

        
        export_sd = {}
        if is_lora:
            for k, v in state_dict.items():
                if "lora_" in k:
                    export_sd[k] = v
            print(f"[CHECKPOINT] Number of LoRA weights: {len(export_sd)} ✅")
        else:
            export_sd = state_dict
            print(f"[CHECKPOINT] Number of pure Transformer weights: {len(export_sd)} ✅")

        save_file(export_sd, saved_weights_path, metadata={"model_type": "ltx"})

        logger.info(f"✅ Model saved: {saved_weights_path}")
        self._checkpoint_paths.append(saved_weights_path)
        self._cleanup_checkpoints()
        return saved_weights_path

    def _cleanup_checkpoints(self) -> None:
        """Clean up old checkpoints."""
        if 0 < self._config.checkpoints.keep_last_n < len(self._checkpoint_paths):
            checkpoints_to_remove = self._checkpoint_paths[: -self._config.checkpoints.keep_last_n]
            for old_checkpoint in checkpoints_to_remove:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.info(f"Removed old checkpoints: {old_checkpoint}")
            # Update the list to only contain kept checkpoints
            self._checkpoint_paths = self._checkpoint_paths[-self._config.checkpoints.keep_last_n :]

    def _build_checkpoint_metadata(self) -> dict[str, str]:
        """Build metadata dictionary for safetensors checkpoint.
        Delegates to the training strategy to get strategy-specific metadata
        that downstream inference pipelines may need.
        Returns:
            Dictionary of string key-value pairs for safetensors metadata.
            Values are converted to strings for safetensors compatibility.
        """
        raw_metadata = self._training_strategy.get_checkpoint_metadata()
        # Convert all values to strings for safetensors compatibility
        metadata = {k: str(v) for k, v in raw_metadata.items()}
        if metadata:
            logger.info(f"Saving checkpoint metadata: {metadata}")
        return metadata

    def _save_config(self) -> None:
        """Save the training configuration as a YAML file in the output directory."""
        if not IS_MAIN_PROCESS:
            return

        config_path = Path(self._config.output_dir) / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self._config.model_dump(), f, default_flow_style=False, indent=2)

        logger.info(f"💾 Training configuration saved to: {config_path.relative_to(self._config.output_dir)}")

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases run."""
        if not self._config.wandb.enabled or not IS_MAIN_PROCESS:
            self._wandb_run = None
            return

        wandb_config = self._config.wandb
        run = wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            name=Path(self._config.output_dir).name,
            tags=wandb_config.tags,
            config=self._config.model_dump(),
        )
        self._wandb_run = run

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to Weights & Biases."""
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)

    def _log_validation_samples(self, sample_paths: list[Path], prompts: list[str]) -> None:
        """Log validation samples (videos or images) to W&B."""
        if not self._config.wandb.log_validation_videos or self._wandb_run is None:
            return

        # Determine if outputs are images or videos based on file extension
        is_image = sample_paths and sample_paths[0].suffix.lower() in (".png", ".jpg", ".jpeg", ".heic", ".webp")
        media_cls = wandb.Image if is_image else wandb.Video

        samples = [media_cls(str(path), caption=prompt) for path, prompt in zip(sample_paths, prompts, strict=True)]
        self._wandb_run.log({"validation_samples": samples}, step=self._global_step)