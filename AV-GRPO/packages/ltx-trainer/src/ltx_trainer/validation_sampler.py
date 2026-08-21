"""Validation sampling for LTX-2 training using ltx-core components.
This module provides a simplified validation pipeline for generating samples during training,
using the new ltx-core components (VideoLatentTools, AudioLatentTools, LatentState, etc.).
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal
import tempfile
import shutil
from ltx_trainer.reward_computation import _compute_vq_batch, _compute_other_metrics
import multiprocessing as mp
import queue
import sys
import os
import uuid
import imageio
import soundfile as sf
from pathlib import Path
from pathlib import Path
import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from typing import List, Literal, Tuple, TYPE_CHECKING
from ltx_trainer.ltx_trajectory_saver import TrajectoryCollector
import math
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from .video_utils import read_video, save_video
from .utils import save_image
import soundfile as sf
from pathlib import Path

class FlowGRPOSDEDiffusionStep(DiffusionStepProtocol):
    """
    SDE sampler exactly matching Flow GRPO paper (Equation 9), with adaptive noise
    clipping to prevent numerical explosion while preserving marginal distributions.

    Args:
        noise_level: "a" parameter in paper (σ_t = a * sqrt(t/(1-t))), default 1.0
        sde_start_step: First step index to enable SDE noise
        sde_end_step: Last step index to enable SDE noise (None = until end)
        max_noise_std: Maximum allowed noise standard deviation per step.
                       When σ_t * sqrt(Δt) exceeds this, noise and score correction
                       are scaled down proportionally to maintain the Fokker-Planck
                       relationship. Recommended: 0.3 ~ 0.6 for 10-step sampling.
    """

    def __init__(
        self,
        noise_level: float = 0.02,
        sde_start_step: int = 0,
        sde_end_step: int | None = None,
        max_noise_std: float = 0.5,
    ):
        self.noise_level = noise_level
        self.sde_start_step = sde_start_step
        self.sde_end_step = sde_end_step
        self.max_noise_std = max_noise_std

    def step(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
        generator: torch.Generator | None = None,
        return_log_prob: bool = False,      
        return_details: bool = False,       
        **_kwargs
    ):
        t = sigmas[step_index]
        t_next = sigmas[step_index + 1]
        dt = t_next - t

        t_b = t.view(-1, *([1] * (sample.ndim - 1)))
        t_next_b = t_next.view(-1, *([1] * (sample.ndim - 1)))
        dt_b = t_next_b - t_b

        v = (sample - denoised_sample) / t_b.clamp(min=1e-5)
        x_next_ode = sample + v * dt_b

        use_sde = (
            self.noise_level > 0.0
            and step_index >= self.sde_start_step
            and (self.sde_end_step is None or step_index <= self.sde_end_step)
        )

        if not use_sde:
            
            dt_abs = -dt_b
            sigma_t_effective = torch.zeros_like(t_b)
            noise = torch.zeros_like(sample)
            prev_sample_mean = x_next_ode
            next_sample = x_next_ode
            variance = torch.zeros_like(t_b)  
            log_prob = torch.zeros(sample.shape[0], device=sample.device, dtype=sample.dtype)
        else:
            sigma_t_uncapped = self.noise_level * torch.sqrt(t_b / (1.0 - t_b).clamp(min=1e-5))
            dt_abs = -dt_b
            noise_std_uncapped = sigma_t_uncapped * torch.sqrt(dt_abs)

            scale = torch.clamp(self.max_noise_std / (noise_std_uncapped + 1e-8), max=1.0)
            sigma_t_effective = sigma_t_uncapped * scale
            noise_std = noise_std_uncapped * scale

            correction_coeff = (sigma_t_effective ** 2) / (2.0 * t_b.clamp(min=1e-5))
            inner = sample + (1.0 - t_b) * v
            score_correction = - correction_coeff * inner * dt_abs

            prev_sample_mean = sample + v * dt_b + score_correction

            noise = torch.randn_like(sample) if generator is None else \
                    torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            noise_term = noise_std * noise
            next_sample = prev_sample_mean + noise_term

            variance = noise_std ** 2

        if return_details:
            
            if not use_sde:
                log_prob = torch.zeros(sample.shape[0], device=sample.device, dtype=sample.dtype)
            else:
                log_prob = -((next_sample.detach() - prev_sample_mean.detach()) ** 2) / (2 * variance) \
                           - 0.5 * torch.log(2 * torch.pi * variance)
                log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            return {
                "next_sample": next_sample.to(sample.dtype),
                "log_prob": log_prob,
                "prev_sample_mean": prev_sample_mean,
                "dt_abs": dt_abs,
                "sigma_t_effective": sigma_t_effective,
                "noise": noise,
            }

        if return_log_prob:
            if not use_sde:
                log_prob = torch.zeros(sample.shape[0], device=sample.device, dtype=sample.dtype)
            else:
                log_prob = -((next_sample.detach() - prev_sample_mean.detach()) ** 2) / (2 * variance) \
                           - 0.5 * torch.log(2 * torch.pi * variance)
                log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            return next_sample.to(sample.dtype), log_prob

        return next_sample.to(sample.dtype)

from ltx_core.components.guiders import CFGGuider, STGGuider
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import (
    AudioPatchifier,
    VideoLatentPatchifier,
    get_pixel_coords,
)
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import X0Model
from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import AudioLatentShape, LatentState, SpatioTemporalScaleFactors, VideoLatentShape, VideoPixelShape
from ltx_trainer.progress import SamplingContext

if TYPE_CHECKING:
    from ltx_core.model.audio_vae import AudioDecoder, Vocoder
    from ltx_core.model.transformer import LTXModel
    from ltx_core.model.video_vae import VideoDecoder, VideoEncoder
    from ltx_core.text_encoders.gemma import GemmaTextEncoder
    from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor

VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


@dataclass
class CachedPromptEmbeddings:
    """Pre-computed text embeddings for a validation prompt.
    These embeddings are computed once at training start and reused for all validation runs,
    avoiding the need to load the full Gemma text encoder during validation.
    """

    video_context_positive: Tensor  # [1, seq_len, hidden_dim]
    audio_context_positive: Tensor  # [1, seq_len, hidden_dim]
    video_context_negative: Tensor | None = None
    audio_context_negative: Tensor | None = None


@dataclass
class TiledDecodingConfig:
    """Configuration for tiled video decoding to reduce VRAM usage.
    Tiled decoding splits the latent tensor into overlapping tiles, decodes each
    tile individually, and blends them together. This significantly reduces peak
    VRAM usage at the cost of slightly slower decoding.
    Defaults match the recommended values from ltx-core tests.
    """

    enabled: bool = True  # Whether to use tiled decoding (enabled by default)
    tile_size_pixels: int = 192  # Spatial tile size in pixels (must be ≥64 and divisible by 32)
    tile_overlap_pixels: int = 64  # Spatial tile overlap in pixels (must be divisible by 32)
    tile_size_frames: int = 48  # Temporal tile size in frames (must be ≥16 and divisible by 8)
    tile_overlap_frames: int = 24  # Temporal tile overlap in frames (must be divisible by 8)


@dataclass
class GenerationConfig:
    """Configuration for video/audio generation."""

    prompt: str  # Text prompt for generation
    negative_prompt: str = ""  # Negative prompt to avoid unwanted artifacts
    height: int = 544  # Output video height in pixels
    width: int = 960  # Output video width in pixels
    num_frames: int = 97  # Number of frames to generate
    frame_rate: float = 25.0  # Frame rate for temporal position scaling
    num_inference_steps: int = 30  # Number of denoising steps
    guidance_scale: float = 4.0  # CFG guidance scale
    seed: int = 42  # Random seed for reproducibility
    condition_image: Tensor | None = None  # Optional first frame image for image-to-video
    reference_video: Tensor | None = None  # For IC-LoRA: [F, C, H, W] in [0, 1]
    reference_downscale_factor: int = 1  # For IC-LoRA: downscale factor (1 = same resolution, 2 = half resolution)
    generate_audio: bool = True  # Whether to generate audio alongside video
    include_reference_in_output: bool = False  # For IC-LoRA: concatenate original reference with generated output
    cached_embeddings: CachedPromptEmbeddings | None = None  # Pre-computed text embeddings (avoids loading Gemma)
    stg_scale: float = 0.0  # STG strength (0.0 = disabled, recommended: 1.0)
    stg_blocks: list[int] | None = None  # Transformer blocks to perturb (None = all, recommended: [29])
    stg_mode: Literal["stg_av", "stg_v"] = "stg_av"  # STG mode: "stg_av" (audio+video) or "stg_v" (video only)
    # Tiled decoding config: None = use defaults (enabled), False = disable, or TiledDecodingConfig for custom settings
    tiled_decoding: TiledDecodingConfig | Literal[False] | None = None
    enable_sde: bool = True          
    sde_noise_level: float = 0.02      
    sde_sigma_max: float | None = None  
    num_samples: int = 4                
    output_dir: str | Path | None = str(Path(__file__).resolve().parents[4])
    output_prefix: str = "sample"       
    freeze_modality: Literal["video", "audio"] | None = "audio"
    skip_reward_processing: bool = False   

    def __post_init__(self) -> None:
        """Apply default tiled decoding config if not provided."""
        if self.tiled_decoding is None:
            # Use default config with tiling enabled
            object.__setattr__(self, "tiled_decoding", TiledDecodingConfig())
        elif self.tiled_decoding is False:
            # Explicitly disabled - use config with enabled=False
            object.__setattr__(self, "tiled_decoding", TiledDecodingConfig(enabled=False))

def _compute_rewards_subprocess_file(tmp_dir, prompts, freeze_modality, device_str, output_json):
    import warnings
    import os
    import json
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

    try:
        from ltx_trainer.reward_computation import compute_rewards_from_files
   
        results = compute_rewards_from_files(
            tmp_dir=tmp_dir,
            prompts=prompts,
            freeze_modality=freeze_modality,
            device_str=device_str,
        )
        with open(output_json, 'w') as f:
            json.dump(results, f)
    except Exception as e:
        import traceback
        traceback.print_exc()
        with open(output_json, 'w') as f:
            json.dump({"error": str(e)}, f)

def _compute_rewards_subprocess_v2(tmp_dir, prompts, freeze_modality, device_str, output_json):
    import warnings
    import os
    import json
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

    try:
       
        video_paths = []
        audio_paths = []
        for i in range(len(prompts)):
            video_path = os.path.join(tmp_dir, f"sample_{i}.mp4")
            audio_path = os.path.join(tmp_dir, f"sample_{i}.wav")
            video_paths.append(video_path)
            audio_paths.append(audio_path if os.path.exists(audio_path) else None)

       
        from ltx_trainer.validation_sampler import _compute_rewards_impl
        results = _compute_rewards_impl(
            video_paths, audio_paths, prompts, freeze_modality, device_str, tmp_dir
        )
        with open(output_json, 'w') as f:
            json.dump(results, f)
    except Exception as e:
        import traceback
        traceback.print_exc()
        with open(output_json, 'w') as f:
            json.dump({"error": str(e)}, f)

class ValidationSampler:
    """Generates validation samples during training using ltx-core components.
    This class provides a simplified interface for generating video (and optionally audio)
    samples during training validation. It supports:
    - Text-to-video generation
    - Image-to-video generation (first frame conditioning)
    - Video-to-video generation (IC-LoRA reference video conditioning)
    - Optional audio generation
    The implementation follows the patterns from ltx_pipelines.single_stage.
    Text embeddings can be provided either via:
    - A full text_encoder (encodes prompts on-the-fly)
    - Pre-computed cached_embeddings (avoids loading Gemma during validation)
    """

    def __init__(
        self,
        transformer: "LTXModel",
        vae_decoder: "VideoDecoder",
        vae_encoder: "VideoEncoder | None",
        text_encoder: "GemmaTextEncoder | None" = None,
        audio_decoder: "AudioDecoder | None" = None,
        vocoder: "Vocoder | None" = None,
        sampling_context: SamplingContext | None = None,
        embeddings_processor: "EmbeddingsProcessor | None" = None,
    ):
        """Initialize the validation sampler.
        Args:
            transformer: LTX-2 transformer model
            vae_decoder: Video VAE decoder
            vae_encoder: Video VAE encoder (for image/video conditioning), can be None if not needed
            text_encoder: Gemma text encoder (optional if cached_embeddings in config)
            audio_decoder: Optional audio VAE decoder (for audio generation)
            vocoder: Optional vocoder (for audio generation)
            sampling_context: Optional SamplingContext for progress display during denoising
            embeddings_processor: Optional embeddings processor (required if text_encoder provided)
        """
        self._transformer = transformer
        self._vae_decoder = vae_decoder
        self._vae_encoder = vae_encoder
        self._text_encoder = text_encoder
        self._embeddings_processor = embeddings_processor
        self._audio_decoder = audio_decoder
        self._vocoder = vocoder
        self._sampling_context = sampling_context
        self._reward_models_loaded = False
        self._videoreward_model = None
        self._clip_model = None
        self._clip_processor = None
        self._audiobox_model = None
        self._clap_model = None
        self._syncformer_model = None
        # Patchifiers
        self._video_patchifier = VideoLatentPatchifier(patch_size=1)
        self._audio_patchifier = AudioPatchifier(patch_size=1)

    def _compute_single_reward(
        self,
        video: Tensor,
        audio: Tensor | None,
        prompt: str,
        freeze_modality: str | None,
        device: torch.device,
    ) -> dict:
        """
        Compute all raw reward metrics for a single sample.
        Returns a dictionary, e.g., {'visual_quality': xxx, 'clip': xxx, 'desync': xxx, ...}
        If a metric computation fails, the corresponding value is set to 0.0.
        """
        try:
            from .validation_sampler import _compute_rewards_impl
            videos_cpu = [video.cpu()]
            audios_cpu = [audio.cpu() if audio is not None else None]
            result = _compute_rewards_impl(
                videos_cpu, audios_cpu, [prompt], freeze_modality, str(device), "/tmp/ltx_reward"
            )
            detail = result['sample_details'][0]
          
            import numpy as np
            for k, v in detail.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    print(f"[Rank 0] Warning: {k} is {v}, replacing with 0.0")
                    detail[k] = 0.0
            return detail
        except Exception as e:
            print(f"[Rank 0] ❌ Reward computation failed: {e}, returning all-zero dictionary.")
            return {}

    def _init_reward_models(self, device: torch.device, freeze_modality: str | None):
      
        if self._reward_models_loaded:
            return

       
        AV_GRPO_ROOT = Path(__file__).resolve().parents[4]

        LOCAL_VIDEO_REWARD = str(AV_GRPO_ROOT / "JavisDiT" / "checkpoints" / "VideoReward")
        LOCAL_IMAGEBIND = str(AV_GRPO_ROOT / "JavisDiT" / "checkpoints" / "imagebind_huge.pth")
        LOCAL_SYNCHFORMER = str(AV_GRPO_ROOT / "JavisDiT" / "checkpoints" / "synchformer_state_dict.pth")

        
        import sys
        javisdit_root = str(Path(__file__).resolve().parents[4] / "JavisDiT")
        if javisdit_root not in sys.path:
            sys.path.insert(0, javisdit_root)

        from eval.javisbench.src.metrics import (
            calc_video_quality_score,
            calc_audio_quality_score,
            calc_clap_score,
            calc_clip_score,
            calc_desync_score,
        )

        self._calc_video_quality = calc_video_quality_score
        self._calc_audio_quality = calc_audio_quality_score
        self._calc_clap = calc_clap_score
        self._calc_clip = calc_clip_score
        self._calc_desync = calc_desync_score

   
        if freeze_modality == "audio":
    
            self._videoreward_model = ... 
            self._clip_model, self._clip_processor = ...  
            self._syncformer_model = ...  
        elif freeze_modality == "video":
          
            self._audiobox_model = ... 
            self._clap_model = ...     
            self._syncformer_model = ...  
        else:
           
            pass

        self._reward_models_loaded = True

    def _compute_rewards_via_subprocess(
        self,
        tmp_dir: str,
        prompts: list[str],
        freeze_modality: str | None,
        device: torch.device,
    ) -> dict | None:
        import multiprocessing as mp
        import json

        output_json = os.path.join(tmp_dir, "rewards.json")
        ctx = mp.get_context('spawn')
        p = ctx.Process(
            target=_compute_rewards_subprocess_v2,
            args=(tmp_dir, prompts, freeze_modality, str(device), output_json)
        )
        p.start()
        p.join()

        if not os.path.exists(output_json):
            print("❌ Subprocess did not generate a result file")
            return None
        try:
            with open(output_json, 'r') as f:
                data = json.load(f)
            if "error" in data:
                print(f"❌ Subprocess error: {data['error']}")
                return None
            return data
        except Exception as e:
            print(f"❌ Failed to read result: {e}")
            return None

    def _compute_rewards(
        self,
        videos: list[torch.Tensor],
        audios: list[torch.Tensor | None],
        prompts: list[str],
        freeze_modality: str | None,
        device: torch.device,
        tmp_root: str = "/tmp/ltx_reward",
    ):
        if len(videos) == 0:
            return None

        import uuid
        import imageio
        import soundfile as sf
        from pathlib import Path

        tmp_dir = Path(tmp_root) / str(uuid.uuid4())
        tmp_dir.mkdir(parents=True, exist_ok=True)

        video_paths = []
        audio_paths = []
        for i, (video, audio) in enumerate(zip(videos, audios)):
            video_np = (video.permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)
            video_path = tmp_dir / f"sample_{i}.mp4"
            imageio.mimsave(video_path, video_np, fps=24)
            video_paths.append(str(video_path))

            if audio is not None:
                audio_np = audio.cpu().numpy().T
                audio_path = tmp_dir / f"sample_{i}.wav"
                sf.write(audio_path, audio_np, samplerate=16000)
                audio_paths.append(str(audio_path))
            else:
                audio_paths.append(None)

  
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        p = ctx.Process(
            target=_compute_rewards_subprocess,
            args=(video_paths, audio_paths, prompts, freeze_modality, str(device), tmp_root, result_queue)
        )
        p.start()
        p.join()  


        try:
            status, data = result_queue.get_nowait()
            if status == 'success':
          
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return data
            else:
                print(f"❌ 子进程奖励计算失败: {data}")
                return None
        except queue.Empty:
            print("❌ 子进程无响应")
            return None

    # Note: Use @torch.no_grad() instead of @torch.inference_mode() to avoid FSDP inplace update errors after validation

    @torch.no_grad()
    def generate(
        self,
        config: GenerationConfig,
        device: torch.device | str = "cuda",
        skip_reward_processing: bool = False,
    ) -> tuple[List[Tensor], List[Tensor | None], List[dict] | None]:
        import torch.distributed as dist
        import numpy as np
        import os, json, time, torchaudio

        device = torch.device(device) if isinstance(device, str) else device
        self._validate_config(config)

  
        rng_states = {}
        if dist.is_initialized():
            rng_states["cpu"] = torch.random.get_rng_state()
            if torch.cuda.is_available():
                rng_states["cuda"] = torch.cuda.get_rng_state_all()

        num_samples = getattr(config, 'num_samples', 1)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        samples_per_rank = num_samples // world_size
        start_idx = rank * samples_per_rank
        end_idx = start_idx + samples_per_rank

        output_base = Path(config.output_dir) if hasattr(config, 'output_dir') and config.output_dir else Path("outputs/samples")
        if rank == 0:
            output_base.mkdir(parents=True, exist_ok=True)

        my_videos = []
        my_audios = []
        my_video_paths = []
        my_audio_paths = []


        for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
            sample_seed = config.seed + global_idx if config.seed is not None else 42 + global_idx
            sample_config = replace(config, seed=sample_seed)

            if sample_config.reference_video is not None:
                video, audio = self._generate_with_reference(sample_config, device)
            else:
                video, audio = self._generate_standard(
                    sample_config, device,
                    sample_index=local_idx,
                    save_trajectory=True
                )

            ext = "png" if config.num_frames == 1 else "mp4"
            prefix = getattr(config, 'output_prefix', 'sample')
            output_path = output_base / f"{prefix}_{local_idx:02d}.{ext}"
            if config.num_frames == 1:
                save_image(video, output_path)
            else:
                save_video(
                    video_tensor=video, output_path=output_path, fps=config.frame_rate,
                    audio=audio, audio_sample_rate=self._vocoder.output_sampling_rate if audio is not None else None,
                )
            print(f"[Rank {rank}] Sample {local_idx+1}/{samples_per_rank} saved to {output_path}")

            # 显式保存重采样后的音频文件 (16000 Hz)
            if audio is not None:
                audio_path = output_path.with_suffix('.wav')
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                actual_sr = self._vocoder.output_sampling_rate if self._vocoder else 24000
                if actual_sr != 16000:
                    resampler = torchaudio.transforms.Resample(actual_sr, 16000).to(audio.device)
                    audio = resampler(audio)
                torchaudio.save(str(audio_path), audio.cpu(), sample_rate=16000)
                my_audio_paths.append(str(audio_path))
            else:
                my_audio_paths.append(None)

            my_videos.append(video.cpu())
            my_audios.append(audio.cpu() if audio is not None else None)
            my_video_paths.append(str(output_path))
            del video, audio
            torch.cuda.empty_cache()

        if dist.is_initialized():
            dist.barrier()

     
        if skip_reward_processing:
            if dist.is_initialized():
                if "cpu" in rng_states:
                    torch.random.set_rng_state(rng_states["cpu"])
                if "cuda" in rng_states:
                    torch.cuda.set_rng_state_all(rng_states["cuda"])
            return my_videos, my_audios, []

        freeze_modality = getattr(config, 'freeze_modality', None)


        step_str = config.output_prefix.split('_')[1]
        path_file = output_base / f"video_paths_rank{rank}_step{step_str}.json"
        with open(path_file, 'w') as f:
            json.dump({
                "video_paths": my_video_paths,
                "audio_paths": my_audio_paths,
                "prompt": config.prompt
            }, f)

        if dist.is_initialized():
            dist.barrier()

        # ---------- Visual Quality Computation ----------
        vq_file = output_base / f"vq_results_step{step_str}.json"
        if freeze_modality != "video":
            if rank == 0:
                all_video_paths = []
                all_prompts = []
                for r in range(world_size):
                    r_file = output_base / f"video_paths_rank{r}_step{step_str}.json"
                    for _ in range(600):
                        if r_file.exists():
                            break
                        time.sleep(0.1)
                    with open(r_file) as f:
                        data = json.load(f)
                    all_video_paths.extend(data["video_paths"])
                    saved_prompt = data.get("prompt", config.prompt)
                    all_prompts.extend([saved_prompt] * len(data["video_paths"]))

                print(f"\n🚀 Rank 0 launching subprocess to compute Visual Quality for {len(all_video_paths)} samples...")
                from ltx_trainer.reward_computation import _run_vq_subprocess
                _run_vq_subprocess(all_video_paths, all_prompts, str(vq_file), str(device))
                if not vq_file.exists() or vq_file.stat().st_size == 0:
                    raise RuntimeError("VQ subprocess failed to generate the result file")
                print(f"✅ VQ results saved ({vq_file.stat().st_size} bytes)")

            if dist.is_initialized():
                dist.barrier()
            time.sleep(1.0)

            if not vq_file.exists():
                raise FileNotFoundError(f"VQ result file missing: {vq_file}")
            with open(vq_file) as f:
                vq_all = json.load(f)
            my_vq_list = vq_all[rank * samples_per_rank : (rank + 1) * samples_per_rank]

            print(f"\n[Rank {rank}] Visual Quality / Motion Quality for this group of samples:")
            for idx, vq in enumerate(my_vq_list):
                print(f"  Sample {idx}: visual_quality = {vq['visual_quality']:.4f}, motion_quality = {vq['motion_quality']:.4f}")
        else:
            my_vq_list = [{'visual_quality': 0.0, 'motion_quality': 0.0}] * samples_per_rank
            if rank == 0:
                print(f"[Rank {rank}] Skipping Visual Quality computation（freeze_modality=video），使用占位值。")

        # ---------- Unified CLAP computation (only when video is frozen) ----------
        clap_file = output_base / f"clap_results_step{step_str}.json"
        my_clap_scores = [0.0] * samples_per_rank  
        if freeze_modality == "video":
        
            all_audio_paths = [None] * world_size if rank == 0 else None
            all_prompts_list = [None] * world_size if rank == 0 else None
            if dist.is_initialized():
                dist.gather_object(my_audio_paths, all_audio_paths, dst=0)
                dist.gather_object([config.prompt] * samples_per_rank, all_prompts_list, dst=0)
            else:
                all_audio_paths = [my_audio_paths]
                all_prompts_list = [[config.prompt] * samples_per_rank]

            if rank == 0:
                flat_audio = []
                flat_prompts = []
                for ap_list, p_list in zip(all_audio_paths, all_prompts_list):
                    for ap, p in zip(ap_list, p_list):
                        if ap is not None:
                            flat_audio.append(ap)
                            flat_prompts.append(p)
                print(f"\n🚀 Rank 0 批量计算 {len(flat_audio)} 个样本的 CLAP...")
               
                from ltx_trainer.reward_computation import _compute_clap_batch
                clap_scores = _compute_clap_batch(flat_audio, flat_prompts, device)
               
                global_clap = []
                idx = 0
                for ap_list in all_audio_paths:
                    for ap in ap_list:
                        if ap is not None:
                            global_clap.append(clap_scores[idx])
                            idx += 1
                        else:
                            global_clap.append(float('nan'))
                with open(clap_file, 'w') as f:
                    json.dump(global_clap, f)
                print(f"✅ CLAP results saved ({clap_file.stat().st_size} bytes)")

            if dist.is_initialized():
                dist.barrier()
            time.sleep(1.0)

            if not clap_file.exists():
                raise FileNotFoundError(f"CLAP result file missing: {clap_file}")
            with open(clap_file) as f:
                global_clap = json.load(f)
            my_clap_scores = global_clap[rank * samples_per_rank : (rank + 1) * samples_per_rank]
            print(f"[Rank {rank}] CLAP scores: {my_clap_scores}")


        # ---------- Each rank computes other metrics (CLIP, desync, audio...) ----------
        prompts_local = [config.prompt] * len(my_video_paths)
        from ltx_trainer.reward_computation import _compute_other_metrics
        sample_details = _compute_other_metrics(
            my_video_paths, my_audio_paths, prompts_local,
            freeze_modality, device, my_vq_list,
            clap_scores=my_clap_scores if freeze_modality == "video" else None
        )

        # ========== Within-group normalization (supports custom metric weights) ==========
        if freeze_modality == "audio":
            groups = {"video": {"keys": ["visual_quality", "clip", "desync","overexposure"], "weight": 1.0, "desync_negate": True}}
        elif freeze_modality == "video":
            groups = {"audio": {"keys": ["audio_quality", "clap", "desync"], "weight": 1.0, "desync_negate": True}}
        else:
            groups = {
                "video": {"keys": ["visual_quality", "clip", "desync"], "weight": 0.5, "desync_negate": True},
                "audio": {"keys": ["audio_quality", "clap", "desync"], "weight": 0.5, "desync_negate": True}
            }

        ##----- Custom within-group weights for each metric (modify as needed) ------
        key_weights = {
            "visual_quality": 1.0,
            "clip": 1.0,
            "desync": 1.0,
            "audio_quality": 1.0,
            "clap": 1.0,
            "overexposure": 1.0,
        }
        # If metrics within a group need different weights, simply modify the numbers above.
        #--------------------------------------------------------

        for r in sample_details:
            for group_cfg in groups.values():
                if group_cfg.get("desync_negate", False) and "desync" in r:
                    r["desync"] = -r["desync"]
                if "overexposure" in r:
                    r["overexposure"] = -r["overexposure"]

        all_keys = set()
        for group_cfg in groups.values():
            all_keys.update(group_cfg["keys"])
        metrics = {}
        for k in all_keys:
            arr = np.array([r.get(k, 0.0) for r in sample_details])
            arr = np.nan_to_num(arr, nan=0.0)
            metrics[k] = arr

        # Discarded
        vq_mean = metrics["visual_quality"].mean() if "visual_quality" in metrics else None
        use_vq_only = vq_mean is not None and vq_mean <= 0

        # Audio side: when the mean CLAP score < 0.4, only the CLAP metric is used for scoring.
        clap_mean = metrics["clap"].mean() if "clap" in metrics else None
        use_clap_only = clap_mean is not None and clap_mean < 0.4
        #=================================================

      
        print ("\n" + "=" * 80)
        print ("📊 Reward summary (weighted sum of normalized z-scores)")
        print ("=" * 80)

      
        if vq_mean is not None:
            if use_vq_only:
                print(f"⚠️ Video side：visual_quality mean = {vq_mean:.3f} ≤ 0.0，enabling [visual_quality-only scoring] mode")
            else:
                print(f"ℹ️ Video side：visual_quality mean = {vq_mean:.3f} > 0.0，normal multi-metric weighted scoring")
        if clap_mean is not None:
            if use_clap_only:
                print(f"⚠️ Audio side：CLAP mean = {clap_mean:.3f} < 0.4，enabling [CLAP-only scoring] mode")
            else:
                print(f"ℹ️ Audio side：CLAP mean = {clap_mean:.3f} ≥ 0.4，normal multi-metric weighted scoring")

        total_scores = []
        for i, r in enumerate(sample_details):
            detail_parts = []
            sample_score = 0.0

       
            for group_name, group_cfg in groups.items():
                group_score = 0.0
                weight_sum = 0.0

                # ---------- Video group scoring logic ----------
                if group_name == "video":
                    if use_vq_only:
                        k = "visual_quality"
                        mu = metrics[k].mean()
                        sigma = metrics[k].std() + 1e-8
                        z = (r[k] - mu) / sigma
                        group_score = z
                        detail_parts.append(f"{k}: {r[k]:.4f} → z={z:+.3f} (single-metric mode)")
                    else:
                        for k in group_cfg["keys"]:
                            if k == "overexposure":
                                continue
                            w = key_weights.get(k, 1.0)
                            mu = metrics[k].mean()
                            sigma = metrics[k].std() + 1e-8
                            z = (r[k] - mu) / sigma
                            group_score += z * w
                            weight_sum += w
                            detail_parts.append(f"{k}: {r[k]:.4f} → z={z:+.3f} (w={w})")
                        group_score /= weight_sum

                # ---------- Audio group scoring logic ----------
                elif group_name == "audio":
                    if use_clap_only:
                        k = "clap"
                        mu = metrics[k].mean()
                        sigma = metrics[k].std() + 1e-8
                        z = (r[k] - mu) / sigma
                        group_score = z
                        detail_parts.append(f"{k}: {r[k]:.4f} → z={z:+.3f} (single-metric mode)")
                    else:
                        for k in group_cfg["keys"]:
                            w = key_weights.get(k, 1.0)
                            mu = metrics[k].mean()
                            sigma = metrics[k].std() + 1e-8
                            z = (r[k] - mu) / sigma
                            group_score += z * w
                            weight_sum += w
                            detail_parts.append(f"{k}: {r[k]:.4f} → z={z:+.3f} (w={w})")
                        group_score /= weight_sum

                
                sample_score += group_cfg["weight"] * group_score
                detail_parts.append(f"[{group_name}_score={group_score:.3f}]")

            total_scores.append(sample_score)
            print(f"Sample {i+1:02d}: {' | '.join(detail_parts)} | total score = {sample_score:.3f}")

        print("=" * 80 + "\n")

        scores_tensor = torch.tensor(total_scores, dtype=torch.float32)
        mean_score = scores_tensor.mean()
        std_score = scores_tensor.std() + 1e-8
        advantages = [(s - mean_score) / std_score for s in total_scores]
        for i, adv in enumerate(advantages):
            sample_details[i]['advantage'] = adv.item()

        print(f"[Rank {rank}] Within-group normalization completed, advantage values: {[d['advantage'] for d in sample_details]}")

        
        for r in range(world_size):
            try:
                os.remove(output_base / f"video_paths_rank{r}_step{step_str}.json")
            except:
                pass
        if rank == 0:
            if vq_file.exists():
                os.remove(vq_file)
            if clap_file.exists():
                os.remove(clap_file)

        if dist.is_initialized():
            if "cpu" in rng_states:
                torch.random.set_rng_state(rng_states["cpu"])
            if "cuda" in rng_states:
                torch.cuda.set_rng_state_all(rng_states["cuda"])

        return my_videos, my_audios, sample_details

    def _generate_standard(
        self,
        config: GenerationConfig,
        device: torch.device,
        sample_index: int = 0,               
        save_trajectory: bool = False        
    ) -> tuple[Tensor, Tensor | None]:
        """Standard generation (text-to-video or image-to-video)."""
        # Get prompt embeddings (from cache or encode on-the-fly)
        v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg = self._get_prompt_embeddings(config, device)

        # Setup generator
        generator = torch.Generator(device=device).manual_seed(config.seed)

        # Create latent tools
        video_tools = self._create_video_latent_tools(config)
        audio_tools = self._create_audio_latent_tools(config) if config.generate_audio else None

        # Create initial states
        video_clean_state = video_tools.create_initial_state(device=device, dtype=torch.bfloat16)
        audio_clean_state = (
            audio_tools.create_initial_state(device=device, dtype=torch.bfloat16) if audio_tools else None
        )

        # Apply image conditioning if provided
        if config.condition_image is not None:
            video_clean_state = self._apply_image_conditioning(
                video_clean_state, config.condition_image, config, device
            )

        # Add noise
        noiser = GaussianNoiser(generator=generator)
        video_state = noiser(latent_state=video_clean_state, noise_scale=1.0)
        audio_state = noiser(latent_state=audio_clean_state, noise_scale=1.0) if audio_clean_state else None

        # Run denoising loop
        video_state, audio_state, video_log_probs, audio_log_probs = self._run_denoising(
            config=config,
            video_state=video_state,
            audio_state=audio_state,
            video_clean_state=video_clean_state,
            audio_clean_state=audio_clean_state,
            v_ctx_pos=v_ctx_pos,
            a_ctx_pos=a_ctx_pos,
            v_ctx_neg=v_ctx_neg,
            a_ctx_neg=a_ctx_neg,
            device=device,
            freeze_modality=config.freeze_modality,
            sample_index=sample_index,        
            save_trajectory=save_trajectory,  
        )

        # Decode outputs
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        video_output = self._decode_video(video_state, device, config.tiled_decoding)

        audio_output = None
        if audio_state is not None and audio_tools is not None:
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)
            audio_output = self._decode_audio(audio_state, device)

        return video_output, audio_output

    def _generate_with_reference(self, config: GenerationConfig, device: torch.device) -> tuple[Tensor, Tensor | None]:
        """Generate with reference video conditioning (IC-LoRA style).
        For IC-LoRA:
        - Reference video latents are concatenated with target latents
        - Reference latents have timestep=0 (clean, not denoised)
        - Target latents are denoised normally
        - If condition_image is also provided, the first frame of the target is conditioned
        - If include_reference_in_output is True, the preprocessed reference video
          is concatenated side-by-side with the generated video
        """
        # Get prompt embeddings (from cache or encode on-the-fly)
        v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg = self._get_prompt_embeddings(config, device)

        # Setup generator
        generator = torch.Generator(device=device).manual_seed(config.seed)

        # Preprocess and encode reference video
        ref_video_preprocessed = self._preprocess_reference_video(config)
        ref_latent, ref_positions = self._encode_video(ref_video_preprocessed, config.frame_rate, device)
        ref_seq_len = ref_latent.shape[1]

        # Scale reference positions to match target coordinate space
        # Position tensor shape: [B, 3, seq_len, 2] where dim 1 is (time, height, width)
        if config.reference_downscale_factor != 1:
            ref_positions = ref_positions.clone()
            ref_positions[:, 1, ...] *= config.reference_downscale_factor  # height axis
            ref_positions[:, 2, ...] *= config.reference_downscale_factor  # width axis
            # Time axis (index 0) remains unchanged

        # Create target video state
        video_tools = self._create_video_latent_tools(config)
        target_clean_state = video_tools.create_initial_state(device=device, dtype=torch.bfloat16)

        # Apply first-frame image conditioning to target if provided
        if config.condition_image is not None:
            target_clean_state = self._apply_image_conditioning(
                target_clean_state, config.condition_image, config, device
            )

        # Create combined state (reference + target)
        # denoise_mask shape is [B, seq_len, 1] after patchification
        ref_denoise_mask = torch.zeros(1, ref_seq_len, 1, device=device, dtype=torch.float32)
        combined_clean_state = LatentState(
            latent=torch.cat([ref_latent, target_clean_state.latent], dim=1),
            denoise_mask=torch.cat([ref_denoise_mask, target_clean_state.denoise_mask], dim=1),
            positions=torch.cat([ref_positions, target_clean_state.positions], dim=2),
            clean_latent=torch.cat([ref_latent, target_clean_state.clean_latent], dim=1),
        )

        # Add noise (only to the target portion via denoise_mask)
        noiser = GaussianNoiser(generator=generator)
        combined_state = noiser(latent_state=combined_clean_state, noise_scale=1.0)

        # Create audio state if needed
        audio_tools = self._create_audio_latent_tools(config) if config.generate_audio else None
        audio_clean_state = (
            audio_tools.create_initial_state(device=device, dtype=torch.bfloat16) if audio_tools else None
        )
        audio_state = noiser(latent_state=audio_clean_state, noise_scale=1.0) if audio_clean_state else None

        # Run denoising loop
        video_state, audio_state, video_log_probs, audio_log_probs = self._run_denoising(
            config=config,
            video_state=combined_state,
            audio_state=audio_state,
            video_clean_state=combined_clean_state,
            audio_clean_state=audio_clean_state,
            v_ctx_pos=v_ctx_pos,
            a_ctx_pos=a_ctx_pos,
            v_ctx_neg=v_ctx_neg,
            a_ctx_neg=a_ctx_neg,
            device=device,
            freeze_modality=config.freeze_modality,
        )

        # Extract target portion and decode
        target_latent = combined_state.latent[:, ref_seq_len:]
        video_output = self._decode_video_latent(target_latent, config, device)

        # Optionally concatenate original reference video side-by-side
        if config.include_reference_in_output:
            # Use preprocessed reference (already resized/cropped, in pixel space)
            # Convert from [B, C, F, H, W] to [C, F, H, W]
            ref_video_pixels = ref_video_preprocessed[0].cpu()
            # Normalize from [-1, 1] to [0, 1]
            ref_video_pixels = ((ref_video_pixels + 1.0) / 2.0).clamp(0.0, 1.0)
            video_output = self._concatenate_videos_side_by_side(ref_video_pixels, video_output)

        # Decode audio
        audio_output = None
        if audio_state is not None and audio_tools is not None:
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)
            audio_output = self._decode_audio(audio_state, device)

        return video_output, audio_output

    def _create_video_latent_tools(self, config: GenerationConfig) -> VideoLatentTools:
        """Create video latent tools for the given configuration."""
        pixel_shape = VideoPixelShape(
            batch=1,
            frames=config.num_frames,
            height=config.height,
            width=config.width,
            fps=config.frame_rate,
        )
        return VideoLatentTools(
            patchifier=self._video_patchifier,
            target_shape=VideoLatentShape.from_pixel_shape(shape=pixel_shape),
            fps=config.frame_rate,
            scale_factors=VIDEO_SCALE_FACTORS,
            causal_fix=True,
        )

    def _create_audio_latent_tools(self, config: GenerationConfig) -> AudioLatentTools:
        """Create audio latent tools for the given configuration."""
        return AudioLatentTools(
            patchifier=self._audio_patchifier,
            target_shape=AudioLatentShape.from_duration(batch=1, duration=config.num_frames / config.frame_rate),
        )

    def _apply_image_conditioning(
        self, video_state: LatentState, image: Tensor, config: GenerationConfig, device: torch.device
    ) -> LatentState:
        """Apply first-frame image conditioning to the video state."""
        # Encode the image
        encoded_image = self._encode_conditioning_image(image, config.height, config.width, device)

        # Patchify the encoded image (single frame)
        patchified_image = self._video_patchifier.patchify(encoded_image)  # [1, 1, C] -> [1, num_patches, C]
        num_image_tokens = patchified_image.shape[1]

        # Update the first frame tokens in the latent
        new_latent = video_state.latent.clone()
        new_latent[:, :num_image_tokens] = patchified_image.to(new_latent.dtype)

        # Update clean_latent as well (conditioning image is clean)
        new_clean_latent = video_state.clean_latent.clone()
        new_clean_latent[:, :num_image_tokens] = patchified_image.to(new_clean_latent.dtype)

        # Set denoise_mask to 0 for conditioned tokens (don't denoise them)
        new_denoise_mask = video_state.denoise_mask.clone()
        new_denoise_mask[:, :num_image_tokens] = 0.0

        return LatentState(
            latent=new_latent,
            denoise_mask=new_denoise_mask,
            positions=video_state.positions,
            clean_latent=new_clean_latent,
        )

    @staticmethod
    def _preprocess_reference_video(config: GenerationConfig) -> Tensor:
        """Preprocess reference video: resize, crop, and convert to model input format.
        When reference_downscale_factor > 1, the reference video is downscaled to a smaller
        resolution for more efficient inference. The positions will be scaled up later
        to match the target coordinate space.
        Args:
            config: Generation configuration
        Returns:
            Preprocessed video tensor [B, C, F, H, W] in [-1, 1] range
        """
        ref_video = config.reference_video  # [F, C, H, W] in [0, 1]
        scale_factor = config.reference_downscale_factor

        # Target dimensions for reference (scaled down if scale_factor > 1)
        target_height = config.height // scale_factor
        target_width = config.width // scale_factor

        # Validate scaled dimensions
        if target_height % 32 != 0 or target_width % 32 != 0:
            raise ValueError(
                f"Scaled reference dimensions ({target_height}x{target_width}) must be divisible by 32. "
                f"Original: {config.height}x{config.width}, scale_factor: {scale_factor}"
            )

        current_height, current_width = ref_video.shape[2:]

        # Resize maintaining aspect ratio and center crop if needed
        if current_height != target_height or current_width != target_width:
            aspect_ratio = current_width / current_height
            target_aspect_ratio = target_width / target_height

            if aspect_ratio > target_aspect_ratio:
                resize_height, resize_width = target_height, int(target_height * aspect_ratio)
            else:
                resize_height, resize_width = int(target_width / aspect_ratio), target_width

            ref_video = torch.nn.functional.interpolate(
                ref_video, size=(resize_height, resize_width), mode="bilinear", align_corners=False
            )

            # Center crop
            h_start = (resize_height - target_height) // 2
            w_start = (resize_width - target_width) // 2
            ref_video = ref_video[:, :, h_start : h_start + target_height, w_start : w_start + target_width]

        # Convert to [B, C, F, H, W] and trim to valid frame count (k*8 + 1)
        ref_video = rearrange(ref_video, "f c h w -> 1 c f h w")
        valid_frames = (ref_video.shape[2] - 1) // 8 * 8 + 1
        ref_video = ref_video[:, :, :valid_frames]

        # Convert to [-1, 1] range
        return ref_video * 2.0 - 1.0

    def _encode_video(self, video: Tensor, fps: float, device: torch.device) -> tuple[Tensor, Tensor]:
        """Encode video to patchified latents and compute positions.
        Args:
            video: Video tensor [B, C, F, H, W] in [-1, 1] range
            fps: Frame rate for temporal position scaling
            device: Device to run encoding on
        Returns:
            Tuple of (patchified_latents, positions)
        """
        video = video.to(device=device, dtype=torch.float32)

        # Encode with VAE
        self._vae_encoder.to(device)
        with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
            latents = self._vae_encoder(video)
        self._vae_encoder.to("cpu")

        latents = latents.to(torch.bfloat16)
        patchified = self._video_patchifier.patchify(latents)

        # Compute positions
        latent_shape = VideoLatentShape(
            batch=1,
            channels=latents.shape[1],
            frames=latents.shape[2],
            height=latents.shape[3],
            width=latents.shape[4],
        )
        latent_coords = self._video_patchifier.get_patch_grid_bounds(output_shape=latent_shape, device=device)
        positions = get_pixel_coords(latent_coords, scale_factors=VIDEO_SCALE_FACTORS, causal_fix=True)
        positions = positions.to(torch.bfloat16)
        positions[:, 0, ...] = positions[:, 0, ...] / fps

        return patchified, positions

    def _run_denoising(
        self,
        config: GenerationConfig,
        video_state: LatentState,
        audio_state: LatentState | None,
        video_clean_state: LatentState,
        audio_clean_state: LatentState | None,
        v_ctx_pos: Tensor,
        a_ctx_pos: Tensor,
        v_ctx_neg: Tensor | None,
        a_ctx_neg: Tensor | None,
        device: torch.device,
        freeze_modality: Literal["video", "audio"] | None = None,
        sample_index: int = 0,
        save_trajectory: bool = False,
    ) -> tuple[LatentState, LatentState | None, list[Tensor], list[Tensor] | None]:
        import os
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0

        # -------- Determine whether it is in "reference generation mode" ----------
        is_reference_generation = (
            freeze_modality is None
            and save_trajectory
            and not getattr(config, 'enable_sde', False)
        )

        scheduler = LTX2Scheduler()
        sigmas = scheduler.execute(steps=config.num_inference_steps).to(device).float()

        if getattr(config, 'enable_sde', False):
            stepper = FlowGRPOSDEDiffusionStep(
                noise_level=getattr(config, 'sde_noise_level', 0.1),
                sde_start_step=getattr(config, 'sde_start_step', 0),
                sde_end_step=getattr(config, 'sde_end_step', None),
            )
        else:
            stepper = EulerDiffusionStep()

        cfg_guider = CFGGuider(config.guidance_scale)
        stg_guider = STGGuider(config.stg_scale)
        collector = TrajectoryCollector(
            indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14],
            total_steps=config.num_inference_steps,
        )

        # Cache file path: independent for each rank
        cache_dir = str(Path(__file__).resolve().parents[4] / "outputs" / "trajectory")
        cache_path = os.path.join(cache_dir, f"validation_trajectory_cached_rank{rank}.pt")

        # In reference generation mode, cache is not used; in normal freezing mode, cache is used.
        use_cache = False
        if not is_reference_generation:
            use_cache = os.path.exists(cache_path) and freeze_modality is not None

        step_to_cache_idx = {}
        cached_video_latents = None
        cached_video_sigmas = None
        cached_audio_latents = None
        cached_audio_sigmas = None

        if use_cache:
            cache_data = torch.load(cache_path, map_location='cpu')
            data_dict = cache_data['data']
            cached_video_latents = data_dict.get('video_latent')
            cached_video_sigmas = data_dict.get('video_sigma')
            cached_audio_latents = data_dict.get('audio_latent')
            cached_audio_sigmas = data_dict.get('audio_sigma')
            cached_indices = cache_data['collected_indices']
            num_cached = len(cached_indices)
            for i in range(config.num_inference_steps):
                if i < num_cached:
                    step_to_cache_idx[i] = i
            if getattr(self, '_accelerator', None) is None or self._accelerator.is_main_process:
                print(f"[INFO] Rank {rank} loaded its own trajectory cache from {cache_path}")
        else:
            if freeze_modality is not None and (getattr(self, '_accelerator', None) is None or self._accelerator.is_main_process):
                print(f"[WARN] Cache file not found for rank {rank} at {cache_path}, cannot freeze {freeze_modality}.")

        # ========== generation loop ==========
        stg_perturbation_config = self._build_stg_perturbation_config(config) if stg_guider.enabled() else None

        video = Modality(enabled=True, latent=video_state.latent,
                         sigma=sigmas[0].repeat(video_state.latent.shape[0]),
                         timesteps=video_state.denoise_mask,
                         positions=video_state.positions, context=v_ctx_pos, context_mask=None)
        audio = None
        if audio_state is not None:
            audio = Modality(enabled=True, latent=audio_state.latent,
                             sigma=sigmas[0].repeat(audio_state.latent.shape[0]),
                             timesteps=audio_state.denoise_mask,
                             positions=audio_state.positions, context=a_ctx_pos, context_mask=None)

        self._transformer.to(device)
        x0_model = X0Model(self._transformer)
        video_log_probs = []
        audio_log_probs = [] if audio_state is not None else None

        with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
            for step_idx, sigma in enumerate(sigmas[:-1]):
             
                if use_cache and step_idx in step_to_cache_idx:
                    cache_idx = step_to_cache_idx[step_idx]
                    if freeze_modality == "video" and cached_video_latents is not None:
                        cached_latent = cached_video_latents[cache_idx].to(device=device, dtype=video_state.latent.dtype)
                        video_state = replace(video_state, latent=cached_latent)
                        video_sigma_val = cached_video_sigmas[cache_idx].to(device) if cached_video_sigmas is not None else sigma
                        audio_sigma_val = sigma
                    elif freeze_modality == "audio" and audio_state is not None and cached_audio_latents is not None:
                        cached_latent = cached_audio_latents[cache_idx].to(device=device, dtype=audio_state.latent.dtype)
                        audio_state = replace(audio_state, latent=cached_latent)
                        audio_sigma_val = cached_audio_sigmas[cache_idx].to(device) if cached_audio_sigmas is not None else sigma
                        video_sigma_val = sigma
                    else:
                        video_sigma_val = sigma
                        audio_sigma_val = sigma
                else:
                    video_sigma_val = sigma
                    audio_sigma_val = sigma

                # update modality
                video = replace(video, latent=video_state.latent,
                                sigma=video_sigma_val.repeat(video_state.latent.shape[0]),
                                timesteps=video_sigma_val * video_state.denoise_mask,
                                positions=video_state.positions)
                if audio is not None and audio_state is not None:
                    audio = replace(audio, latent=audio_state.latent,
                                    sigma=audio_sigma_val.repeat(audio_state.latent.shape[0]),
                                    timesteps=audio_sigma_val * audio_state.denoise_mask,
                                    positions=audio_state.positions)

               
                pos_video, pos_audio = x0_model(video=video, audio=audio, perturbations=None)
                denoised_video, denoised_audio = pos_video, pos_audio

                # CFG / STG
                if cfg_guider.enabled() and v_ctx_neg is not None:
                    video_neg = replace(video, context=v_ctx_neg)
                    audio_neg = replace(audio, context=a_ctx_neg) if audio is not None else None
                    neg_video, neg_audio = x0_model(video=video_neg, audio=audio_neg, perturbations=None)
                    denoised_video = denoised_video + cfg_guider.delta(pos_video, neg_video)
                    if audio is not None and denoised_audio is not None:
                        denoised_audio = denoised_audio + cfg_guider.delta(pos_audio, neg_audio)
                if stg_guider.enabled() and stg_perturbation_config is not None:
                    perturbed_video, perturbed_audio = x0_model(video=video, audio=audio, perturbations=stg_perturbation_config)
                    denoised_video = denoised_video + stg_guider.delta(pos_video, perturbed_video)
                    if audio is not None and denoised_audio is not None and perturbed_audio is not None:
                        denoised_audio = denoised_audio + stg_guider.delta(pos_audio, perturbed_audio)

               
                denoised_video = denoised_video * video_state.denoise_mask + video_clean_state.latent.float() * (1 - video_state.denoise_mask)
                if audio is not None and audio_state is not None and audio_clean_state is not None:
                    denoised_audio = denoised_audio * audio_state.denoise_mask + audio_clean_state.latent.float() * (1 - audio_state.denoise_mask)

                x_t_video = video.latent.detach().clone()
                x_t_audio = audio.latent.detach().clone() if audio is not None else None

                # ---------- Step update (distinguish between reference generation / normal mode) ----------
                if is_reference_generation:
                    # Reference generation mode: skip details, directly get next_sample
                    next_video_latent = stepper.step(
                        sample=video.latent,
                        denoised_sample=denoised_video,
                        sigmas=sigmas,
                        step_index=step_idx,
                        return_details=False,
                    )
                    video_state = replace(video_state, latent=next_video_latent)
                    video_log_probs.append(torch.zeros(video.latent.shape[0], device=device))
                    v_log_prob = None
                    video_dt_abs = video_sigma_t_eff = video_noise = None

                    if audio is not None:
                        next_audio_latent = stepper.step(
                            sample=audio.latent,
                            denoised_sample=denoised_audio,
                            sigmas=sigmas,
                            step_index=step_idx,
                            return_details=False,
                        )
                        audio_state = replace(audio_state, latent=next_audio_latent)
                        audio_log_probs.append(torch.zeros(audio.latent.shape[0], device=device))
                    a_log_prob = None
                    audio_dt_abs = audio_sigma_t_eff = audio_noise = None
                else:
                    # Normal freezing / no-freezing mode: details are needed for trajectory collection
                    if freeze_modality != "video":
                        details_v = stepper.step(
                            sample=video.latent,
                            denoised_sample=denoised_video,
                            sigmas=sigmas,
                            step_index=step_idx,
                            return_details=True
                        )
                        next_video_latent = details_v["next_sample"]
                        v_log_prob = details_v["log_prob"].detach().cpu()
                        video_dt_abs = details_v["dt_abs"].detach().cpu()
                        video_sigma_t_eff = details_v["sigma_t_effective"].detach().cpu()
                        video_noise = details_v["noise"].detach().cpu()
                        video_state = replace(video_state, latent=next_video_latent)
                        video_log_probs.append(v_log_prob)
                    else:
                        next_video_latent = None
                        v_log_prob = video_dt_abs = video_sigma_t_eff = video_noise = None

                    if freeze_modality != "audio" and audio is not None:
                        details_a = stepper.step(
                            sample=audio.latent,
                            denoised_sample=denoised_audio,
                            sigmas=sigmas,
                            step_index=step_idx,
                            return_details=True
                        )
                        next_audio_latent = details_a["next_sample"]
                        a_log_prob = details_a["log_prob"].detach().cpu()
                        audio_dt_abs = details_a["dt_abs"].detach().cpu()
                        audio_sigma_t_eff = details_a["sigma_t_effective"].detach().cpu()
                        audio_noise = details_a["noise"].detach().cpu()
                        audio_state = replace(audio_state, latent=next_audio_latent)
                        audio_log_probs.append(a_log_prob)
                    else:
                        next_audio_latent = None
                        a_log_prob = audio_dt_abs = audio_sigma_t_eff = audio_noise = None

               
                collector.collect(
                    step_idx=step_idx,
                    video_latent=x_t_video.cpu(),
                    next_video_latent=next_video_latent.cpu() if next_video_latent is not None else None,
                    video_log_prob=v_log_prob,
                    video_x0_pred=pos_video.detach().cpu(),
                    video_sigma=video_sigma_val.detach().cpu(),
                    video_dt_abs=video_dt_abs,
                    video_sigma_t_eff=video_sigma_t_eff,
                    video_noise=video_noise,
                    video_context=video.context.detach().cpu(),
                    video_positions=video.positions.detach().cpu(),
                    audio_latent=x_t_audio.cpu() if x_t_audio is not None else None,
                    next_audio_latent=next_audio_latent.cpu() if next_audio_latent is not None else None,
                    audio_log_prob=a_log_prob,
                    audio_x0_pred=pos_audio.detach().cpu() if pos_audio is not None else None,
                    audio_sigma=audio_sigma_val.detach().cpu() if audio_state is not None else None,
                    audio_dt_abs=audio_dt_abs,
                    audio_sigma_t_eff=audio_sigma_t_eff,
                    audio_noise=audio_noise,
                    audio_context=audio.context.detach().cpu() if audio is not None else None,
                    audio_positions=audio.positions.detach().cpu() if audio is not None else None,
                )
                if self._sampling_context is not None:
                    self._sampling_context.advance_step()

        # ========== Save trajectory ==========
        if save_trajectory:
            traj_data = collector.get_result()
            if is_reference_generation:
                # Reference generation mode: save to rank-specific cache file
                collected_indices = list(range(config.num_inference_steps))
                data_dict = {key: traj_data[key] for key in traj_data}
                cache_data = {'data': data_dict, 'collected_indices': collected_indices}
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(cache_data, cache_path)
                if rank == 0:
                    print(f"💾 Reference trajectory cached for all ranks (rank {rank} saved to {cache_path})")
            else:
                # Normal training trajectory saving
                save_dir = Path(__file__).resolve().parents[4] / "outputs" / "sample_trajectory_train"
                save_dir.mkdir(parents=True, exist_ok=True)
                file_path = save_dir / f"traj_rank{rank}_sample{sample_index}.pt"
                torch.save(traj_data, file_path)
                if rank == 0:
                    print(f"💾 Saved trajectory to {file_path}")

        return video_state, audio_state, video_log_probs, audio_log_probs

    @staticmethod
    def _build_stg_perturbation_config(config: GenerationConfig) -> BatchedPerturbationConfig:
        """Build the perturbation config for STG based on the stg_mode."""
        # Always skip video self-attention for STG
        perturbations: list[Perturbation] = [
            Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=config.stg_blocks)
        ]

        # Optionally also skip audio self-attention (stg_av mode)
        if config.stg_mode == "stg_av":
            perturbations.append(Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=config.stg_blocks))

        perturbation_config = PerturbationConfig(perturbations=perturbations)
        # Batch size is 1 for validation
        return BatchedPerturbationConfig(perturbations=[perturbation_config])

    def _decode_video_latent(self, latent: Tensor, config: GenerationConfig, device: torch.device) -> Tensor:
        """Decode patchified video latent to pixel space."""
        # Unpatchify
        latent_frames = config.num_frames // VIDEO_SCALE_FACTORS.time + 1
        latent_height = config.height // VIDEO_SCALE_FACTORS.height
        latent_width = config.width // VIDEO_SCALE_FACTORS.width

        unpatchified = self._video_patchifier.unpatchify(
            latent,
            output_shape=VideoLatentShape(
                height=latent_height,
                width=latent_width,
                frames=latent_frames,
                batch=1,
                channels=128,
            ),
        )

        # Decode - ensure bfloat16 to match decoder weights
        self._vae_decoder.to(device)
        unpatchified = unpatchified.to(dtype=torch.bfloat16)
        tiled_config = config.tiled_decoding

        if tiled_config is not None and tiled_config.enabled:
            # Use tiled decoding for reduced VRAM
            tiling_config = TilingConfig(
                spatial_config=SpatialTilingConfig(
                    tile_size_in_pixels=tiled_config.tile_size_pixels,
                    tile_overlap_in_pixels=tiled_config.tile_overlap_pixels,
                ),
                temporal_config=TemporalTilingConfig(
                    tile_size_in_frames=tiled_config.tile_size_frames,
                    tile_overlap_in_frames=tiled_config.tile_overlap_frames,
                ),
            )
            chunks = []
            for video_chunk in self._vae_decoder.tiled_decode(
                unpatchified,
                tiling_config=tiling_config,
            ):
                chunks.append(video_chunk)
            decoded_video = torch.cat(chunks, dim=2)
        else:
            # Standard full decoding
            decoded_video = self._vae_decoder(unpatchified)

        decoded_video = ((decoded_video + 1.0) / 2.0).clamp(0.0, 1.0)
        self._vae_decoder.to("cpu")

        return decoded_video[0].float().cpu()

    def _validate_config(self, config: GenerationConfig) -> None:
        """Validate generation configuration."""
        if config.height % 32 != 0 or config.width % 32 != 0:
            raise ValueError(f"height and width must be divisible by 32, got {config.height}x{config.width}")
        if config.num_frames % 8 != 1:
            raise ValueError(f"num_frames must satisfy num_frames % 8 == 1, got {config.num_frames}")
        if config.generate_audio and (self._audio_decoder is None or self._vocoder is None):
            raise ValueError("Audio generation requires audio_decoder and vocoder")
        if config.condition_image is not None and self._vae_encoder is None:
            raise ValueError("Image conditioning requires vae_encoder")
        if config.reference_video is not None and self._vae_encoder is None:
            raise ValueError("Reference video conditioning requires vae_encoder")

        # Validate prompt embedding source
        if config.cached_embeddings is None and self._text_encoder is None:
            raise ValueError("Either text_encoder or config.cached_embeddings must be provided")
        if config.cached_embeddings is None and self._embeddings_processor is None:
            raise ValueError("embeddings_processor is required when encoding prompts on-the-fly")

    def _get_prompt_embeddings(
        self, config: GenerationConfig, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """Get prompt embeddings from config cache or encode on-the-fly."""
        if config.cached_embeddings is not None:
            # Use pre-computed embeddings from config
            cached = config.cached_embeddings
            v_ctx_pos = cached.video_context_positive.to(device)
            a_ctx_pos = cached.audio_context_positive.to(device)
            v_ctx_neg = cached.video_context_negative.to(device) if cached.video_context_negative is not None else None
            a_ctx_neg = cached.audio_context_negative.to(device) if cached.audio_context_negative is not None else None
            return v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg

        # Fall back to encoding on-the-fly
        return self._encode_prompts(config, device)

    def _encode_prompts(
        self, config: GenerationConfig, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        """Encode positive and negative prompts using the text encoder + embeddings processor."""
        self._text_encoder.to(device)
        self._embeddings_processor.to(device)

        pos_hs, pos_mask = self._text_encoder.encode(config.prompt)
        pos_out = self._embeddings_processor.process_hidden_states(pos_hs, pos_mask)
        v_ctx_pos, a_ctx_pos = pos_out.video_encoding, pos_out.audio_encoding

        v_ctx_neg, a_ctx_neg = None, None
        if config.guidance_scale != 1.0:
            neg_hs, neg_mask = self._text_encoder.encode(config.negative_prompt)
            neg_out = self._embeddings_processor.process_hidden_states(neg_hs, neg_mask)
            v_ctx_neg, a_ctx_neg = neg_out.video_encoding, neg_out.audio_encoding

        # Move the base Gemma model to CPU
        self._text_encoder.model.to("cpu")

        return v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg

    def _decode_video(
        self, video_state: LatentState, device: torch.device, tiled_config: TiledDecodingConfig | None = None
    ) -> Tensor:
        """Decode video latents to pixel space.
        Args:
            video_state: Video latent state to decode
            device: Device to run decoding on
            tiled_config: Optional tiled decoding configuration for reduced VRAM usage
        Returns:
            Decoded video tensor [C, F, H, W] in [0, 1] range
        """
        self._vae_decoder.to(device)
        # Ensure latent is bfloat16 to match decoder weights
        latent = video_state.latent.to(dtype=torch.bfloat16)

        if tiled_config is not None and tiled_config.enabled:
            # Use tiled decoding for reduced VRAM
            tiling_config = TilingConfig(
                spatial_config=SpatialTilingConfig(
                    tile_size_in_pixels=tiled_config.tile_size_pixels,
                    tile_overlap_in_pixels=tiled_config.tile_overlap_pixels,
                ),
                temporal_config=TemporalTilingConfig(
                    tile_size_in_frames=tiled_config.tile_size_frames,
                    tile_overlap_in_frames=tiled_config.tile_overlap_frames,
                ),
            )
            chunks = []
            for video_chunk in self._vae_decoder.tiled_decode(
                latent,
                tiling_config=tiling_config,
            ):
                chunks.append(video_chunk)
            decoded_video = torch.cat(chunks, dim=2)
        else:
            # Standard full decoding
            decoded_video = self._vae_decoder(latent)

        decoded_video = ((decoded_video + 1.0) / 2.0).clamp(0.0, 1.0)
        self._vae_decoder.to("cpu")
        return decoded_video[0].float().cpu()

    def _decode_audio(self, audio_state: LatentState, device: torch.device) -> Tensor:
        """Decode audio latents to waveform."""
        self._audio_decoder.to(device)
        # Ensure latent is bfloat16 to match decoder weights
        latent = audio_state.latent.to(dtype=torch.bfloat16)
        decoded_audio = self._audio_decoder(latent)
        self._audio_decoder.to("cpu")

        self._vocoder.to(device)
        audio_waveform = self._vocoder(decoded_audio)
        self._vocoder.to("cpu")

        return audio_waveform.squeeze(0).float().cpu()

    @staticmethod
    def _concatenate_videos_side_by_side(left_video: Tensor, right_video: Tensor) -> Tensor:
        """Concatenate two videos side-by-side (horizontally).
        If the videos have different frame counts, the shorter one is padded with
        its last frame repeated.
        Args:
            left_video: Left video tensor [C, F1, H1, W1] in [0, 1]
            right_video: Right video tensor [C, F2, H2, W2] in [0, 1]
        Returns:
            Concatenated video tensor [C, max(F1,F2), H2, W1_scaled+W2] in [0, 1]
        """
        left_height, left_width = left_video.shape[2], left_video.shape[3]
        right_height = right_video.shape[2]

        # Resize left video to match right video's height if needed
        if left_height != right_height:
            # Scale width proportionally to maintain aspect ratio
            scale = right_height / left_height
            new_width = int(left_width * scale)
            # Interpolate expects [N, C, H, W], we have [C, F, H, W]
            # Reshape to [C*F, 1, H, W] -> interpolate -> reshape back
            c, f, h, w = left_video.shape
            left_video = left_video.reshape(c * f, 1, h, w)
            left_video = torch.nn.functional.interpolate(
                left_video, size=(right_height, new_width), mode="bilinear", align_corners=False
            )
            left_video = left_video.reshape(c, f, right_height, new_width)

        left_frames = left_video.shape[1]
        right_frames = right_video.shape[1]

        # Pad shorter video by repeating last frame
        if left_frames < right_frames:
            padding = left_video[:, -1:, :, :].expand(-1, right_frames - left_frames, -1, -1)
            left_video = torch.cat([left_video, padding], dim=1)
        elif right_frames < left_frames:
            padding = right_video[:, -1:, :, :].expand(-1, left_frames - right_frames, -1, -1)
            right_video = torch.cat([right_video, padding], dim=1)

        # Concatenate along width dimension
        return torch.cat([left_video, right_video], dim=3)

    def _encode_conditioning_image(
        self,
        image: Tensor,
        target_height: int,
        target_width: int,
        device: torch.device,
    ) -> Tensor:
        """Encode a conditioning image to latent space.
        The image is resized to cover the target dimensions while preserving aspect ratio,
        then center-cropped to exactly match the target size.
        """
        # image is [C, H, W] in [0, 1]  # noqa: ERA001
        current_height, current_width = image.shape[1:]

        # Resize maintaining aspect ratio (cover target, then center crop)
        if current_height != target_height or current_width != target_width:
            aspect_ratio = current_width / current_height
            target_aspect_ratio = target_width / target_height

            if aspect_ratio > target_aspect_ratio:
                # Image is wider than target - resize to match height, crop width
                resize_height = target_height
                resize_width = int(target_height * aspect_ratio)
            else:
                # Image is taller than target - resize to match width, crop height
                resize_height = int(target_width / aspect_ratio)
                resize_width = target_width

            image = rearrange(image, "c h w -> 1 c h w")
            image = torch.nn.functional.interpolate(
                image, size=(resize_height, resize_width), mode="bilinear", align_corners=False
            )

            # Center crop to target dimensions
            h_start = (resize_height - target_height) // 2
            w_start = (resize_width - target_width) // 2
            image = image[:, :, h_start : h_start + target_height, w_start : w_start + target_width]
        else:
            image = rearrange(image, "c h w -> 1 c h w")

        # Add frame dimension and convert to [-1, 1]
        image = rearrange(image, "b c h w -> b c 1 h w")
        image = (image * 2.0 - 1.0).to(device=device, dtype=torch.float32)

        # Encode
        self._vae_encoder.to(device)
        with torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
            encoded = self._vae_encoder(image)
        self._vae_encoder.to("cpu")

        return encoded