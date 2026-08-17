import os
import sys
import gc
import math
import warnings
import torch
import numpy as np


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

def _load_audio_aes_model(device):
    """Load the local audiobox-aesthetics model"""
    from audiobox_aesthetics.infer import AesPredictor
    predictor = AesPredictor(checkpoint_pth=LOCAL_AUDIO_AES_PATH)
    predictor.model = predictor.model.to(device)
    predictor.device = device
    return predictor

_MODEL_CACHE = {}
_JAVISDIT_PATH_ADDED = False

def _ensure_javisdit_path():
    global _JAVISDIT_PATH_ADDED
    if not _JAVISDIT_PATH_ADDED:
        javisdit_root = ".../AV-GRPO/JavisDiT"
        if javisdit_root not in sys.path:
            sys.path.insert(0, javisdit_root)
        _JAVISDIT_PATH_ADDED = True

def _frame_transform(frame, clip_preprocess):
    import cv2
    from PIL import Image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = clip_preprocess(frame)
    return frame

def _compute_vq_batch(video_paths, prompts, device):
    """Batch compute visual_quality and motion_quality, [force clear GPU memory before computation]"""
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    gc.collect()
    if torch.distributed.is_initialized():
        torch.cuda.ipc_collect()
    
    
    FIX_PATH = ".../AV-GRPO/JavisDiT"
    sys.path.insert(0, FIX_PATH)

 
    from eval.javisbench.src.dataset import create_dataloader as create_vq_dataloader
    from eval.javisbench.src.VideoAlign.inference import VideoVLMRewardInference


    sys.path.remove(FIX_PATH)


    LOCAL_VIDEO_REWARD = ".../AV-GRPO/JavisDiT/checkpoints/VideoReward"
    
    predictor = None
    try:
        predictor = VideoVLMRewardInference(
            load_from_pretrained=LOCAL_VIDEO_REWARD,
            device=device
        )
        
        results = []
        for vp, prompt in zip(video_paths, prompts):
            dataloader = None
            try:
                pred_video_list = [os.path.abspath(vp)]
                prompt_list = [prompt]
                dataloader = create_vq_dataloader(
                    metric='video-quality',
                    video_path_list=pred_video_list,
                    prompt_list=prompt_list,
                    data_config=predictor.data_config,
                    processor=predictor.processor,
                    batch_size=1,
                )
                vq_val = mq_val = float('nan')
                with torch.no_grad():
                    for batch in dataloader:
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        outputs = predictor.model(return_dict=True, **batch)["logits"]
                        outputs = predictor.post_process(outputs, use_norm=False)
                        vq_val = outputs[0]['VQ']
                        mq_val = outputs[0]['MQ']
                        
                        del batch, outputs
                        torch.cuda.empty_cache()
                if not isinstance(vq_val, (int, float)) or math.isnan(vq_val):
                    vq_val = float('nan')
                if not isinstance(mq_val, (int, float)) or math.isnan(mq_val):
                    mq_val = float('nan')
            except Exception as e:
                print(f"  ❌ Visual Quality Computation Fails: {e}")
                vq_val = float('nan')
                mq_val = float('nan')
            finally:
                if dataloader is not None:
                    del dataloader
                torch.cuda.empty_cache()
            
            results.append({'visual_quality': vq_val, 'motion_quality': mq_val})
        return results
    finally:
        if predictor is not None:
            del predictor
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        gc.collect()
        if torch.distributed.is_initialized():
            torch.cuda.ipc_collect()

def _compute_clap_batch(audio_paths, prompts, device):
    
    from transformers import AutoProcessor, ClapModel
    import torchaudio as ta

    clap_model = ClapModel.from_pretrained(
        "laion/clap-htsat-unfused",
        local_files_only=True,
    ).to(device).eval()
    clap_processor = AutoProcessor.from_pretrained(
        "laion/clap-htsat-unfused",
        local_files_only=True
    )

    results = []
    try:
        for audio_path, prompt in zip(audio_paths, prompts):
            try:
                audio_wav, sr = ta.load(audio_path)
                if audio_wav.shape[0] > 1:
                    audio_wav = audio_wav.mean(dim=0, keepdim=True)
                if sr != 48000:
                    resampler = ta.transforms.Resample(sr, 48000)
                    audio_wav = resampler(audio_wav)
                audio_wav = audio_wav.squeeze(0)
                max_len = 48000 * 10
                if audio_wav.shape[0] > max_len:
                    audio_wav = audio_wav[:max_len]

                inputs = clap_processor(
                    text=[prompt],
                    audios=audio_wav.numpy(),
                    return_tensors="pt",
                    padding=True,
                    sampling_rate=48000,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = clap_model(**inputs)
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                score = cos(outputs.text_embeds, outputs.audio_embeds).mean().item()
                del audio_wav, inputs, outputs, cos
            except Exception as e:
                print(f"  ❌ CLAP computation failed: {e}")
                score = float('nan')
            results.append(score)
            torch.cuda.empty_cache()
    finally:
        del clap_model, clap_processor
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.cuda.ipc_collect()
        gc.collect()

    return results

def _compute_other_metrics(video_paths, audio_paths, prompts, freeze_modality, device, vq_list, clap_scores=None):
    import cv2  
    from PIL import Image
    _ensure_javisdit_path()
    need_video = (freeze_modality == "audio" or freeze_modality is None)
    need_audio = (freeze_modality == "video" or freeze_modality is None)

    num_samples = len(video_paths)
    sample_details = []
    for i in range(num_samples):
        sample_details.append({
            "index": i,
            "visual_quality": vq_list[i]['visual_quality'],
            "motion_quality": vq_list[i]['motion_quality'],
            "clip": 0.0,
            "audio_quality": 0.0,
            "clap": 0.0,
            "desync": 0.0,
        })

   
    clip_model = None
    synchformer = None
    sync_grid = None
    sync_mel_spectrogram = None
    aq_predictor = None
    clap_model_local = None
    clap_processor_local = None

    # load CLIP
    if need_video:
        import clip as clip_module
        clip_model, clip_preprocess = clip_module.load("ViT-B/32", device=device)

    # load Synchformer
    try:
        print("🔄 loading Synchformer model...")
        from eval.javisbench.src.synchformer.synchformer import Synchformer as Synch, make_class_grid
        LOCAL_SYNCHFORMER = ".../AV-GRPO/JavisDiT/checkpoints/synchformer_state_dict.pth"
        synchformer = Synch().to(device).eval()
        sd = torch.load(LOCAL_SYNCHFORMER, weights_only=True)
        synchformer.load_state_dict(sd)
        sync_grid = make_class_grid(-2, 2, 21)
        import torchaudio
        sync_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, win_length=400, hop_length=160,
            n_fft=1024, n_mels=128,
        ).to(device)
        print("✅ Synchformer loaded successfully")
    except Exception as e:
        print(f"❌ Synchformer failed to load: {e}")

    # load audio model
    if need_audio:
        from audiobox_aesthetics.infer import initialize_predictor as init_aq
        aq_predictor = init_aq()
        aq_predictor.model = aq_predictor.model.to(device)
        aq_predictor.device = device
        from transformers import AutoProcessor, ClapModel
        clap_model_local = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval().to(device)
        clap_processor_local = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

    try:
        from einops import rearrange
        from eval.javisbench.src.utils import pad_or_truncate

        for idx in range(num_samples):
            video_path = video_paths[idx]
            audio_path = audio_paths[idx]
            prompt = prompts[idx]
            print(f"\n--- Sample {idx+1}/{num_samples} (Rank local) ---")

            # CLIP computation
            if need_video and video_path is not None:
                clip_dataloader = None
                try:
                    import clip as clip_module
                    _, local_preprocess = clip_module.load("ViT-B/32", device=device)

                    def _local_frame_transform(frame):
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = Image.fromarray(frame)
                        frame = local_preprocess(frame)
                        return frame

                    from eval.javisbench.src.dataset import create_dataloader as create_clip_dataloader
                    clip_dataloader = create_clip_dataloader(
                        metric='clip-score',
                        video_path_list=[video_path],
                        prompt_list=[prompt],
                        num_frames=48,
                        frame_transform=_local_frame_transform,
                        batch_size=1
                    )
                    clip_val = float('nan')
                    for frames, prompts_batch in clip_dataloader:
                        frames = frames.to(device)
                        text = clip_module.tokenize(prompts_batch, truncate=True).to(device)
                        with torch.no_grad():
                            text_features = clip_model.encode_text(text)
                            image_features = clip_model.encode_image(frames[0])
                            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                            scores = (image_features @ text_features.T).squeeze()
                            mean_score = scores.mean().item()
                        clip_val = mean_score
                        del frames, text, text_features, image_features, scores
                    sample_details[idx]['clip'] = clip_val
                    print(f"  CLIP: {clip_val:.4f}")
                except Exception as e:
                    print(f"  ❌ CLIP fails: {e}")
                finally:
                    if clip_dataloader is not None:
                        del clip_dataloader
                    torch.cuda.empty_cache()

            # Desync computation
            if synchformer is not None and video_path is not None and audio_path is not None and os.path.exists(audio_path):
                desync_dataloader = None
                try:
                    from eval.javisbench.src.dataset import create_dataloader as create_desync_dataloader
                    desync_dataloader = create_desync_dataloader(
                        metric='desync-score',
                        video_path_list=[video_path],
                        audio_path_list=[audio_path],
                        batch_size=1, max_length_s=4
                    )
                    for video, audio in desync_dataloader:
                        video, audio = video.to(device), audio.to(device)
                        b, t, c, h, w = video.shape
                        segment_size = 16
                        step_size = 8
                        num_segments = (t - segment_size) // step_size + 1
                        segments = []
                        for i in range(num_segments):
                            segments.append(video[:, i * step_size:i * step_size + segment_size])
                        vx = torch.stack(segments, dim=1)
                        vx = rearrange(vx, 'b s t c h w -> (b s) 1 t c h w')
                        vx = synchformer.extract_vfeats(vx)
                        vx = rearrange(vx, '(b s) 1 t d -> b s t d', b=b)

                        _, at = audio.shape
                        segment_size_a = 10240
                        step_size_a = 10240 // 2
                        num_segments_a = (at - segment_size_a) // step_size_a + 1
                        segments_a = []
                        for i in range(num_segments_a):
                            segments_a.append(audio[:, i * step_size_a:i * step_size_a + segment_size_a])
                        ax = torch.stack(segments_a, dim=1)
                        ax_spec = torch.log(sync_mel_spectrogram(ax) + 1e-6)
                        ax_spec = pad_or_truncate(ax_spec, 66)
                        if ax_spec.device != device:
                            ax_spec = ax_spec.to(device)
                        mean, std = -4.2677393, 4.5689974
                        ax_spec = (ax_spec - mean) / (2 * std)
                        ax_feat = synchformer.extract_afeats(ax_spec.unsqueeze(2))

                        batch_sync_scores = []
                        frame_num = vx.shape[1]
                        seg_size = 14
                        seg_num = math.ceil(frame_num / seg_size)
                        for si in range(seg_num):
                            fstart = si * seg_size
                            fend = min((si + 1) * seg_size, frame_num)
                            vx_seg = vx[:, fstart:fend]
                            ax_seg = ax_feat[:, fstart:fend]
                            flen = fend - fstart
                            delta = seg_size - flen
                            if delta > 0:
                                if si == 0:
                                    repeat = math.ceil(delta / flen)
                                    vpad = vx_seg.repeat(1, repeat, *([1]*(vx_seg.dim()-2)))[:, :delta]
                                    vx_seg = torch.cat((vx_seg, vpad), dim=1)
                                    apad = ax_seg.repeat(1, repeat, *([1]*(ax_seg.dim()-2)))[:, :delta]
                                    ax_seg = torch.cat((ax_seg, apad), dim=1)
                                else:
                                    vx_seg = vx[:, -seg_size:]
                                    ax_seg = ax_feat[:, -seg_size:]
                            logits = synchformer.compare_v_a(vx_seg, ax_seg)
                            top_id = torch.argmax(logits, dim=-1).cpu().numpy()
                            for j in range(vx_seg.shape[0]):
                                batch_sync_scores.append(abs(sync_grid[top_id[j]].item()))
                        batch_sync_scores = torch.tensor(batch_sync_scores)
                        batch_sync_scores = batch_sync_scores.reshape(b, -1).mean(dim=1)
                        desync_val = batch_sync_scores[0].item()
                    sample_details[idx]['desync'] = desync_val
                    print(f"  Desync: {desync_val:.4f}")
                    del video, audio, vx, ax_spec, ax_feat, batch_sync_scores, logits
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"  ❌ desync fails: {e}")
                finally:
                    if desync_dataloader is not None:
                        del desync_dataloader
                    torch.cuda.empty_cache()

            # Audio Quality computation
            if need_audio and audio_path is not None:
                aq_dataloader = None
                try:
                    from eval.javisbench.src.dataset import create_dataloader as create_vq_dataloader
                    aq_dataloader = create_vq_dataloader(
                        metric='audio-quality',
                        audio_path_list=[audio_path],
                        prompt_list=[prompt],
                        sr=16000, backend="torchaudio", mono=True,
                        keepdim=True, norm=False, resample=True,
                        max_audio_len_s=4.0, batch_size=1,
                    )
                    for audios, _ in aq_dataloader:
                        batch = [{"path": wav, "sample_rate": 16000} for wav in audios]
                        outputs = aq_predictor.forward(batch)
                        aq_val = np.mean([list(o.values()) for o in outputs])
                    sample_details[idx]['audio_quality'] = aq_val
                    print(f"  Audio Quality: {aq_val:.4f}")
                    del audios, batch, outputs
                except Exception as e:
                    print(f"  ❌ Audio Quality fails: {e}")
                finally:
                    if aq_dataloader is not None:
                        del aq_dataloader
                    torch.cuda.empty_cache()

            # CLAP computation
            if clap_scores is not None and idx < len(clap_scores):
                sample_details[idx]['clap'] = clap_scores[idx]
                print(f"  CLAP: {clap_scores[idx]:.4f} (computation)")

            if need_video and video_path is not None and os.path.exists(video_path):
                try:
                    oe_max_ratio = compute_overexposure(video_path, threshold=240)
                    sample_details[idx]['overexposure'] = oe_max_ratio
                    print(f"  Overexposure max ratio: {oe_max_ratio:.4f}")
                except Exception as e:
                    print(f"  ❌ Overexposure fails: {e}")
                    sample_details[idx]['overexposure'] = 0.0
            # =====================================
    finally:
        pass

    # final cleanup of all models
    if clip_model is not None:
        del clip_model
    if synchformer is not None:
        del synchformer
        del sync_grid, sync_mel_spectrogram
    if aq_predictor is not None:
        del aq_predictor
    if clap_model_local is not None:
        del clap_model_local
    if clap_processor_local is not None:
        del clap_processor_local

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.cuda.ipc_collect()
    gc.collect()

    return sample_details

def _run_vq_subprocess(all_video_paths, all_prompts, output_json, device_str):
    """Independent subprocess computes VQ, releasing all GPU memory upon exit"""


    import subprocess, json
    tmp_json = output_json + ".input"
    with open(tmp_json, 'w', encoding='utf-8') as f:
        json.dump({
            "video_paths": all_video_paths,
            "prompts": all_prompts,
        }, f)


    cmd = [
        sys.executable, "-c",
        """
import os
import sys
import json
import torch

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


sys.path.insert(0, ".../AV-GRPO/JavisDiT")
from ltx_trainer.reward_computation import _compute_vq_batch


with open(sys.argv[1], 'r', encoding='utf-8') as f:
    inp = json.load(f)
device = torch.device('cuda:0')
results = _compute_vq_batch(inp['video_paths'], inp['prompts'], device)


with open(sys.argv[2], 'w', encoding='utf-8') as f:
    json.dump(results, f)
""",
        tmp_json,
        output_json
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    proc = subprocess.Popen(cmd, env=env)
    proc.communicate()
    

    try:
        os.remove(tmp_json)
    except:
        pass

def compute_overexposure(video_path, threshold=240):
    """
    Compute the maximum overexposed pixel ratio of the video (the maximum overexposed area of a single frame)

    Args:
    video_path: Path to the video file
    threshold: Overexposure brightness threshold, 0-255, default 240 (close to pure white)

    Returns:
    float: Maximum overexposure ratio (0~1), returns 0.0 if the video cannot be read
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0

    max_ratio = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ratio = np.mean(gray > threshold)
            if ratio > max_ratio:
                max_ratio = ratio
    finally:
        cap.release()

    return float(max_ratio)