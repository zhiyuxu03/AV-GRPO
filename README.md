# 🎬 AV-GRPO


[![Model](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/Dr-Loser/AV-GRPO)
[![Paper](https://img.shields.io/badge/Paper-PDF-EC1C24?logo=adobeacrobatreader&logoColor=white)]()


## 📋 Overview

AV-GRPO is the first GRPO framework designed for joint audio-video generation models. It enables full-parameter training or LoRA training of the 22B LTX-2.3 model on just **8 A800 GPUs**.

---

## 🚀 Training

### 📦 Step 1: Install the evaluators required for computing audio-video sample rewards during training

```bash
cd /path/to/AV-GRPO/JavisDiT
pip install -r requirements/requirements-eval.txt
```

> 💡 All necessary pre-trained models will be automatically downloaded to `./checkpoints`.

### 📥 Step 2: Download LTX-2.3 related weights

```bash
hf download google/gemma-3-12b-it-qat-q4_0-unquantized
hf download Lightricks/LTX-2.3
```

### 🔧 Step 3: Install required packages

```bash
cd /path/to/AV-GRPO
pip install -r requirements.txt
```

### 🔍 Step 4: Replace paths in the following files

Use a search command to find `.../AV-GRPO/` in the files below and replace all occurrences with your actual file path. **Make sure not to miss any.**

| File | Description |
|------|-------------|
| `.../AV-GRPO/packages/ltx-trainer/src/ltx_trainer/trainer.py` | Training loop |
| `.../AV-GRPO/packages/ltx-trainer/src/ltx_trainer/validation_sampler.py` | Sampler |
| `.../AV-GRPO/packages/ltx-trainer/src/ltx_trainer/reward_computation.py` | Reward computation |
| `.../AV-GRPO/packages/ltx-trainer/configs/ltx2_av_lora.yaml` | Training config |

### 📝 Step 5: Configure WandB 

Enter your WandB information at lines 367–381 of `.../AV-GRPO/packages/ltx-trainer/src/ltx_trainer/trainer.py` to log training status. Alternatively, you can comment out all WandB logging.

## 📊 Dataset

Our training dataset is already available at `/AV-GRPO/Dataset/dataset.json`. No separate download is required.


### 🏃 Step 6: Launch training

```bash
cd /path/to/AV-GRPO/packages/ltx-trainer/scripts
python3 -m accelerate.commands.launch --config_file /path/to/AV-GRPO/packages/ltx-trainer/configs/accelerate/fsdp_compile.yaml train.py /path/to/AV-GRPO/packages/ltx-trainer/configs/ltx2_av_lora.yaml
```

> ⚠️ **Note:** All `/path/to/` above should be replaced with your actual paths.

### 🔗 Step 7: Process and merge weights after training

**Full fine-tuning:**

```bash
cd /path/to/AV-GRPO/packages/ltx-trainer/scripts
python merge_full.py
```

**LoRA:**

```bash
cd /path/to/AV-GRPO/packages/ltx-trainer/scripts
python merge_lora.py
```

---

## 🎯 Inference

### 📥 Download our weights

```bash
huggingface-cli download --resume-download Dr-Loser/AV-GRPO AV-GRPO_FULL.safetensors
huggingface-cli download --resume-download Dr-Loser/AV-GRPO AV-GRPO_lora.safetensors
```

### ▶️ Run inference

```bash
cd /path/to/LTX-2/packages/ltx-trainer/scripts
```

```bash
python3 inference.py \
    --checkpoint CHECKPOINT_PATH \
    --text-encoder-path TEXT_ENCODER_PATH \
    --prompt "your prompt here" \
    --output /path/to/output
```
## 🎬 Comparison with LTX-2.3

<table>
  <tr>
    <th width="35%">Prompt</th>
    <th width="33%">LTX-2.3</th>
    <th width="33%">AV-GRPO FULL (ours)</th>
  </tr>
  <tr>
    <td>There is a large building on fire with intense flames and a lot of smoke billowing out. The building is surrounded by water, and there are some boats visible. A loud explosion is heard, followed by the sound of fire crackling and burning. The sky is clear, and there are some buildings in the background.</td>
    <td><video src="https://raw.githubusercontent.com/zhiyuxu03/AV-GRPO/main/AV-GRPO/assets/ltx2.3_1.mp4" width="100%" controls></video></td>
    <td><video src="https://raw.githubusercontent.com/zhiyuxu03/AV-GRPO/main/AV-GRPO/assets/av_grpo_full_1.mp4" width="100%" controls></video></td>
  </tr>
  <tr>
    <td>A woman is playing the violin in an orchestra setting. She is wearing a black top and green pants. The background shows other musicians with various instruments, and there's a stained glass window behind them. The sound of the violin blends with the orchestra's accompaniment.</td>
    <td><video src="https://raw.githubusercontent.com/zhiyuxu03/AV-GRPO/main/AV-GRPO/assets/ltx2.3_2.mp4" width="100%" controls></video></td>
    <td><video src="https://raw.githubusercontent.com/zhiyuxu03/AV-GRPO/main/AV-GRPO/assets/av_grpo_full_2.mp4" width="100%" controls></video></td>
  </tr>
  <tr>
    <td>A small stream flows over rocks and grass, with the water clear and the rocks covered in moss. A cartoonish green character with a round body and a single eye jumps into the water, making a croaking sound.</td>
    <td><video src="https://raw.githubusercontent.com/zhiyuxu03/AV-GRPO/main/AV-GRPO/assets/ltx2.3_3.mp4" width="100%" controls></video></td>
    <td><video src="https://raw.githubusercontent.com/zhiyuxu03/AV-GRPO/main/AV-GRPO/assets/av_grpo_full_3.mp4" width="100%" controls></video></td>
  </tr>
  <tr>
    <td>A child pushes a toy truck, says \"Vroom,\" pulls it back, says \"Beep,\" then crashes it into a block and shouts \"Boom.\"</td>
    <td><video src="https://raw.githubusercontent.com/zhiyuxu03/AV-GRPO/main/AV-GRPO/assets/ltx2.3_4.mp4" width="100%" controls></video></td>
    <td><video src="https://raw.githubusercontent.com/zhiyuxu03/AV-GRPO/main/AV-GRPO/assets/av_grpo_full_4.mp4" width="100%" controls></video></td>
  </tr>
  <tr>
    <td>A colossal waterfall thunders down from mist-shrouded emerald cliffs into a sapphire abyss, its deafening crash echoing through the valley, sending up explosive plumes of spray that catch the morning light.</td>
    <td><video src="https://raw.githubusercontent.com/zhiyuxu03/AV-GRPO/main/AV-GRPO/assets/ltx2.3_5.mp4" width="100%" controls></video></td>
    <td><video src="https://raw.githubusercontent.com/zhiyuxu03/AV-GRPO/main/AV-GRPO/assets/av_grpo_full_5.mp4" width="100%" controls></video></td>
  </tr>
  <tr>
    <td>Two men face each other nose-to-nose in a tense confrontation within a dimly lit bank. The older man says, \"我们才不会害怕残忍的流氓.\" The clown replies, \"你知道吗？你让我想起了我的父亲，我恨我的父亲.\" The audio shows suffocating silence broken by tense dialogue.</td>
    <td><video src="https://raw.githubusercontent.com/zhiyuxu03/AV-GRPO/main/AV-GRPO/assets/ltx2.3_6.mp4" width="100%" controls></video></td>
    <td><video src="https://raw.githubusercontent.com/zhiyuxu03/AV-GRPO/main/AV-GRPO/assets/av_grpo_full_6.mp4" width="100%" controls></video></td>
  </tr>
  <tr>
    <td>In a medium close-up, a young woman with blonde shoulder-length hair stands in a lavender field under a twilight sky. She says, \"告诉我这条光滑的绿色带子见证了多少年的沉重。\" The audio features gentle, atmospheric singing establishing a calm and wistful mood.</td>
    <td><video src="https://raw.githubusercontent.com/zhiyuxu03/AV-GRPO/main/AV-GRPO/assets/ltx2.3_7.mp4" width="100%" controls></video></td>
    <td><video src="https://raw.githubusercontent.com/zhiyuxu03/AV-GRPO/main/AV-GRPO/assets/av_grpo_full_7.mp4" width="100%" controls></video></td>
  </tr>
</table>


## ⚠️ License

Research use only. See individual submodule licenses (JavisDiT, LTX, etc.) for their terms.