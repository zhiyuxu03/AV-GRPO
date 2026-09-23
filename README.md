# 🎬 AV-GRPO


[![Model](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/Dr-Loser/AV-GRPO)
[![Paper](https://img.shields.io/badge/Paper-PDF-EC1C24?logo=adobeacrobatreader&logoColor=white)]()


## 📋 Overview

AV-GRPO, to our knowledge, is the first GRPO framework designed for joint audio-video generation models. It enables full-parameter training or LoRA training of the 22B LTX-2.3 model on just **8 A800 GPUs**.

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

### 🔍 Step 4: Replace paths in the configuration file

Use a search command to find `.../AV-GRPO/` in the file below and replace all occurrences with your actual file path. **Make sure not to miss any.**

| File | Description |
|------|-------------|
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
    <td><video src="https://github.com/user-attachments/assets/aec4139e-10ea-43ad-b1d3-f7571b50a846" width="100%" controls></video></td>
    <td><video src="https://github.com/user-attachments/assets/e70c8247-2ceb-4b7e-b0ca-a60e23d00c79" width="100%" controls></video></td>
  </tr>
  <tr>
    <td>A woman is playing the violin in an orchestra setting. She is wearing a black top and green pants. The background shows other musicians with various instruments, and there's a stained glass window behind them. The sound of the violin blends with the orchestra's accompaniment.</td>
    <td><video src="https://github.com/user-attachments/assets/095ee065-be53-4712-a84a-82fd5c459a3c" width="100%" controls></video></td>
    <td><video src="https://github.com/user-attachments/assets/721106ec-d343-4a26-99bf-c2ffdc5748df" width="100%" controls></video></td>
  </tr>
  <tr>
    <td>A small stream flows over rocks and grass, with the water clear and the rocks covered in moss. A cartoonish green character with a round body and a single eye jumps into the water, making a croaking sound.</td>
    <td><video src="https://github.com/user-attachments/assets/bd8a4b98-d659-43dc-b5f1-8430e391f4ce" width="100%" controls></video></td>
    <td><video src="https://github.com/user-attachments/assets/253fa17f-cdc0-4c1b-9c8d-8af847fb96e0" width="100%" controls></video></td>
  </tr>
  <tr>
    <td>A child pushes a toy truck, says \"Vroom,\" pulls it back, says \"Beep,\" then crashes it into a block and shouts \"Boom.\"</td>
    <td><video src="https://github.com/user-attachments/assets/1ec280f7-055c-4f73-9ee7-96ac16df56af" width="100%" controls></video></td>
    <td><video src="https://github.com/user-attachments/assets/4c7375fb-5398-4d93-8277-eeb3fb85e33e" width="100%" controls></video></td>
  </tr>
  <tr>
    <td>A colossal waterfall thunders down from mist-shrouded emerald cliffs into a sapphire abyss, its deafening crash echoing through the valley, sending up explosive plumes of spray that catch the morning light.</td>
    <td><video src="https://github.com/user-attachments/assets/0216653e-07db-4184-a343-b2d8adf650ae" width="100%" controls></video></td>
    <td><video src="https://github.com/user-attachments/assets/b42cdb3c-8a18-4ce7-983e-ae4d7d165464" width="100%" controls></video></td>
  </tr>
  <tr>
    <td>Two men face each other nose-to-nose in a tense confrontation within a dimly lit bank. The older man says, \"我们才不会害怕残忍的流氓.\" The clown replies, \"你知道吗？你让我想起了我的父亲，我恨我的父亲.\" The audio shows suffocating silence broken by tense dialogue.</td>
    <td><video src="https://github.com/user-attachments/assets/e157e826-4c5b-45f7-973c-71d713a7d52e" width="100%" controls></video></td>
    <td><video src="https://github.com/user-attachments/assets/8860cec7-fc8a-4384-a8d5-bdae0a8c3e1e" width="100%" controls></video></td>
  </tr>
  <tr>
    <td>In a medium close-up, a young woman with blonde shoulder-length hair stands in a lavender field under a twilight sky. She says, \"告诉我这条光滑的绿色带子见证了多少年的沉重。\" The audio features gentle, atmospheric singing establishing a calm and wistful mood.</td>
    <td><video src="https://github.com/user-attachments/assets/f83b6a36-4e7c-4512-a720-9f3f302787b7" width="100%" controls></video></td>
    <td><video src="https://github.com/user-attachments/assets/61b5c9c2-5c51-418b-afe2-a20d7d013ad8" width="100%" controls></video></td>
  </tr>
</table>


## ⚠️ License

See individual submodule licenses (JavisDiT, LTX, etc.) for their terms.
