# Sketch-to-Real Face Generation via ControlNet + LoRA

A deep learning project that generates photorealistic human face images from sketch-style edge maps, using **Stable Diffusion v1.5 + ControlNet + LoRA fine-tuning**. Three different edge extraction methods are compared: **Canny**, **HED**, and **Morphological Gradient**.

---

## Table of Contents

- [Project Goal](#project-goal)
- [Method Overview](#method-overview)
- [Repository Structure](#repository-structure)
- [File Descriptions](#file-descriptions)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Dependencies](#dependencies)

---

## Project Goal

Sketch-to-image generation is a challenging task in conditional image synthesis. Given a hand-drawn or algorithmically extracted sketch of a human face, the goal is to generate a photorealistic photo that faithfully follows the structural outline of the sketch.

This project investigates:

1. **How different edge extraction methods** (Canny, HED, Morphological Gradient) affect the quality and fidelity of the generated face images.
2. **Whether LoRA fine-tuning** on a domain-specific face dataset (FFHQ-256) improves generation quality over a vanilla ControlNet baseline.

The pipeline is:

```
Edge Map (Sketch)  →  ControlNet (scribble)  →  SD v1.5 UNet (+ LoRA)  →  Generated Face Photo
```

LoRA is applied exclusively to the UNet's attention layers (`to_q`, `to_k`, `to_v`, `to_out.0`), keeping the VAE, text encoder, and ControlNet frozen during training. This keeps training lightweight and focused on adapting the UNet to the face domain.

---

## Method Overview

| Method | Description | ControlNet Conditioning |
|---|---|---|
| **Canny** | Classic gradient-based edge detection with hysteresis thresholding | `sd-controlnet-scribble` |
| **HED** | Holistically-nested Edge Detection; produces softer, hierarchical edges | `sd-controlnet-scribble` |
| **Morphological Gradient** | Dilation minus erosion; captures structural boundaries with filled regions | `sd-controlnet-scribble` |


---

## Repository Structure

```
.
├── 01_prepare_dataset_Canny.py
├── 01_prepare_dataset_HED.py
├── 01_prepare_dataset_morph.py
├── 03_face_canny_p1_lora_inference_evaluate.py
├── 03_face_hed_p1_lora_inference_evaluate.py
├── 03_face_morph_p1_lora_inference_evaluate.py
├── 04_real_test.py
│
├── training_data/                  # Canny dataset + LoRA training notebook
│   ├── train.jsonl                 # Metadata linking images to captions
│   └── train_lora.ipynb            # LoRA training script (Canny)
│
├── training_data_HED/              # HED dataset + LoRA training notebook
│   ├── train.jsonl
│   └── train_lora.ipynb            # LoRA training script (HED)
│
├── training_data_morph/            # Morphological Gradient dataset + LoRA training notebook
│   ├── train.jsonl
│   └── train_lora.ipynb            # LoRA training script (Morph)
│
└── real_test/                      # Real user-provided sketch images for inference
```

---

## File Descriptions

### Dataset Preparation

**`01_prepare_dataset_Canny.py`** / **`01_prepare_dataset_HED.py`** / **`01_prepare_dataset_morph.py`**

These three scripts handle dataset construction for each edge method. They share the same overall pipeline, differing only in how the conditioning (edge) image is generated:

- Downloads 500 face images from the [FFHQ-256](https://huggingface.co/datasets/merkol/ffhq-256) dataset on Hugging Face (streaming mode to avoid downloading the full 5 GB).
- Resizes all images to 512×512.
- Applies the respective edge extraction algorithm to produce the conditioning image:
  - **Canny**: Gaussian blur + Canny edge detector (thresholds 30/100), result inverted to white-on-black sketch style.
  - **HED**: Holistically-nested edge detection producing soft, multi-scale edge probability maps.
  - **Morph**: Morphological gradient (dilation − erosion) to extract structural boundaries.
- Saves ground truth photos to `images/` and edge maps to `conditioning_images/`.
- Writes a `train.jsonl` metadata file pairing each image with its conditioning image and a fixed text prompt: `"a person, photorealistic, high quality, detailed skin"`.
- Backs up the dataset to Google Drive.

### LoRA Training

**`training_data/train_lora.ipynb`** / **`training_data_HED/train_lora.ipynb`** / **`training_data_morph/train_lora.ipynb`**

These notebooks fine-tune the SD v1.5 UNet with LoRA on each respective edge-conditioned dataset. All three share identical hyperparameters:

| Hyperparameter | Value |
|---|---|
| Base model | `runwayml/stable-diffusion-v1-5` |
| ControlNet | `lllyasviel/sd-controlnet-scribble` |
| LoRA rank | 4 |
| LoRA alpha | 4 |
| Target modules | `to_q`, `to_k`, `to_v`, `to_out.0` |
| Learning rate | 5e-5 |
| Training steps | 500 |
| Batch size | 1 (gradient accumulation: 2) |
| Mixed precision | fp16 |
| Optimizer | AdamW + cosine schedule (100 warmup steps) |

Only the LoRA parameters in the UNet are trained; VAE, text encoder, and ControlNet are frozen. The trained LoRA weights are saved via `unet.save_pretrained()` and backed up to Google Drive.

### Inference & Evaluation

**`03_face_canny_p1_lora_inference_evaluate.py`** / **`03_face_hed_p1_lora_inference_evaluate.py`** / **`03_face_morph_p1_lora_inference_evaluate.py`**

These scripts run inference on the first 50 test images and compute six evaluation metrics comparing the **Baseline** (ControlNet only, no LoRA) against the **LoRA fine-tuned** model. See [Evaluation Metrics](#evaluation-metrics) for details.

Outputs:
- `eval_6metrics.csv` — per-image metric scores for both models.
- FID scores computed via `pytorch-fid` over all 50 generated images.

**`04_real_test.py`**

Runs inference and evaluation on real user-provided sketches stored in the `real_test/` folder. This is the end-to-end demo script — it accepts arbitrary hand-drawn or externally sourced sketches (not from the FFHQ training set) and generates photorealistic face images. Evaluation metrics are also computed where ground truth images are available.

---

## Usage

All scripts are designed to run on **Google Colab** with a GPU runtime. Google Drive is used for data persistence.
Please ensure all paths are correct.

### Step 1 — Prepare the Dataset

Run one of the dataset preparation scripts depending on the edge method you want to use:

```bash
# For Canny
python 01_prepare_dataset_Canny.py

# For HED
python 01_prepare_dataset_HED.py

# For Morphological Gradient
python 01_prepare_dataset_morph.py
```

This will create a `training_data/` (or `training_data_HED/`, `training_data_morph/`) directory and back it up to your Google Drive at:
```
MyDrive/++DL/04_project_face/training_data
```

### Step 2 — Train LoRA

Open the corresponding `train_lora.ipynb` notebook inside the training data folder in Google Colab and run all cells. The trained LoRA weights will be saved to:
```
MyDrive/++DL/04_project_face/lora_output_test_p1
```

Adjust `MAX_STEPS` in the config block if you want a longer or shorter training run.

### Step 3 — Evaluate

Run the corresponding evaluation script:

```bash
# For Canny
python 03_face_canny_p1_lora_inference_evaluate.py

# For HED
python 03_face_hed_p1_lora_inference_evaluate.py

# For Morphological Gradient
python 03_face_morph_p1_lora_inference_evaluate.py
```

Results will be saved under:
```
MyDrive/++DL/04_project_face/eval_6metrics/
```

### Step 4 — Real Sketch Test

Place your sketch images in the `real_test/` folder, or upload images, then run:

```bash
python 04_real_test.py
```

---

## Evaluation Metrics

Six metrics are computed for each generated image, comparing Baseline vs. LoRA:

| Metric | Direction | Description |
|---|---|---|
| **CLIP Score** (ViT-L/14) | ↑ higher is better | Cosine similarity between the generated image embedding and the text prompt embedding. Measures semantic alignment with the prompt. |
| **LPIPS** (VGG) | ↓ lower is better | Perceptual similarity between generated image and ground truth using VGG intermediate features. More perceptually accurate than pixel-wise L2. |
| **SSIM** | ↑ higher is better | Structural Similarity Index comparing luminance, contrast, and structure against ground truth. |
| **DINOv2 Similarity** | ↑ higher is better | Cosine similarity of DINOv2 [CLS] token embeddings between generated image and ground truth. Captures visual semantics without text-image pairing bias. |
| **ArcFace Identity Score** | ↑ higher is better | Cosine similarity of ArcFace (buffalo_l) 512-dim face embeddings. Directly measures how well the generated face preserves the identity of the ground truth. Returns `None` if no face is detected. |
| **Edge IoU** | ↑ higher is better | IoU between Canny edges of the generated image and the input sketch. Measures how faithfully the generated image follows the sketch's structural outline. |

In addition, **FID (Fréchet Inception Distance)** is computed over the full 50-image test set for an overall distribution-level quality comparison.

---

## Dependencies

Install via pip in Colab:

```bash
pip install datasets opencv-python-headless diffusers transformers accelerate peft
pip install open_clip_torch lpips pytorch-fid insightface onnxruntime
pip install scikit-image torchao
```

Key library versions used during development:

| Library | Role |
|---|---|
| `diffusers` | SD v1.5, ControlNet, DDPMScheduler, pipelines |
| `peft` | LoRA via `LoraConfig` and `get_peft_model` |
| `transformers` | CLIP text encoder, DINOv2 |
| `open_clip_torch` | CLIP Score evaluation (ViT-L/14) |
| `lpips` | Perceptual similarity metric |
| `insightface` | ArcFace identity score |
| `pytorch-fid` | FID computation |
| `datasets` | FFHQ-256 streaming download |

---

## Acknowledgements

- Face dataset: [FFHQ-256](https://huggingface.co/datasets/merkol/ffhq-256)
- Base diffusion model: [runwayml/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- ControlNet weights: [lllyasviel/sd-controlnet-scribble](https://huggingface.co/lllyasviel/sd-controlnet-scribble)
- LoRA implementation: [HuggingFace PEFT](https://github.com/huggingface/peft)
