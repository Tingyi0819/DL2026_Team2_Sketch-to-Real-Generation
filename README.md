# DL2026_Team2_Sketch-to-Real-Generation
Controllable Image Generation  Using ControlNet or Stable Diffusion, implement a “Sketch-to-Real” system where users draw rough outlines and the model generates a high-fidelity images.


1. Project Goal
[This project aims to synthesize realistic face images from sketch inputs. We evaluate and compare three distinct edge detection techniques to extract contours from images, using these sketches to condition our generative model. By integrating LoRA fine-tuning, we enhance the model's ability to retain delicate structural details and generate high-quality outputs.]

2. Dataset
* **Dataset Name:** xxx
* **Public Dataset Link:** Insert URL here
* **Description:** The raw images were preprocessed using our scripts to extract Canny, HED, and morphological edges. The processed edge maps are organized and stored under the respective `training_data` directories.

3. File Structure
The organization of this repository and the purpose of each file are outlined below:

├── training_data/                # Original / baseline training data directory<br>
├── training_data_HED/            # Dataset preprocessed with HED edge detection<br>
├── training_data_morph/          # Dataset preprocessed with Morphological operations<br>
│<br>
├── 01_prepare_dataset_Canny.py   # Preprocessing: Generates Canny edge sketches<br>
├── 01_prepare_dataset_HED.py     # Preprocessing: Generates HED edge sketches<br>
├── 01_prepare_dataset_morph.py   # Preprocessing: Generates Morphological sketches<br>
│<br>
├── 03_face_canny_p1_lora_inference_evaluate.py  # Inference & evaluation for the Canny model<br>
├── 03_face_hed_p1_lora_inference_evaluate.py    # Inference & evaluation for the HED model<br>
├── 03_face_morph_p1_lora_inference_evaluate.py  # Inference & evaluation for the Morph model<br>
│<br>
├── 04_real_test.py               # Final evaluation and generation script<br>
├── LICENSE                       # Project license (MIT License)<br>
└── README.md                     # This documentation file<br>

4. Installation & Usage

### Prerequisites
Ensure you have the required Python packages installed:
```bash
pip install -r requirements.txt
#[torch, diffusers, opencv-python]
