# DL2026_Team2_Sketch-to-Real-Generation
Controllable Image Generation  Using ControlNet or Stable Diffusion, implement a “Sketch-to-Real” system where users draw rough outlines and the model generates a high-fidelity images.


1. Project Goal
[This project aims to synthesize realistic face images from sketch inputs. We evaluate and compare three distinct edge detection techniques to extract contours from images, using these sketches to condition our generative model. By integrating LoRA fine-tuning, we enhance the model's ability to retain delicate structural details and generate high-quality outputs.]

2. Dataset
* **Dataset Name:** xxx
* **Public Dataset Link:** Insert URL here
* **Description:** The raw images were preprocessed using our scripts to extract Canny, HED, and morphological edges. The processed edge maps are organized and stored under the respective `training_data` directories.
