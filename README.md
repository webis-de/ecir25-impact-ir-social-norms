# Classification of Multimodal Social Media Posts

This repository contains the code and resources for the "Classification of Multimodal Social Media Posts" project.

## Repository Structure
<pre>
├── 📃 .gitignore <-- Specifies untracked files to ignore\
├── 📃 README.md <-- You are here! Project overview and repo structure\
├── 📃 requirements.txt <-- Python dependencies for the project\
│
├── 📁 notebooks <-- Jupyter notebooks for initial experiments:\
│ ├── bg_removal.ipynb <-- For background removal in images\
│ ├── data_augmentation.ipynb <-- Data augmentation techniques\
│ ├── fine_tuning.ipynb <-- Fine-tuning of models\
│ ├── model_testing.ipynb <-- Testing the models\
│ └── resize.ipynb <-- Resizing images for uniformity\
│
├── 📁 results <-- Evaluation results and model comparisons:\
│ ├── 📁 bert <-- Results from BERT model\
│ ├── 📁 late_fusion <-- Results from Late Fusion model\
│ ├── 📁 zero_shot <-- Results from zero_shot models\
│ └── 📁 vision_models <-- Results from various vision models\
│
└── 📁 src <-- Source code for the project:\
│ ├── 📁 fusion_model <-- The Late Fusion Model\
│ ├── 📁 imag_processing <-- Data Augmentation and Image Processing\
│ ├── 📁 llava <-- The LLaVA Model Experiments\
│ ├── 📁 text_models <-- Bert Fine-Tuning\
│ └── 📁 text_processing <-- GPT4 Reformulation\
│ └── 📁 vision_models <-- Fine-Tuning Vision Models + CLIP Zero-Shot\
</pre>