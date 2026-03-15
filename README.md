# vision-language

Lightweight training and inference scripts for chest X-ray report-to-image experiments built around Stable Diffusion and MIMIC-CXR style data.

## What is included

- `train.py`: Stable Diffusion fine-tuning script with a lightweight `--demo_mode`
- `dataset.py`: metadata building, filtering, tokenization, and DataLoader creation
- `inference.py`: checkpoint loading and image generation
- `requirements.txt`: pinned package versions used by the project

## Notes before running

- This repository does not include the dataset.
- Full Stable Diffusion fine-tuning is heavy. For first runs, use `--demo_mode`.
- A CUDA-enabled PyTorch install is required. Install the correct PyTorch build for your CUDA version before `pip install -r requirements.txt`.

## Expected data layout

```text
data/
  mimic_cxr_aug_train.csv
  mimic_cxr_aug_validate.csv
  official_data_iccv_final/
    files/
      p10/
      p11/
      ...
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Adjust the PyTorch install command to match the CUDA version on the target machine.

## Training

Recommended first run:

```bash
python train.py --demo_mode --max_train_steps 100 --max_samples 128
```

Slightly larger run:

```bash
python train.py --image_size 256 --max_train_steps 300 --max_samples 256 --gradient_accumulation_steps 4 --per_device_batch_size 1
```

## Inference

```bash
python inference.py --checkpoint_dir ./checkpoints/checkpoint-100 --prompt "Right lower lobe pneumonia"
```

## Repo structure

```text
vision-language/
  dataset.py
  train.py
  inference.py
  requirements.txt
  roentgen_project_blueprint.md
  data/
  checkpoints/
```

## Practical hardware note

- `RTX 4070 12 GB`: use `--demo_mode` or small image sizes
- `RTX A5000 24 GB`: much more suitable for real training runs

## GitHub upload note

The `.gitignore` file excludes:

- dataset files
- generated checkpoints
- local cache
- IDE files
- the local reference PDF

That keeps the GitHub repository clean and small.
