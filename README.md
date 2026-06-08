# vision-language

Lightweight training and inference scripts for chest X-ray report-to-image experiments built around Stable Diffusion and MIMIC-CXR style data.

## Project framing

The baseline paper trained RoentGen-scale Stable Diffusion models on large multi-GPU hardware. This repository keeps the same core protocol where possible, but adapts it for a single local GPU by using smaller runs, gradient accumulation, LoRA adapters, and optional pathology reward-guided fine-tuning.

The goal is not to claim a full 64-GPU reproduction from a small run. The goal is a clean, lower-hardware experimental setup that can compare:

- full RoentGen-style fine-tuning
- resource-efficient LoRA adaptation
- PRG: LoRA plus pathology reward guidance

## What is included

- `train.py`: Stable Diffusion fine-tuning script with a lightweight `--demo_mode`
- `train_lora.py`: parameter-efficient LoRA fine-tuning for single-GPU runs
- `train_prg.py`: LoRA fine-tuning with pathology reward guidance
- `dataset.py`: metadata building, filtering, tokenization, and DataLoader creation
- `pathology_reward.py`: CheXpert label loading and differentiable XRV pathology reward
- `evaluate.py`: common evaluation metrics for baseline, LoRA, and PRG checkpoints
- `compare.py`: multi-model evaluation table generation
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
  mimic-cxr-2.0.0-chexpert.csv
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

## Data split protocol

The RoentGen paper uses MIMIC-CXR folders `p10`-`p18` for development/training and `p19` for the held-out in-distribution test set. This repository now follows that split by default:

- training scripts use `split=train` (`p10`-`p18`)
- `evaluate.py` and `compare.py` use `--eval_split test` (`p19`) by default
- `split=all` is only for debugging and should not be used for reported results

Metadata caches include the split name, so older mixed caches are not reused for new train/test runs.

## Training

Recommended first run:

```bash
python train.py --demo_mode --max_train_steps 100 --max_samples 128
```

Slightly larger run:

```bash
python train.py --image_size 256 --max_train_steps 300 --max_samples 256 --gradient_accumulation_steps 4 --per_device_batch_size 1
```

Single-GPU LoRA run:

```bash
python train_lora.py --data_root ./data --image_size 512 --max_train_steps 3000 --max_samples 4000 --lora_rank 8 --save_every 1000
```

Pathology reward-guided run:

```bash
python train_prg.py --data_root ./data --chexpert_csv ./data/mimic-cxr-2.0.0-chexpert.csv --image_size 256 --max_train_steps 3000 --max_samples 4000 --lambda_reward 0.5 --reward_every 4 --lora_rank 8 --save_every 1000
```

For fair comparisons, use the same seed, resolution, sample count, and training steps for LoRA and PRG; change only the reward settings.

## Inference

```bash
python inference.py --checkpoint_dir ./checkpoints/checkpoint-100 --prompt "Right lower lobe pneumonia"
```

## Evaluation

Evaluate one checkpoint on the held-out p19 split:

```bash
python evaluate.py --checkpoint_dir ./checkpoints_lora/checkpoint-3000 --data_root ./data --model_name lora_3000 --num_eval_samples 200
```

Compare multiple checkpoints:

```bash
python compare.py --data_root ./data --num_eval_samples 200 --model baseline:./checkpoints/checkpoint-3000 --model lora:./checkpoints_lora/checkpoint-3000 --model prg:./checkpoints_prg/checkpoint-3000
```

## Repo structure

```text
vision-language/
  dataset.py
  train.py
  train_lora.py
  train_prg.py
  pathology_reward.py
  evaluate.py
  compare.py
  inference.py
  requirements.txt
  roentgen_project_blueprint.md
  data/
  checkpoints/
```

## Practical hardware note

- `RTX 4070 12 GB`: use LoRA/PRG, small batch size, gradient accumulation, and capped sample counts
- `RTX A5000 24 GB`: more suitable for 512px LoRA and longer runs
- Full RoentGen-scale 60k-step joint fine-tuning remains a multi-GPU experiment; local runs should be reported as constrained reproductions/adaptations

## GitHub upload note

The `.gitignore` file excludes:

- dataset files
- generated checkpoints
- local cache
- IDE files
- the local reference PDF

That keeps the GitHub repository clean and small.
