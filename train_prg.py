"""
train_prg.py — Pathology Reward-Guided fine-tuning (MERKEZ KATKI)
==================================================================
Baseline (RoentGen-tarzı SD) üzerine, üretilen görüntünün prompttaki HER
patolojiyi içermesini zorlayan diferansiyellenebilir bir ödül ekler.

Toplam kayıp:
    L = MSE_denoising  +  lambda_reward * BCE( XRV(x̂₀_decode), hedef_etiketler )

- LoRA ile eğitilir → tek tüketici GPU'da sığar (verimlilik aracı).
- Ödül her `reward_every` adımda hesaplanır (VAE decode + XRV pahalı).
- Kayıt: LoRA merge edilip standart formatta → evaluate.py/inference.py değişmeden çalışır.

Önkoşullar:
    pip install peft torchxrayvision
    MIMIC CheXpert etiket dosyası: mimic-cxr-2.0.0-chexpert.csv

Adil kıyas: baseline (train.py veya train_lora.py, ödülsüz) ile AYNI
seed/veri/çözünürlük/adım; tek değişen = patoloji ödülü.

Örnek:
  python train_prg.py --data_root .\data --chexpert_csv .\data\mimic-cxr-2.0.0-chexpert.csv \
      --image_size 256 --max_train_steps 3000 --max_samples 4000 \
      --lambda_reward 0.5 --reward_every 4 --lora_rank 8 --save_every 1000
"""

import argparse
import contextlib
import logging
import os
import platform

import torch
import torch.nn.functional as F

import time

from dataset import create_dataloader
from train_lora import setup_lora_model, _save_merged, MODEL_ID  # LoRA altyapısını yeniden kullan
from pathology_reward import PathologyReward, latents_to_pixels
from run_logging import log_efficiency, reset_peak_vram, count_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LEARNING_RATE = 1e-4
TOTAL_TRAIN_STEPS = 3_000
DEMO_IMAGE_SIZE = 256
DEMO_MAX_SAMPLES = 256
DEMO_MAX_STEPS = 100
_DEFAULT_WORKERS = 0 if platform.system() == "Windows" else 4


def train(
    data_root: str,
    chexpert_csv: str,
    output_dir: str = "./checkpoints_prg",
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_train_steps: int = TOTAL_TRAIN_STEPS,
    learning_rate: float = LEARNING_RATE,
    use_bf16: bool = True,
    num_workers: int = _DEFAULT_WORKERS,
    save_every: int = 1000,
    csv_path: str = None,
    image_size: int = 256,
    max_samples: int = None,
    demo_mode: bool = False,
    use_gradient_checkpointing: bool = True,
    lora_rank: int = 8,
    lora_alpha: int = None,
    lora_dropout: float = 0.0,
    adapt_text_encoder: bool = True,
    lambda_reward: float = 0.5,
    reward_every: int = 4,
    positive_weight: float = 2.0,
    seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)

    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Seed: %d", seed)

    if lora_alpha is None:
        lora_alpha = lora_rank

    if demo_mode:
        image_size = min(image_size, DEMO_IMAGE_SIZE)
        max_samples = DEMO_MAX_SAMPLES if max_samples is None else min(max_samples, DEMO_MAX_SAMPLES)
        max_train_steps = min(max_train_steps, DEMO_MAX_STEPS)
        save_every = min(save_every, max_train_steps)
        logger.info("Demo modu: image_size=%s, max_samples=%s, steps=%s",
                    image_size, max_samples, max_train_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = (
        torch.bfloat16
        if use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    logger.info("Cihaz: %s | dtype: %s | λ_reward: %.3f | reward_every: %d",
                device, weight_dtype, lambda_reward, reward_every)

    vae, unet, text_encoder, tokenizer, noise_scheduler = setup_lora_model(
        device=device, weight_dtype=weight_dtype,
        lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        adapt_text_encoder=adapt_text_encoder,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    # ── Donuk patoloji ödül modeli (XRV DenseNet) ────────────────────────────
    reward_model = PathologyReward(device) if lambda_reward > 0 else None

    # ── Veri (CheXpert etiketleriyle) ────────────────────────────────────────
    train_loader = create_dataloader(
        data_root=data_root, batch_size=per_device_batch_size, num_workers=num_workers,
        csv_path=csv_path, image_size=image_size, max_samples=max_samples,
        chexpert_csv=chexpert_csv,
        split="train",
    )
    if len(train_loader) == 0:
        raise RuntimeError("DataLoader 0 batch üretti.")

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if adapt_text_encoder:
        trainable_params += [p for p in text_encoder.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    # ── Metrik CSV ────────────────────────────────────────────────────────────
    import csv as _csv
    metrics_path = os.path.join(output_dir, "metrics_prg.csv")
    csv_file = open(metrics_path, "w", newline="", encoding="utf-8")
    csv_writer = _csv.writer(csv_file)
    csv_writer.writerow(["step", "mse_loss", "reward_loss", "total_loss"])

    global_step = 0
    accum_mse = 0.0
    accum_reward = 0.0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    use_autocast = device.type == "cuda" and weight_dtype != torch.float32
    reset_peak_vram()
    start_time = time.time()

    logger.info("PRG eğitimi başlıyor: hedef %s adım", max_train_steps)
    done = False
    while not done:
        for batch in train_loader:
            if global_step >= max_train_steps:
                done = True
                break

            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target_labels = batch.get("labels", None)
            if target_labels is not None:
                target_labels = target_labels.to(device, non_blocking=True)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            autocast_ctx = (
                torch.cuda.amp.autocast(dtype=weight_dtype) if use_autocast else contextlib.nullcontext()
            )
            with autocast_ctx:
                if adapt_text_encoder:
                    encoder_hidden_states = text_encoder(input_ids)[0]
                else:
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(input_ids)[0]
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            mse_loss = F.mse_loss(noise_pred.float(), noise.float())

            # ── Patoloji ödülü (her reward_every adımda) ──────────────────────
            reward_loss = torch.tensor(0.0, device=device)
            do_reward = (
                reward_model is not None
                and target_labels is not None
                and (global_step + 1) % reward_every == 0
            )
            if do_reward:
                decoded01 = latents_to_pixels(
                    noise_pred, noisy_latents, timesteps, noise_scheduler, vae, weight_dtype,
                )
                reward_loss = reward_model.reward_loss(
                    decoded01, target_labels, positive_weight=positive_weight,
                )

            total_loss = (mse_loss + lambda_reward * reward_loss) / gradient_accumulation_steps
            total_loss.backward()

            accum_mse += mse_loss.item()
            accum_reward += float(reward_loss.item()) if torch.is_tensor(reward_loss) else 0.0
            micro_step += 1

            if micro_step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_mse = accum_mse / gradient_accumulation_steps
                avg_reward = accum_reward / gradient_accumulation_steps
                logger.info("Step %s/%s | mse=%.6f | reward=%.6f | total=%.6f",
                            global_step, max_train_steps, avg_mse, avg_reward, avg_mse + lambda_reward * avg_reward)
                csv_writer.writerow([global_step, round(avg_mse, 6), round(avg_reward, 6),
                                     round(avg_mse + lambda_reward * avg_reward, 6)])
                csv_file.flush()
                accum_mse = accum_reward = 0.0

                if global_step % save_every == 0:
                    _save_merged(unet, text_encoder, tokenizer, vae, adapt_text_encoder, global_step, output_dir)

                if global_step >= max_train_steps:
                    done = True
                    break

    _save_merged(unet, text_encoder, tokenizer, vae, adapt_text_encoder, global_step, output_dir)
    csv_file.close()

    _, total = count_params(unet, text_encoder, vae)
    eff_batch = per_device_batch_size * gradient_accumulation_steps
    rec = log_efficiency(
        output_dir=output_dir, method=f"prg_lam{lambda_reward}_r{lora_rank}", image_size=image_size,
        steps=global_step, effective_batch=eff_batch,
        trainable=n_trainable, total=total, start_time=start_time,
    )
    logger.info("Verimlilik: %s", rec)
    logger.info("PRG eğitimi tamamlandı. Metrikler: %s", metrics_path)


def parse_args():
    p = argparse.ArgumentParser(description="Pathology Reward-Guided fine-tuning (PRG)")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--chexpert_csv", type=str, required=True,
                   help="mimic-cxr-2.0.0-chexpert.csv yolu (patoloji hedef etiketleri)")
    p.add_argument("--csv_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./checkpoints_prg")
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_train_steps", type=int, default=TOTAL_TRAIN_STEPS)
    p.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--num_workers", type=int, default=_DEFAULT_WORKERS)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--demo_mode", action="store_true")
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--no_text_encoder_lora", action="store_true")
    p.add_argument("--lambda_reward", type=float, default=0.5,
                   help="Patoloji ödülü ağırlığı (0 = saf baseline LoRA)")
    p.add_argument("--reward_every", type=int, default=4,
                   help="Ödül kaç optimizer-micro adımda bir hesaplansın (maliyet kontrolü)")
    p.add_argument("--positive_weight", type=float, default=2.0,
                   help="Pozitif (mevcut) bulgulara verilen BCE ağırlığı")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_root=args.data_root,
        chexpert_csv=args.chexpert_csv,
        output_dir=args.output_dir,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_train_steps=args.max_train_steps,
        learning_rate=args.learning_rate,
        use_bf16=not args.no_bf16,
        num_workers=args.num_workers,
        save_every=args.save_every,
        csv_path=args.csv_path,
        image_size=args.image_size,
        max_samples=args.max_samples,
        demo_mode=args.demo_mode,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        adapt_text_encoder=not args.no_text_encoder_lora,
        lambda_reward=args.lambda_reward,
        reward_every=args.reward_every,
        positive_weight=args.positive_weight,
        seed=args.seed,
    )
