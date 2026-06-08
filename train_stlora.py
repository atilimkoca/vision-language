"""
train_stlora.py — Spectral-Temporal LoRA eğitimi (ÖZGÜN YÖNTEM denemesi)
=========================================================================
ΔW(t) = (α/r) B diag(s(t)) A — timestep-modülasyonlu düşük-rank adaptasyon.

LoRA ile AYNI bütçede (aynı rank/adım/veri/batch) eğitilir; tek fark: UNet
adaptasyonu gürültü seviyesine göre spektral olarak modüle edilir. Hipotez:
sabit bütçede fidelity'yi artırır (verimiz "kapasite eklemek yardım etmiyor,
tahsis önemli" diyordu).

NOT: ΔW(t) merge edilemez → checkpoint, base SD + ST-LoRA state olarak kaydedilir
(stlora.pt). Değerlendirme/çıkarım base'e enjekte edip state yükler (ayrı adım).

Önce DUMAN TESTİ:
  python train_stlora.py --data_root .\data --demo_mode --max_samples 64 --lora_rank 8
Gerçek (lora_256 ile eşit bütçe):
  python train_stlora.py --data_root .\data --output_dir .\checkpoints_stlora_256 \
      --image_size 256 --max_train_steps 5000 --max_samples 5000 \
      --per_device_batch_size 8 --gradient_accumulation_steps 4 --lora_rank 8 --save_every 2500
"""

import argparse
import contextlib
import logging
import os
import platform
import time

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from dataset import create_dataloader
from run_logging import LossLogger, log_efficiency, reset_peak_vram, count_params
from spectral_temporal_lora import (
    inject_stlora, attach_timestep_hook, trainable_parameters, save_stlora,
    UNET_TARGETS, TEXT_ENCODER_TARGETS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "CompVis/stable-diffusion-v1-4"
LEARNING_RATE = 1e-4
TOTAL_TRAIN_STEPS = 5_000
DEMO_IMAGE_SIZE = 256
DEMO_MAX_SAMPLES = 256
DEMO_MAX_STEPS = 100
_DEFAULT_WORKERS = 0 if platform.system() == "Windows" else 4


def train(
    data_root: str,
    output_dir: str = "./checkpoints_stlora",
    per_device_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    max_train_steps: int = TOTAL_TRAIN_STEPS,
    learning_rate: float = LEARNING_RATE,
    use_bf16: bool = True,
    num_workers: int = _DEFAULT_WORKERS,
    save_every: int = 2500,
    csv_path: str = None,
    image_size: int = 256,
    max_samples: int = None,
    demo_mode: bool = False,
    use_gradient_checkpointing: bool = True,
    lora_rank: int = 8,
    lora_alpha: int = None,
    adapt_text_encoder: bool = True,
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
    logger.info("Cihaz: %s | dtype: %s | rank: %s | text encoder ST-LoRA: %s",
                device, weight_dtype, lora_rank, adapt_text_encoder)

    # ── Base SD (donuk) ──────────────────────────────────────────────────────
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    for m in (vae, unet, text_encoder):
        m.requires_grad_(False)
    vae.eval()
    if use_gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Base'i bf16'ya taşı (ST-LoRA enjeksiyonundan ÖNCE)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    # ── ST-LoRA enjekte et ───────────────────────────────────────────────────
    unet_layers = inject_stlora(unet, UNET_TARGETS, lora_rank, lora_alpha, timestep_gate=True)
    injected = [unet_layers]
    if adapt_text_encoder:
        te_layers = inject_stlora(text_encoder, TEXT_ENCODER_TARGETS, lora_rank, lora_alpha,
                                  timestep_gate=False)   # metin kodlayıcı timestep görmez
        injected.append(te_layers)
        text_encoder.train()
    else:
        text_encoder.eval()
    unet.train()

    # Yeni (fp32) ST-LoRA paramlarını cihaza taşı (dtype korunur)
    unet.to(device)
    text_encoder.to(device)

    # Timestep'i UNet ST-LoRA katmanlarına ileten hook
    attach_timestep_hook(unet, list(unet_layers.values()))

    trainable = trainable_parameters(injected)
    n_trainable = sum(p.numel() for p in trainable)
    logger.info("Eğitilen ST-LoRA parametresi: %s (~%.3fM)", f"{n_trainable:,}", n_trainable / 1e6)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)

    train_loader = create_dataloader(
        data_root=data_root, batch_size=per_device_batch_size, num_workers=num_workers,
        csv_path=csv_path, image_size=image_size, max_samples=max_samples,
    )
    if len(train_loader) == 0:
        raise RuntimeError("DataLoader 0 batch üretti.")

    global_step = 0
    accum_loss = 0.0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    use_autocast = device.type == "cuda" and weight_dtype != torch.float32
    loss_csv = LossLogger(os.path.join(output_dir, "metrics.csv"))
    reset_peak_vram()
    start_time = time.time()

    logger.info("ST-LoRA eğitimi başlıyor: hedef %s adım", max_train_steps)
    done = False
    while not done:
        for batch in train_loader:
            if global_step >= max_train_steps:
                done = True
                break

            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            ac = torch.cuda.amp.autocast(dtype=weight_dtype) if use_autocast else contextlib.nullcontext()
            with ac:
                encoder_hidden_states = text_encoder(input_ids)[0]
                # hook UNet forward öncesi timesteps'i ST-LoRA katmanlarına stash eder
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            loss = F.mse_loss(noise_pred.float(), noise.float())
            accum_loss += loss.item()
            (loss / gradient_accumulation_steps).backward()
            micro_step += 1

            if micro_step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_loss = accum_loss / gradient_accumulation_steps
                logger.info("Step %s/%s loss=%.6f", global_step, max_train_steps, avg_loss)
                loss_csv.log(global_step, round(avg_loss, 6))
                accum_loss = 0.0

                if global_step % save_every == 0:
                    _save(injected, unet_layers, lora_rank, lora_alpha, image_size,
                          adapt_text_encoder, global_step, output_dir)

                if global_step >= max_train_steps:
                    done = True
                    break

    _save(injected, unet_layers, lora_rank, lora_alpha, image_size, adapt_text_encoder,
          global_step, output_dir)
    loss_csv.close()

    rec = log_efficiency(
        output_dir=output_dir, method=f"stlora_r{lora_rank}", image_size=image_size,
        steps=global_step, effective_batch=per_device_batch_size * gradient_accumulation_steps,
        trainable=n_trainable, total=count_params(unet, text_encoder, vae)[1], start_time=start_time,
    )
    logger.info("Verimlilik: %s", rec)
    logger.info("ST-LoRA eğitimi tamamlandı.")


def _save(injected, unet_layers, rank, alpha, image_size, adapt_text_encoder, step, output_dir):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    config = {
        "base_model_id": MODEL_ID,
        "rank": rank, "alpha": alpha, "image_size": image_size,
        "adapt_text_encoder": adapt_text_encoder,
        "unet_targets": list(UNET_TARGETS),
        "text_encoder_targets": list(TEXT_ENCODER_TARGETS),
    }
    # Tüm enjekte edilmiş katmanları tek state'te birleştir
    merged = {}
    for d in injected:
        merged.update(d)
    save_stlora(merged, config, os.path.join(ckpt_dir, "stlora.pt"))
    logger.info("ST-LoRA checkpoint kaydedildi: %s", ckpt_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Spectral-Temporal LoRA eğitimi (özgün yöntem)")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--csv_path", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./checkpoints_stlora")
    p.add_argument("--per_device_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_train_steps", type=int, default=TOTAL_TRAIN_STEPS)
    p.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--num_workers", type=int, default=_DEFAULT_WORKERS)
    p.add_argument("--save_every", type=int, default=2500)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--demo_mode", action="store_true")
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--no_text_encoder_lora", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_root=args.data_root, output_dir=args.output_dir,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_train_steps=args.max_train_steps, learning_rate=args.learning_rate,
        use_bf16=not args.no_bf16, num_workers=args.num_workers, save_every=args.save_every,
        csv_path=args.csv_path, image_size=args.image_size, max_samples=args.max_samples,
        demo_mode=args.demo_mode, use_gradient_checkpointing=not args.no_gradient_checkpointing,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        adapt_text_encoder=not args.no_text_encoder_lora, seed=args.seed,
    )
