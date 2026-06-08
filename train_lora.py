"""
Resource-Efficient LoRA Fine-tuning — Katkı modeli
===================================================
RoentGen (SD v1.4) full fine-tuning ~1 milyar parametre eğitir → optimizer
durumu tek başına ~16 GB → 12 GB'a SIĞMAZ (bu yüzden baseline 256px'e düşüyor).

Bu script LoRA (Low-Rank Adaptation) kullanır:
  - Base UNet + text encoder + VAE DONUK kalır (bf16).
  - Yalnızca küçük LoRA adapter matrisleri eğitilir (~birkaç milyon param).
  - Optimizer durumu minik → 12 GB'da 512×512 SIĞAR.
  - Daha az parametre → küçük tıbbi veride daha az overfitting (makale lim. #4).

Makalenin ana bulgusu "U-Net + CLIP text encoder'ı BİRLİKTE eğitmek en iyisi"
idi. Burada ikisine de LoRA uygulanabilir (--lora_text_encoder) → "joint
adaptation faydasını verimli şekilde kurtarabilir miyiz?" sorusu.

Kayıt: LoRA adapterleri base'e MERGE edilip standart diffusers formatında
(unet/, text_encoder/, tokenizer/, vae/) kaydedilir → evaluate.py / inference.py
/ compare.py hiç değişmeden çalışır.

Kurulum:  pip install peft
Örnek (4070, 512px):
  python train_lora.py --data_root .\data --image_size 512 --max_train_steps 3000 \
      --max_samples 4000 --lora_rank 8 --lora_text_encoder --save_every 1000
"""

import argparse
import contextlib
import logging
import os
import platform

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

import time

from dataset import create_dataloader
from run_logging import LossLogger, log_efficiency, reset_peak_vram, count_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "CompVis/stable-diffusion-v1-4"
LEARNING_RATE = 1e-4          # LoRA genelde full-FT'den biraz yüksek LR sever
TOTAL_TRAIN_STEPS = 3_000
DEMO_IMAGE_SIZE = 256
DEMO_MAX_SAMPLES = 256
DEMO_MAX_STEPS = 100
_DEFAULT_WORKERS = 0 if platform.system() == "Windows" else 4

# LoRA hedef modülleri (SD v1.4 attention katmanları)
UNET_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]
TEXT_ENCODER_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]


def setup_lora_model(
    device: torch.device,
    weight_dtype: torch.dtype,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    adapt_text_encoder: bool,
    use_gradient_checkpointing: bool,
    use_rslora: bool = False,
):
    """SD bileşenlerini yükle, base'i dondur, LoRA adapterlerini ekle.

    Standart diffusers-LoRA deseni: donuk base bf16, LoRA paramları fp32,
    compute autocast(bf16) ile.

    use_rslora: rank-stabilized LoRA (α/√r ölçekleme). Yüksek rank'te gradient
    collapse'i önler (Kalajdzievski 2023). peft desteklemezse standart LoRA'ya düşer.
    """
    from peft import LoraConfig, get_peft_model

    def _lora_config(targets):
        base = dict(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=targets)
        if use_rslora:
            try:
                cfg = LoraConfig(use_rslora=True, **base)
                logger.info("rsLoRA aktif (α/√r ölçekleme)")
                return cfg
            except TypeError:
                logger.warning("peft sürümü use_rslora desteklemiyor → standart LoRA. (pip install -U peft)")
        return LoraConfig(**base)

    logger.info("Model yükleniyor: %s", MODEL_ID)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    # Tüm base ağırlıkları dondur
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()

    if use_gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        logger.info("UNet gradient checkpointing açık")

    # Base'i hedef cihaz/dtype'a taşı (donuk → bf16 bellek için)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    # ── UNet'e LoRA ──────────────────────────────────────────────────────────
    unet = get_peft_model(unet, _lora_config(UNET_TARGET_MODULES))
    unet.print_trainable_parameters()

    # ── (opsiyonel) Text encoder'a LoRA — makalenin joint-FT bulgusu ─────────
    if adapt_text_encoder:
        text_encoder = get_peft_model(text_encoder, _lora_config(TEXT_ENCODER_TARGET_MODULES))
        text_encoder.print_trainable_parameters()
        text_encoder.train()
    else:
        text_encoder.eval()

    unet.train()

    # LoRA paramlarını fp32'ye çek (kararlı güncelleme; optimizer durumu yine minik)
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if adapt_text_encoder:
        for p in text_encoder.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    return vae, unet, text_encoder, tokenizer, noise_scheduler


def train(
    data_root: str,
    output_dir: str = "./checkpoints_lora",
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_train_steps: int = TOTAL_TRAIN_STEPS,
    learning_rate: float = LEARNING_RATE,
    use_bf16: bool = True,
    num_workers: int = _DEFAULT_WORKERS,
    save_every: int = 1000,
    csv_path: str = None,
    use_augmented: bool = False,
    image_size: int = 512,
    max_samples: int = None,
    demo_mode: bool = False,
    use_gradient_checkpointing: bool = True,
    lora_rank: int = 8,
    lora_alpha: int = None,
    lora_dropout: float = 0.0,
    adapt_text_encoder: bool = True,
    use_rslora: bool = False,
    seed: int = 42,
):
    """LoRA ile parametre-verimli fine-tuning."""
    os.makedirs(output_dir, exist_ok=True)

    # Tekrarlanabilirlik (baseline ile adil kıyas için aynı seed)
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Seed: %d", seed)

    if lora_alpha is None:
        lora_alpha = lora_rank   # yaygın varsayılan: alpha = rank

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
    logger.info("Cihaz: %s | Base dtype: %s | LoRA rank: %s | text encoder LoRA: %s",
                device, weight_dtype, lora_rank, adapt_text_encoder)

    vae, unet, text_encoder, tokenizer, noise_scheduler = setup_lora_model(
        device=device,
        weight_dtype=weight_dtype,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        adapt_text_encoder=adapt_text_encoder,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_rslora=use_rslora,
    )

    train_loader = create_dataloader(
        data_root=data_root,
        batch_size=per_device_batch_size,
        num_workers=num_workers,
        csv_path=csv_path,
        use_augmented=use_augmented,
        image_size=image_size,
        max_samples=max_samples,
        split="train",
    )
    if len(train_loader) == 0:
        raise RuntimeError("DataLoader 0 batch üretti. batch/max_samples ayarını kontrol et.")

    # Yalnızca LoRA paramları eğitilir
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if adapt_text_encoder:
        trainable_params += [p for p in text_encoder.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    logger.info("Eğitilen LoRA parametre sayısı: %s (~%.2fM)", f"{n_trainable:,}", n_trainable / 1e6)

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    global_step = 0
    accum_loss = 0.0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    use_autocast = device.type == "cuda" and weight_dtype != torch.float32

    loss_csv = LossLogger(os.path.join(output_dir, "metrics.csv"))
    reset_peak_vram()
    start_time = time.time()

    logger.info("Eğitim başlıyor: hedef %s adım", max_train_steps)
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

            loss = F.mse_loss(noise_pred.float(), noise.float())
            accum_loss += loss.item()
            (loss / gradient_accumulation_steps).backward()
            micro_step += 1

            if micro_step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_loss = accum_loss / gradient_accumulation_steps
                logger.info("Step %s/%s loss=%.6f", global_step, max_train_steps, avg_loss)
                loss_csv.log(global_step, round(avg_loss, 6))
                accum_loss = 0.0

                if global_step % save_every == 0:
                    _save_merged(unet, text_encoder, tokenizer, vae, adapt_text_encoder, global_step, output_dir)

                if global_step >= max_train_steps:
                    done = True
                    break

    _save_merged(unet, text_encoder, tokenizer, vae, adapt_text_encoder, global_step, output_dir)
    loss_csv.close()

    method = "lora_unet_te" if adapt_text_encoder else "lora_unet"
    _, total = count_params(unet, text_encoder, vae)
    eff_batch = per_device_batch_size * gradient_accumulation_steps
    rec = log_efficiency(
        output_dir=output_dir, method=f"{method}_r{lora_rank}", image_size=image_size,
        steps=global_step, effective_batch=eff_batch,
        trainable=n_trainable, total=total, start_time=start_time,
    )
    logger.info("Verimlilik: %s", rec)
    logger.info("Eğitim tamamlandı (%s adım, %.2fM eğitilen param).", global_step, n_trainable / 1e6)


def _save_merged(unet, text_encoder, tokenizer, vae, adapt_text_encoder, step, output_dir):
    """LoRA'yı base'e merge edip STANDART diffusers formatında kaydet.

    merge_and_unload(): LoRA katmanlarını söküp orijinal modül yapısını DOĞRU
    anahtar adlarıyla (to_q.weight vb.) geri verir → vanilla UNet yükleyebilir.
    (Eski merge_adapter+get_base_model yolu 'to_q.base_layer.weight' kaydedip
    yüklemeyi bozuyordu.) Eğitim modelini bozmamak için DERİN KOPYA üzerinde yapılır.
    """
    import copy

    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── UNet ─────────────────────────────────────────────────────────────────
    unet_merged = copy.deepcopy(unet).merge_and_unload()
    unet_merged.save_pretrained(os.path.join(ckpt_dir, "unet"))
    del unet_merged

    # ── Text encoder ─────────────────────────────────────────────────────────
    if adapt_text_encoder and hasattr(text_encoder, "merge_and_unload"):
        te_merged = copy.deepcopy(text_encoder).merge_and_unload()
        te_merged.save_pretrained(os.path.join(ckpt_dir, "text_encoder"))
        del te_merged
    else:
        text_encoder.save_pretrained(os.path.join(ckpt_dir, "text_encoder"))

    tokenizer.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))
    vae.save_pretrained(os.path.join(ckpt_dir, "vae"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Checkpoint (merged) kaydedildi: %s", ckpt_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Resource-Efficient LoRA fine-tuning (katkı modeli)")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--csv_path", type=str, default=None)
    p.add_argument("--use_augmented", action="store_true")
    p.add_argument("--output_dir", type=str, default="./checkpoints_lora")
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_train_steps", type=int, default=TOTAL_TRAIN_STEPS)
    p.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--num_workers", type=int, default=_DEFAULT_WORKERS)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--demo_mode", action="store_true")
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--lora_rank", type=int, default=8, help="LoRA rank (4/8/16/32 ile ablasyon)")
    p.add_argument("--lora_alpha", type=int, default=None, help="Varsayılan: rank'e eşit")
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--no_text_encoder_lora", action="store_true",
                   help="Sadece UNet'e LoRA (varsayılan: UNet + text encoder = joint)")
    p.add_argument("--use_rslora", action="store_true",
                   help="Rank-stabilized LoRA (α/√r). Yüksek rank'te gradient collapse'i önler.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_root=args.data_root,
        output_dir=args.output_dir,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_train_steps=args.max_train_steps,
        learning_rate=args.learning_rate,
        use_bf16=not args.no_bf16,
        num_workers=args.num_workers,
        save_every=args.save_every,
        csv_path=args.csv_path,
        use_augmented=args.use_augmented,
        image_size=args.image_size,
        max_samples=args.max_samples,
        demo_mode=args.demo_mode,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        adapt_text_encoder=not args.no_text_encoder_lora,
        use_rslora=args.use_rslora,
        seed=args.seed,
    )
