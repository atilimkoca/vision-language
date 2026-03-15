"""
RoentGen Stable Diffusion fine-tuning script.

This keeps the original training idea intact, but also supports lighter
concept/demo runs through:
  - smaller image sizes
  - optional dataset subsampling
  - shorter runs via demo_mode
  - optional disabling of gradient checkpointing for speed
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

from dataset import create_dataloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "CompVis/stable-diffusion-v1-4"
LEARNING_RATE = 5e-5
TOTAL_TRAIN_STEPS = 6_000
EFFECTIVE_BATCH_SIZE = 256
DEMO_IMAGE_SIZE = 256
DEMO_MAX_SAMPLES = 256
DEMO_MAX_STEPS = 200
DEMO_GRAD_ACCUM = 4
_DEFAULT_WORKERS = 0 if platform.system() == "Windows" else 4


def setup_model(
    device: torch.device,
    dtype: torch.dtype,
    use_gradient_checkpointing: bool = True,
):
    """Load Stable Diffusion components and move them to the target device."""
    logger.info("Loading model: %s", MODEL_ID)

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    vae.requires_grad_(False)
    vae.eval()
    logger.info("VAE frozen")

    if use_gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        logger.info("UNet gradient checkpointing enabled")
    else:
        logger.info("UNet gradient checkpointing disabled")

    unet.train()
    text_encoder.train()

    vae.to(device, dtype=dtype)
    unet.to(device, dtype=dtype)
    text_encoder.to(device, dtype=dtype)

    return vae, unet, text_encoder, tokenizer, noise_scheduler


def train(
    data_root: str,
    output_dir: str = "./checkpoints",
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    max_train_steps: int = TOTAL_TRAIN_STEPS,
    learning_rate: float = LEARNING_RATE,
    use_bf16: bool = True,
    num_workers: int = _DEFAULT_WORKERS,
    save_every: int = 5000,
    resume_step: int = 0,
    csv_path: str = None,
    use_augmented: bool = False,
    freeze_text_encoder: bool = True,
    image_size: int = 512,
    max_samples: int = None,
    demo_mode: bool = False,
    use_gradient_checkpointing: bool = True,
):
    """Run training."""
    os.makedirs(output_dir, exist_ok=True)

    if demo_mode:
        image_size = min(image_size, DEMO_IMAGE_SIZE)
        max_samples = DEMO_MAX_SAMPLES if max_samples is None else min(max_samples, DEMO_MAX_SAMPLES)
        max_train_steps = min(max_train_steps, DEMO_MAX_STEPS)
        gradient_accumulation_steps = min(gradient_accumulation_steps, DEMO_GRAD_ACCUM)
        save_every = min(save_every, max_train_steps)
        use_gradient_checkpointing = False
        logger.info(
            "Demo mode enabled: image_size=%s, max_samples=%s, max_train_steps=%s, grad_accum=%s",
            image_size,
            max_samples,
            max_train_steps,
            gradient_accumulation_steps,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    logger.info("Device: %s", device)
    logger.info("Dtype: %s", dtype)

    vae, unet, text_encoder, tokenizer, noise_scheduler = setup_model(
        device=device,
        dtype=dtype,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    if freeze_text_encoder:
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        logger.info("Text encoder frozen")

    train_loader = create_dataloader(
        data_root=data_root,
        batch_size=per_device_batch_size,
        num_workers=num_workers,
        csv_path=csv_path,
        use_augmented=use_augmented,
        image_size=image_size,
        max_samples=max_samples,
    )
    if len(train_loader) == 0:
        raise RuntimeError("DataLoader produced zero batches. Lower batch size or increase max_samples.")

    if freeze_text_encoder:
        trainable_params = list(unet.parameters())
    else:
        trainable_params = list(unet.parameters()) + list(text_encoder.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    effective_batch_size = per_device_batch_size * gradient_accumulation_steps
    logger.info(
        "Effective batch size: %s x %s = %s (reference target: %s)",
        per_device_batch_size,
        gradient_accumulation_steps,
        effective_batch_size,
        EFFECTIVE_BATCH_SIZE,
    )

    global_step = resume_step
    accum_loss = 0.0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    use_autocast = device.type == "cuda" and dtype != torch.float32

    logger.info("Training starts: global_step=%s target=%s", global_step, max_train_steps)

    done = False
    while not done:
        for batch in train_loader:
            if global_step >= max_train_steps:
                done = True
                break

            pixel_values = batch["pixel_values"].to(device, dtype=dtype, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            if freeze_text_encoder:
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]
            else:
                encoder_hidden_states = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            autocast_context = (
                torch.cuda.amp.autocast(dtype=dtype)
                if use_autocast
                else contextlib.nullcontext()
            )
            with autocast_context:
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            loss = F.mse_loss(noise_pred.float(), noise.float())
            loss = loss / gradient_accumulation_steps
            loss.backward()

            accum_loss += loss.item()
            micro_step += 1

            if micro_step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_loss = accum_loss / gradient_accumulation_steps
                logger.info("Step %s/%s loss=%.6f", global_step, max_train_steps, avg_loss)
                accum_loss = 0.0

                if global_step % save_every == 0:
                    _save_checkpoint(unet, text_encoder, tokenizer, vae, optimizer, global_step, output_dir)

                if global_step >= max_train_steps:
                    done = True
                    break

    _save_checkpoint(unet, text_encoder, tokenizer, vae, optimizer, global_step, output_dir)
    logger.info("Training finished after %s optimizer steps", global_step)


def _save_checkpoint(unet, text_encoder, tokenizer, vae, optimizer, step, output_dir):
    """Save a checkpoint in diffusers component format."""
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    unet.save_pretrained(os.path.join(ckpt_dir, "unet"))
    text_encoder.save_pretrained(os.path.join(ckpt_dir, "text_encoder"))
    tokenizer.save_pretrained(os.path.join(ckpt_dir, "tokenizer"))
    vae.save_pretrained(os.path.join(ckpt_dir, "vae"))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))

    logger.info("Checkpoint saved: %s", ckpt_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="RoentGen Stable Diffusion fine-tuning")

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--use_augmented", action="store_true")

    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_train_steps", type=int, default=TOTAL_TRAIN_STEPS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--num_workers", type=int, default=_DEFAULT_WORKERS)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--resume_step", type=int, default=0)
    parser.add_argument("--no_freeze_text_encoder", action="store_true")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--demo_mode", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")

    return parser.parse_args()


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
        resume_step=args.resume_step,
        csv_path=args.csv_path,
        use_augmented=args.use_augmented,
        freeze_text_encoder=not args.no_freeze_text_encoder,
        image_size=args.image_size,
        max_samples=args.max_samples,
        demo_mode=args.demo_mode,
        use_gradient_checkpointing=not args.no_gradient_checkpointing,
    )
