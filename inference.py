"""
RoentGen – Çıkarım (Inference) Scripti
========================================
Eğitilmiş fine-tuned modelden sentetik göğüs röntgeni üretimi.

Blueprint §6 kuralları:
  • Scheduler:  PNDMScheduler
  • Inference steps: 75
  • CFG scale: 4.0
  • Çıktı çözünürlüğü: 512×512
  • Safety checker: devre dışı
"""

import os
import argparse
import logging
from typing import List, Optional

import torch
from diffusers import (
    AutoencoderKL,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Blueprint §6 sabitleri
NUM_INFERENCE_STEPS = 75
GUIDANCE_SCALE = 4.0
IMAGE_SIZE = 512
BASE_MODEL_ID = "CompVis/stable-diffusion-v1-4"


def load_pipeline(
    checkpoint_dir: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> StableDiffusionPipeline:
    """
    Eğitilmiş checkpoint'ten StableDiffusionPipeline oluşturur.

    Checkpoint dizinindeki beklenen yapı (train.py tarafından kaydedilir):
        checkpoint-XXXXX/
        ├── unet/
        ├── text_encoder/
        ├── tokenizer/
        └── vae/

    Args:
        checkpoint_dir: train.py'nin kaydettiği checkpoint klasörü yolu.
        device:         "cuda" veya "cpu".
        dtype:          torch.float16 veya torch.bfloat16.

    Returns:
        Kullanıma hazır StableDiffusionPipeline.
    """
    logger.info(f"Checkpoint yükleniyor: {checkpoint_dir}")

    # Fine-tuned bileşenler
    unet = UNet2DConditionModel.from_pretrained(os.path.join(checkpoint_dir, "unet"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(checkpoint_dir, "text_encoder"))
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(checkpoint_dir, "tokenizer"))
    vae = AutoencoderKL.from_pretrained(os.path.join(checkpoint_dir, "vae"))

    # PNDMScheduler (blueprint §6)
    scheduler = PNDMScheduler.from_pretrained(BASE_MODEL_ID, subfolder="scheduler")

    # Pipeline oluştur — safety_checker devre dışı (blueprint §4)
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    )

    pipeline = pipeline.to(device, dtype=dtype)
    logger.info("Pipeline hazır")
    return pipeline


def generate(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    num_images: int = 1,
    seed: Optional[int] = None,
) -> list:
    """
    Verilen metin prompt'undan sentetik göğüs röntgeni üretir.

    Args:
        pipeline:    Yüklenmiş StableDiffusionPipeline.
        prompt:      Radyolojik metin (ör. "Right lower lobe pneumonia").
        num_images:  Üretilecek görüntü sayısı.
        seed:        Tekrarlanabilirlik için rastgele tohum.

    Returns:
        PIL Image listesi.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    result = pipeline(
        prompt=prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        generator=generator,
    )

    return result.images


def generate_and_save(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    output_dir: str = "./generated",
    num_images: int = 1,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Görüntü üretir ve diske PNG olarak kaydeder.

    Returns:
        Kaydedilen dosya yollarının listesi.
    """
    os.makedirs(output_dir, exist_ok=True)
    images = generate(pipeline, prompt, num_images=num_images, seed=seed)

    saved_paths = []
    for i, img in enumerate(images):
        # Dosya adını prompt'tan türet (güvenli karakterler)
        safe_name = "".join(c if c.isalnum() or c in " _-" else "" for c in prompt)
        safe_name = safe_name.strip().replace(" ", "_")[:80]
        fname = f"{safe_name}_{i}.png"
        fpath = os.path.join(output_dir, fname)
        img.save(fpath)
        saved_paths.append(fpath)
        logger.info(f"Kaydedildi: {fpath}")

    return saved_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="RoentGen – Sentetik CXR Üretimi (Inference)")

    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Eğitilmiş model checkpoint dizini (ör. ./checkpoints/checkpoint-60000)")
    parser.add_argument("--prompt", type=str, default="Right lower lobe pneumonia",
                        help="Radyolojik metin prompt'u")
    parser.add_argument("--num_images", type=int, default=1,
                        help="Üretilecek görüntü sayısı")
    parser.add_argument("--output_dir", type=str, default="./generated",
                        help="Çıktı görüntülerinin kaydedileceği dizin")
    parser.add_argument("--seed", type=int, default=42,
                        help="Rastgele tohum (tekrarlanabilirlik)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--bf16", action="store_true",
                        help="bfloat16 kullan (varsayılan: float16)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    # Pipeline yükle
    pipe = load_pipeline(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        dtype=dtype,
    )

    # Görüntü üret ve kaydet
    paths = generate_and_save(
        pipeline=pipe,
        prompt=args.prompt,
        output_dir=args.output_dir,
        num_images=args.num_images,
        seed=args.seed,
    )

    print(f"\n{'='*50}")
    print(f"Üretilen {len(paths)} görüntü:")
    for p in paths:
        print(f"  → {p}")
    print(f"Ayarlar: steps={NUM_INFERENCE_STEPS}, CFG={GUIDANCE_SCALE}, size={IMAGE_SIZE}x{IMAGE_SIZE}")
