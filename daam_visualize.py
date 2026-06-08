"""
daam_visualize.py — Çapraz-dikkat atıf haritaları (DAAM), diffusers 0.30 uyumlu
==============================================================================
Paylaşılan örnekteki gibi: üretilen göğüs röntgeni + seçili radyolojik terimler
için çapraz-dikkat ısı haritaları. `daam` kütüphanesi yeni diffusers ile
çalışmadığından, DAAM burada SIFIRDAN ve diffusers 0.30 uyumlu olarak
gerçeklenmiştir: cross-attention processor'ı değiştirilip dikkat olasılıkları
katmanlar/başlıklar/zaman-adımları boyunca toplanır, kelime token'larına göre
ısı haritasına dönüştürülüp görüntü üzerine bindirilir.

Çıktı: paper/figures/daam.png  → paper.tex otomatik gösterir.

Kullanım:
  python daam_visualize.py --checkpoint_dir .\checkpoints_lora_256\checkpoint-5000_fixed
"""

import argparse
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL_ID = "CompVis/stable-diffusion-v1-4"
AGG_RES = 64   # dikkat haritalarının toplandığı ortak çözünürlük (yüksek = daha keskin)

DEFAULT_ROWS = [
    "Left-sided pleural effusion and pacemaker | effusion, pacemaker",
    "Right-sided pleural effusion and right upper lobe pneumonia | effusion, pneumonia",
]


class DAAMStore:
    """Cross-attention olasılıklarını (cond yarısı) AGG_RES×AGG_RES×77 olarak toplar.

    min_side: yalnızca yerel çözünürlüğü >= min_side olan katmanlar toplanır
    (düşük çözünürlüklü/bulanık katmanları dışlayarak haritayı keskinleştirir).
    """
    def __init__(self, device, min_side=16):
        self.sum = torch.zeros(AGG_RES, AGG_RES, 77, device=device)
        self.count = 0
        self.min_side = min_side

    def reset(self):
        self.sum.zero_()
        self.count = 0

    def add(self, probs_cond):  # probs_cond: [q, 77]
        q, kv = probs_cond.shape
        side = int(math.sqrt(q))
        if side * side != q:
            return  # kare olmayan haritaları atla
        if side < self.min_side:
            return  # düşük çözünürlüklü (bulanık) katmanları dışla
        m = probs_cond.reshape(side, side, kv).permute(2, 0, 1).unsqueeze(0).float()  # [1,77,s,s]
        m = F.interpolate(m, size=(AGG_RES, AGG_RES), mode="bilinear", align_corners=False)
        self.sum += m.squeeze(0).permute(1, 2, 0)  # [R,R,77]
        self.count += 1


class DAAMAttnProcessor:
    """diffusers 0.30 legacy AttnProcessor + cross-attention yakalama."""
    def __init__(self, store: DAAMStore):
        self.store = store

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        is_cross = encoder_hidden_states is not None
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        batch_size, seq_len, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # ── Yakalama: sadece cross-attention, 77 metin token'ı ──────────────
        if is_cross and attention_probs.shape[-1] == 77:
            heads = attn.heads
            bh = attention_probs.shape[0]
            if bh == 2 * heads:          # CFG: [uncond, cond]
                probs = attention_probs.view(2, heads, attention_probs.shape[1], 77)[1]
            else:
                probs = attention_probs.view(-1, heads, attention_probs.shape[1], 77)[-1]
            probs = probs.mean(0)        # başlıklar üzerinden ortalama → [q, 77]
            self.store.add(probs)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def load_pipeline(checkpoint_dir, device, dtype):
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, StableDiffusionPipeline
    from transformers import CLIPTextModel, CLIPTokenizer
    ckpt = Path(checkpoint_dir)
    unet = UNet2DConditionModel.from_pretrained(ckpt / "unet")
    text_encoder = CLIPTextModel.from_pretrained(ckpt / "text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(ckpt / "tokenizer")
    vae = AutoencoderKL.from_pretrained(ckpt / "vae")
    scheduler = DDIMScheduler.from_pretrained(BASE_MODEL_ID, subfolder="scheduler")
    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=scheduler, safety_checker=None, feature_extractor=None,
    ).to(device, dtype=dtype)
    return pipe


def word_token_indices(tokenizer, prompt, word):
    """İstemdeki 'word' için 77-uzunluklu token dizisindeki konumlar."""
    enc = tokenizer(prompt, padding="max_length", max_length=77, truncation=True)
    ids = enc["input_ids"]
    toks = tokenizer.convert_ids_to_tokens(ids)
    wl = word.lower()
    idxs = []
    for i, t in enumerate(toks):
        clean = t.replace("</w>", "").lower()
        if clean and (clean in wl or wl in clean):
            idxs.append(i)
    return idxs


def parse_rows(rows):
    out = []
    for r in rows:
        if "|" in r:
            p, w = r.split("|", 1)
            words = [x.strip() for x in w.split(",") if x.strip()]
        else:
            p, words = r, []
        out.append((p.strip(), words))
    return out


def main():
    ap = argparse.ArgumentParser(description="DAAM çapraz-dikkat atıf figürü (custom)")
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--row", action="append", dest="rows", default=None)
    ap.add_argument("--gen_size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="paper/figures/daam.png")
    ap.add_argument("--min_side", type=int, default=16,
                    help="Yalnızca >= bu çözünürlükteki cross-attn katmanlarını topla (keskinlik)")
    ap.add_argument("--pct", type=float, default=0.6,
                    help="Isı haritası yüzdelik eşiği (altı bastırılır; kontrast)")
    ap.add_argument("--gamma", type=float, default=1.5,
                    help="Isı haritası gamma (>1 = tepe noktaları keskinleştirir)")
    args = ap.parse_args()

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = parse_rows(args.rows if args.rows else DEFAULT_ROWS)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    logger.info("Pipeline yükleniyor: %s", args.checkpoint_dir)
    pipe = load_pipeline(args.checkpoint_dir, device, dtype)

    store = DAAMStore(device, min_side=args.min_side)
    pipe.unet.set_attn_processor(DAAMAttnProcessor(store))   # tüm attn'lara, sadece cross yakalanır

    max_words = max((len(w) for _, w in rows), default=1)
    ncols = 1 + max_words
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.8 * ncols, 2.8 * nrows), squeeze=False)

    for ri, (prompt, words) in enumerate(rows):
        logger.info("[%d/%d] '%s' | %s", ri + 1, nrows, prompt, words)
        store.reset()
        gen = torch.Generator(device=pipe.device).manual_seed(args.seed + ri)
        with torch.no_grad():
            result = pipe(prompt=prompt, num_inference_steps=args.steps,
                          guidance_scale=args.cfg, height=args.gen_size, width=args.gen_size,
                          generator=gen)
        image = np.array(result.images[0].convert("L"))
        agg = (store.sum / max(store.count, 1)).cpu().float()   # [R,R,77]

        axes[ri][0].imshow(image, cmap="gray")
        axes[ri][0].set_title(prompt, fontsize=7)
        axes[ri][0].axis("off")

        for ci in range(1, ncols):
            ax = axes[ri][ci]; ax.axis("off")
            wi = ci - 1
            if wi >= len(words):
                continue
            w = words[wi]
            idxs = word_token_indices(pipe.tokenizer, prompt, w)
            if not idxs:
                ax.imshow(image, cmap="gray"); ax.set_title(f"{w} (token yok)", fontsize=7); continue
            heat = agg[:, :, idxs].sum(-1)                       # [R,R]
            # Yüzdelik eşik: alt %pct bastırılır → tepe bölgeler lokalize olur
            thr = torch.quantile(heat.flatten(), args.pct)
            heat = (heat - thr).clamp(min=0)
            heat = heat / (heat.max() + 1e-8)
            heat = heat ** args.gamma                            # gamma: tepe keskinleştirme
            heat = F.interpolate(heat[None, None], size=image.shape, mode="bilinear",
                                 align_corners=False)[0, 0].numpy()
            ax.imshow(image, cmap="gray")
            ax.imshow(heat, cmap="jet", alpha=0.5)
            ax.set_title(w, fontsize=9)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    logger.info("DAAM figürü kaydedildi: %s", args.output)


if __name__ == "__main__":
    main()
