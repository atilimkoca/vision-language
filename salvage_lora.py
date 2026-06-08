"""
salvage_lora.py — Bozuk LoRA checkpoint'ini onarır (yeniden eğitim GEREKTİRMEZ)
==============================================================================
Eski train_lora.py'deki kayıt hatası, LoRA-hedefli katmanları PEFT iç
isimlendirmesiyle ('...to_q.base_layer.weight') kaydetti; vanilla UNet
'...to_q.weight' beklediği için yüklenemiyordu. AMA ağırlıklar DOĞRU (merge
edilmiş haldeydi) — sadece anahtar adları yanlış.

Bu script: kaydedilmiş state_dict'i okur, '.base_layer.' → '.' yapar, artık
gereksiz 'lora_*' anahtarlarını atar, temiz bir checkpoint olarak yeniden
kaydeder. Orijinali bozmaz; çıktı <checkpoint>_fixed/ klasörüne yazılır.

Kullanım:
  python salvage_lora.py --checkpoint .\checkpoints_lora_256\checkpoint-5000
  # → .\checkpoints_lora_256\checkpoint-5000_fixed  (compare.py bunu kullanır)
"""

import argparse
import glob
import os
import shutil

import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel

BASE_MODEL_ID = "CompVis/stable-diffusion-v1-4"


def _load_state(folder: str) -> dict:
    """Bir diffusers/transformers klasöründeki tüm ağırlıkları tek dict'e yükler."""
    sd = {}
    st_files = glob.glob(os.path.join(folder, "*.safetensors"))
    if st_files:
        from safetensors.torch import load_file
        for f in st_files:
            sd.update(load_file(f))
    else:
        bin_files = glob.glob(os.path.join(folder, "*.bin"))
        if not bin_files:
            raise FileNotFoundError(f"Ağırlık dosyası bulunamadı: {folder}")
        for f in bin_files:
            sd.update(torch.load(f, map_location="cpu"))
    return sd


def _clean_keys(sd: dict) -> dict:
    """PEFT isimlendirmesini vanilla isimlendirmeye çevirir."""
    out = {}
    for k, v in sd.items():
        if "lora_" in k:                       # lora_A/lora_B → merged'e dahil, atılır
            continue
        nk = k.replace(".base_layer.", ".")
        if nk.startswith("base_model.model."):
            nk = nk[len("base_model.model."):]
        out[nk] = v
    return out


def _salvage_module(src_folder, model, name):
    sd = _clean_keys(_load_state(src_folder))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[{name}] yüklendi | eksik={len(missing)} beklenmeyen={len(unexpected)}")
    if missing:
        print(f"  ⚠ UYARI: {len(missing)} anahtar eksik (ör. {missing[:3]}) — "
              f"bu ağırlıklar base SD'den geldi, onarım eksik olabilir.")
    return model


def main():
    ap = argparse.ArgumentParser(description="Bozuk LoRA checkpoint onarımı")
    ap.add_argument("--checkpoint", required=True, help="ör. .\\checkpoints_lora_256\\checkpoint-5000")
    ap.add_argument("--out", default=None, help="Varsayılan: <checkpoint>_fixed")
    args = ap.parse_args()

    ckpt = args.checkpoint.rstrip("\\/")
    out = args.out or (ckpt + "_fixed")
    os.makedirs(out, exist_ok=True)

    # ── UNet ──────────────────────────────────────────────────────────────────
    print("UNet onarılıyor...")
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_ID, subfolder="unet")
    unet = _salvage_module(os.path.join(ckpt, "unet"), unet, "UNet")
    unet.save_pretrained(os.path.join(out, "unet"))

    # ── Text encoder ──────────────────────────────────────────────────────────
    te_src = os.path.join(ckpt, "text_encoder")
    print("Text encoder onarılıyor...")
    te = CLIPTextModel.from_pretrained(BASE_MODEL_ID, subfolder="text_encoder")
    te = _salvage_module(te_src, te, "TextEncoder")
    te.save_pretrained(os.path.join(out, "text_encoder"))

    # ── tokenizer + vae: orijinalden kopyala (LoRA bunlara dokunmadı) ─────────
    for sub in ("tokenizer", "vae"):
        src = os.path.join(ckpt, sub)
        dst = os.path.join(out, sub)
        if os.path.isdir(src):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            # vae yoksa base'den al
            if sub == "vae":
                AutoencoderKL.from_pretrained(BASE_MODEL_ID, subfolder="vae").save_pretrained(dst)

    print(f"\n✓ Onarıldı → {out}")
    print("Şimdi compare.py'de bu klasörü kullan:")
    print(f"  --model lora_256:{out}")


if __name__ == "__main__":
    main()
