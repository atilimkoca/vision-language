"""
evaluate.py — Makale Tablo 1, 3, 4 Metrikleri
============================================================
Tüm modelleri aynı metriklerle adil kıyaslar. Değerlendirilen modeller hepsi
standart diffusers (CLIP-SD) formatında checkpoint üretir:
  • baseline : train.py       → full fine-tuning
  • LoRA     : train_lora.py  → parametre-verimli adaptasyon
  • PRG      : train_prg.py   → LoRA + patoloji ödülü (merkez katkı)

Metrikler
  Tablo 1 — Kalite & Çeşitlilik
    • FID (clean-fid InceptionV3) + opsiyonel in-domain XRV DenseNet-121 FID
      (cleanfid yoksa ham InceptionV3 fallback)
    • MS-SSIM: AYNI prompt için n_repetitions üretimin pairwise MS-SSIM'i
               (grayscale; resmi eval/calculate_ms-ssim.py protokolü)

  Tablo 3 — Factual Correctness (RRG)
    • BLEU-4 / ROUGE-L / BERTScore / F1CheXbert
    Makalenin protokolü: sentetik görüntü → RRG (görüntü→rapor) modeli → yeni
    rapor → gerçek impression ile kıyas. RRG modeli verilmezse metrikler
    dürüstçe N/A bırakılır (uydurma self-comparison YAPILMAZ).

  Tablo 4 — Image-Text Retrieval (proxy)
    • CLIP Cosine Sim: üretilen görüntü ile orijinal metin arası benzerlik
      (üç modelde de çalışır; metin-sadakat için pratik gösterge)

Kullanım:
  python evaluate.py --checkpoint_dir ./checkpoints/checkpoint-3000 \\
      --data_root ./data --num_eval_samples 200 --model_name baseline_256
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL_ID = "CompVis/stable-diffusion-v1-4"

# ── Opsiyonel kütüphaneler ────────────────────────────────────────────────────
_MISSING = []
try:
    from transformers import CLIPModel, CLIPProcessor  # noqa: F401
except ImportError:
    _MISSING.append("transformers")

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except ImportError:
    _MISSING.append("nltk")

try:
    from rouge_score import rouge_scorer
except ImportError:
    _MISSING.append("rouge-score")

try:
    from bert_score import score as bert_score_fn
except ImportError:
    _MISSING.append("bert-score")

try:
    from pytorch_msssim import ms_ssim
except ImportError:
    _MISSING.append("pytorch-msssim")

try:
    from f1chexbert import F1CheXbert
    _HAS_F1CHEXBERT = True
except ImportError:
    _HAS_F1CHEXBERT = False

if _MISSING:
    logger.warning(
        "Eksik kütüphaneler: %s\nKurulum: pip install %s",
        ", ".join(_MISSING), " ".join(_MISSING),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Model tipi algılama ve generator soyutlaması
# ═════════════════════════════════════════════════════════════════════════════
def detect_model_type(checkpoint_dir: str) -> str:
    """Checkpoint dizininden model tipini ('sd' | 'biomedbert') algılar.

    baseline (train.py), LoRA (train_lora.py) ve PRG (train_prg.py) hepsi
    standart diffusers formatında CLIP-SD checkpoint'i üretir → 'sd'.
    """
    ckpt = Path(checkpoint_dir)
    te_cfg = ckpt / "text_encoder" / "config.json"
    if te_cfg.exists():
        with open(te_cfg, encoding="utf-8") as f:
            model_type = json.load(f).get("model_type", "").lower()
        if "bert" in model_type and "clip" not in model_type:
            return "biomedbert"
    return "sd"


class DiffusionGenerator:
    """SD (CLIP) ve BiomedBERT difüzyon modelleri için ortak üretici.

    Tek fark text encoder/tokenizer: CLIP → CLIPTextModel/CLIPTokenizer,
    BiomedBERT → AutoModel/AutoTokenizer (last_hidden_state, 768-dim, SD UNet
    cross-attention ile uyumlu). BiomedBERT tokenizer'ı eğitimdeki gibi 77
    token'a sabitlenir.
    """

    def __init__(self, checkpoint_dir, device, dtype, num_inference_steps, guidance_scale, model_type, gen_size=256):
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, StableDiffusionPipeline

        ckpt = Path(checkpoint_dir)
        if model_type == "biomedbert":
            from transformers import AutoModel, AutoTokenizer
            text_encoder = AutoModel.from_pretrained(ckpt / "text_encoder")
            tokenizer = AutoTokenizer.from_pretrained(ckpt / "tokenizer")
            tokenizer.model_max_length = 77   # eğitimle tutarlı
        else:
            from transformers import CLIPTextModel, CLIPTokenizer
            text_encoder = CLIPTextModel.from_pretrained(ckpt / "text_encoder")
            tokenizer = CLIPTokenizer.from_pretrained(ckpt / "tokenizer")

        unet = UNet2DConditionModel.from_pretrained(ckpt / "unet")
        vae = AutoencoderKL.from_pretrained(ckpt / "vae")
        scheduler = DDIMScheduler.from_pretrained(BASE_MODEL_ID, subfolder="scheduler")

        self.pipe = StableDiffusionPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            unet=unet, scheduler=scheduler,
            safety_checker=None, feature_extractor=None,
        ).to(device, dtype=dtype)
        self.steps = num_inference_steps
        self.cfg = guidance_scale
        self.gen_size = gen_size   # üretim çözünürlüğü = eğitim çözünürlüğü olmalı

    @property
    def device(self):
        return self.pipe.device

    def _one(self, prompt, seed):
        g = torch.Generator(device=self.pipe.device).manual_seed(seed)
        return self.pipe(
            prompt=prompt, num_inference_steps=self.steps, guidance_scale=self.cfg,
            height=self.gen_size, width=self.gen_size, generator=g,
        ).images[0]

    def generate(self, prompts, seed=42):
        return [self._one(p, seed + i) for i, p in enumerate(prompts)]

    def generate_repetitions(self, prompts, n_repetitions, seed=42):
        return [[self._one(p, seed + 1000 * i + r) for r in range(n_repetitions)]
                for i, p in enumerate(prompts)]


def build_generator(checkpoint_dir, device, dtype, num_inference_steps, guidance_scale, model_type, gen_size=256):
    """Model tipine göre uygun generator nesnesini döndürür (baseline / LoRA / PRG)."""
    logger.info("Generator tipi: difüzyon (%s) | üretim çözünürlüğü: %d", model_type, gen_size)
    return DiffusionGenerator(checkpoint_dir, device, dtype, num_inference_steps, guidance_scale, model_type, gen_size=gen_size)


# ═════════════════════════════════════════════════════════════════════════════
# Tablo 1: FID
# ═════════════════════════════════════════════════════════════════════════════
def _fid_cleanfid(real_files, fake_files, device, feat="inception") -> float:
    """clean-fid ile FID (resmi RoentGen eval/calculate_fid.py ile aynı kütüphane)."""
    from cleanfid import fid
    dev = torch.device(device)
    if feat == "clip":
        from cleanfid.clip_features import CLIP_fx, img_preprocess_clip
        feat_model = CLIP_fx("ViT-B/32", device=device)
        resize_fn = img_preprocess_clip
    else:
        feat_model = fid.build_feature_extractor(mode="clean", device=dev)
        resize_fn = None

    f1 = fid.get_files_features([str(x) for x in real_files], model=feat_model, mode="clean",
                                custom_fn_resize=resize_fn, device=dev, verbose=False, num_workers=0)
    f2 = fid.get_files_features([str(x) for x in fake_files], model=feat_model, mode="clean",
                                custom_fn_resize=resize_fn, device=dev, verbose=False, num_workers=0)
    return float(fid.fid_from_feats(f1, f2))


def _fid_xrv(real_files, fake_files, device) -> float:
    """In-domain FID: torchxrayvision DenseNet-121 (weights=all) özellikleri.

    Resmi eval/calculate_fid.py'deki XRV ön-işleme hattını birebir taşır.
    """
    import torchxrayvision as xrv
    from cleanfid import fid

    dev = torch.device(device)
    model = xrv.models.DenseNet(weights="densenet121-res224-all").to(dev)

    def rgb2gray(x):
        return transforms.functional.rgb_to_grayscale(x, 1)

    def xray_crop_center(img):
        y, x = img.shape[2:]
        crop = torch.min(torch.tensor([y, x]))
        sx = x // 2 - (crop // 2)
        sy = y // 2 - (crop // 2)
        return img[:, sy:sy + crop, sx:sx + crop]

    transform = transforms.Compose([
        transforms.Lambda(rgb2gray),
        transforms.Lambda(lambda x: ((2048 * (x / 255)) - 1024)),
        transforms.Lambda(xray_crop_center),
        transforms.Resize(224),
    ])

    class FeatureExtractor:
        def __init__(self, m, tf):
            self.model, self.transform = m, tf

        def __call__(self, x):
            return self.model.features2(self.transform(x))

    extractor = FeatureExtractor(model, transform)
    f1 = fid.get_files_features([str(x) for x in real_files], model=extractor, mode="clean", device=dev, verbose=False, num_workers=0)
    f2 = fid.get_files_features([str(x) for x in fake_files], model=extractor, mode="clean", device=dev, verbose=False, num_workers=0)
    return float(fid.fid_from_feats(f1, f2))


def _fid_raw_inception(real_files, fake_files, device) -> float:
    """cleanfid yoksa ham InceptionV3 fallback (makale FID'iyle birebir kıyaslanamaz)."""
    from torchvision.models import inception_v3, Inception_V3_Weights

    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    pre = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def feats(files, bs=16):
        out = []
        for i in range(0, len(files), bs):
            batch = torch.stack([pre(Image.open(p).convert("RGB")) for p in files[i:i + bs]]).to(device)
            with torch.no_grad():
                out.append(model(batch).cpu().numpy())
        return np.concatenate(out, 0)

    from scipy import linalg
    fr, ff = feats(real_files), feats(fake_files)
    mu_r, sig_r = fr.mean(0), np.cov(fr, rowvar=False)
    mu_f, sig_f = ff.mean(0), np.cov(ff, rowvar=False)
    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sig_r @ sig_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig_r + sig_f - 2.0 * covmean))


def compute_fid(real_files, fake_files, device, use_xrv=True) -> dict:
    """FID(ler) hesaplar. clean-fid varsa InceptionV3 (+ opsiyonel XRV), yoksa ham fallback."""
    out = {}
    logger.info("FID hesaplanıyor (%d gerçek, %d sentetik)", len(real_files), len(fake_files))
    try:
        out["FID"] = round(_fid_cleanfid(real_files, fake_files, device, "inception"), 4)
        logger.info("FID (clean-fid InceptionV3) = %.4f", out["FID"])
    except ImportError:
        logger.warning("cleanfid yok → ham InceptionV3 fallback (pip install clean-fid)")
        try:
            out["FID"] = round(_fid_raw_inception(real_files, fake_files, device), 4)
        except Exception as e:  # noqa: BLE001
            logger.error("FID hesaplanamadı: %s", e)
            out["FID"] = "N/A"

    if use_xrv:
        try:
            out["FID_XRV"] = round(_fid_xrv(real_files, fake_files, device), 4)
            logger.info("FID (in-domain XRV DenseNet-121) = %.4f", out["FID_XRV"])
        except ImportError:
            out["FID_XRV"] = "N/A (torchxrayvision + clean-fid gerekli)"
        except Exception as e:  # noqa: BLE001
            logger.error("XRV FID hesaplanamadı: %s", e)
            out["FID_XRV"] = "N/A"
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Tablo 1: MS-SSIM (intra-prompt çeşitlilik)
# ═════════════════════════════════════════════════════════════════════════════
def compute_ms_ssim_diversity(image_groups: List[List[Image.Image]], device: torch.device) -> float:
    """Aynı prompt'a ait tekrarların pairwise MS-SSIM ortalaması (grayscale).
    Düşük = yüksek çeşitlilik. (resmi eval/calculate_ms-ssim.py protokolü)"""
    from itertools import combinations

    to_tensor = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    scores = []
    for group in image_groups:
        if len(group) < 2:
            continue
        tensors = [to_tensor(img.convert("L")).to(device) for img in group]
        for i, j in combinations(range(len(tensors)), 2):
            scores.append(ms_ssim(tensors[i].unsqueeze(0), tensors[j].unsqueeze(0),
                                  data_range=1.0, size_average=True).item())
    return float(np.mean(scores)) if scores else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Tablo 3: metin metrikleri (gerçek RRG raporları üzerinde)
# ═════════════════════════════════════════════════════════════════════════════
def compute_bleu4(hyps, refs) -> float:
    smooth = SmoothingFunction().method1
    return corpus_bleu([[word_tokenize(r.lower())] for r in refs],
                       [word_tokenize(h.lower()) for h in hyps],
                       weights=(0.25,) * 4, smoothing_function=smooth)


def compute_rouge_l(hyps, refs) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(np.mean([scorer.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)]))


def compute_bertscore(hyps, refs, device) -> float:
    _, _, f1 = bert_score_fn(hyps, refs, lang="en", device=device, verbose=False)
    return float(f1.mean().item())


CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]


def compute_f1chexbert(hyps, refs, device) -> float:
    """Gerçek CheXbert micro-avg F1 (paket varsa), yoksa kural-tabanlı proxy."""
    if _HAS_F1CHEXBERT:
        logger.info("F1CheXbert: gerçek CheXbert modeli")
        _, _, _, chexbert_5 = F1CheXbert(device=device)(hyps, refs)
        return float(chexbert_5["micro avg"]["f1-score"])
    logger.warning("f1chexbert yok → kural-tabanlı PROXY (pip install f1chexbert ile gerçek skor)")
    tp = fp = fn = 0
    for h, r in zip(hyps, refs):
        hv = np.array([1 if l.lower() in h.lower() else 0 for l in CHEXPERT_LABELS])
        rv = np.array([1 if l.lower() in r.lower() else 0 for l in CHEXPERT_LABELS])
        tp += int((hv * rv).sum()); fp += int((hv * (1 - rv)).sum()); fn += int(((1 - hv) * rv).sum())
    p = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    return float(2 * p * rec / (p + rec + 1e-9))


# ═════════════════════════════════════════════════════════════════════════════
# Tablo 4: CLIP image-text cosine similarity
# ═════════════════════════════════════════════════════════════════════════════
def compute_clip_similarity(images, texts, device, clip_model_id="openai/clip-vit-large-patch14") -> float:
    logger.info("CLIP cosine similarity hesaplanıyor...")
    processor = CLIPProcessor.from_pretrained(clip_model_id)
    model = CLIPModel.from_pretrained(clip_model_id).to(device).eval()
    sims = []
    for i in range(0, len(images), 16):
        inputs = processor(text=texts[i:i + 16], images=images[i:i + 16],
                           return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            img_f = F.normalize(model.get_image_features(pixel_values=inputs["pixel_values"]), dim=-1)
            txt_f = F.normalize(model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]), dim=-1)
        sims.append((img_f * txt_f).sum(-1).cpu().numpy())
    return float(np.concatenate(sims).mean())


# ═════════════════════════════════════════════════════════════════════════════
# RRG: sentetik görüntü → rapor (Tablo 3 için; verilmezse N/A)
# ═════════════════════════════════════════════════════════════════════════════
def generate_reports(images, rrg_model_id, device) -> Optional[List[str]]:
    if rrg_model_id is None:
        return None
    try:
        from transformers import pipeline
        captioner = pipeline("image-to-text", model=rrg_model_id,
                             device=0 if (device == "cuda" and torch.cuda.is_available()) else -1)
    except Exception as e:  # noqa: BLE001
        logger.error("RRG modeli yüklenemedi (%s): %s", rrg_model_id, e)
        return None
    logger.info("RRG: %d görüntüden rapor üretiliyor (model=%s)", len(images), rrg_model_id)
    reports = []
    for img in images:
        out = captioner(img.convert("RGB"))
        reports.append(out[0].get("generated_text", "") if out else "")
    return reports


# ═════════════════════════════════════════════════════════════════════════════
# Ana değerlendirme
# ═════════════════════════════════════════════════════════════════════════════
def evaluate(
    checkpoint_dir: str,
    data_root: str,
    num_eval_samples: int = 200,
    output_dir: str = "./eval_results",
    num_inference_steps: int = 20,
    guidance_scale: float = 4.0,
    seed: int = 42,
    device: str = "cuda",
    skip_fid: bool = False,
    use_xrv_fid: bool = True,
    model_name: str = "model",
    model_type: Optional[str] = None,
    rrg_model_id: Optional[str] = None,
    generated_reports_file: Optional[str] = None,
    num_diversity_prompts: int = 20,
    n_repetitions: int = 4,
    eval_split: str = "test",
    gen_size: int = 256,
) -> dict:

    os.makedirs(output_dir, exist_ok=True)
    gen_dir = os.path.join(output_dir, f"generated_{model_name}")
    os.makedirs(gen_dir, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if dev.type == "cuda" else torch.float32

    if model_type is None:
        model_type = detect_model_type(checkpoint_dir)
    logger.info("Model: %s | tip: %s | checkpoint: %s", model_name, model_type, checkpoint_dir)

    # ── Veri ──────────────────────────────────────────────────────────────
    from dataset import build_metadata
    meta = build_metadata(data_root=data_root, max_samples=num_eval_samples, sample_seed=seed, split=eval_split)
    prompts = meta["impression"].tolist()
    image_paths = meta["image_path"].tolist()
    logger.info("%d örnek kullanılacak (split=%s)", len(prompts), eval_split)

    # ── Üretim ────────────────────────────────────────────────────────────
    generator = build_generator(checkpoint_dir, dev, dtype, num_inference_steps, guidance_scale, model_type, gen_size=gen_size)
    logger.info("Görüntü üretiliyor...")
    fake_images = generator.generate(prompts, seed)

    fake_files = []
    for i, img in enumerate(fake_images):
        fp = os.path.join(gen_dir, f"gen_{i:04d}.png")
        img.save(fp)
        fake_files.append(fp)
    logger.info("%d görüntü kaydedildi: %s", len(fake_files), gen_dir)

    results = {"model": model_name, "model_type": model_type, "n_samples": len(prompts)}

    # ── Tablo 1: FID ──────────────────────────────────────────────────────
    if not skip_fid:
        results.update(compute_fid(image_paths, fake_files, dev, use_xrv=use_xrv_fid))
    else:
        results["FID"] = "N/A"

    # ── Tablo 1: MS-SSIM ──────────────────────────────────────────────────
    if "pytorch-msssim" not in _MISSING:
        n_div = min(num_diversity_prompts, len(prompts))
        logger.info("MS-SSIM: %d prompt × %d tekrar...", n_div, n_repetitions)
        groups = generator.generate_repetitions(prompts[:n_div], n_repetitions, seed)
        div_dir = os.path.join(output_dir, f"diversity_{model_name}")
        os.makedirs(div_dir, exist_ok=True)
        for pi, grp in enumerate(groups):
            for ri, img in enumerate(grp):
                img.save(os.path.join(div_dir, f"prompt{pi:03d}_{ri}.png"))
        results["MS-SSIM"] = round(compute_ms_ssim_diversity(groups, dev), 4)
        logger.info("MS-SSIM = %.4f (düşük = çeşitli)", results["MS-SSIM"])
    else:
        results["MS-SSIM"] = "N/A"

    # ── Tablo 3: RRG metinleri ────────────────────────────────────────────
    if generated_reports_file is not None:
        with open(generated_reports_file, encoding="utf-8") as f:
            hyps = [line.strip() for line in f]
        n = min(len(hyps), len(prompts))
        hyps, refs = hyps[:n], prompts[:n]
    else:
        hyps = generate_reports(fake_images, rrg_model_id, device)
        refs = prompts

    if hyps is None:
        logger.warning("RRG modeli yok → Tablo 3 metin metrikleri N/A "
                       "(--rrg_model_id veya --generated_reports_file ile aktif edin).")
        for k in ("BLEU-4", "ROUGE-L", "BERTScore", "F1CheXbert"):
            results[k] = "N/A (RRG modeli gerekli)"
    else:
        with open(os.path.join(output_dir, f"reports_{model_name}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(hyps))
        results["BLEU-4"] = round(compute_bleu4(hyps, refs), 4) if "nltk" not in _MISSING else "N/A"
        results["ROUGE-L"] = round(compute_rouge_l(hyps, refs), 4) if "rouge-score" not in _MISSING else "N/A"
        results["BERTScore"] = round(compute_bertscore(hyps, refs, device), 4) if "bert-score" not in _MISSING else "N/A"
        results["F1CheXbert"] = round(compute_f1chexbert(hyps, refs, device), 4)

    # ── Tablo 4: CLIP-sim ─────────────────────────────────────────────────
    if "transformers" not in _MISSING:
        results["CLIP_sim"] = round(compute_clip_similarity(fake_images, prompts, dev), 4)
        logger.info("CLIP cosine sim = %.4f", results["CLIP_sim"])
    else:
        results["CLIP_sim"] = "N/A"

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "metrics_summary.csv")
    _append_row(csv_path, results)

    logger.info("\n" + "=" * 60)
    logger.info("SONUÇLAR (%s)", model_name)
    for k, v in results.items():
        if k not in ("model", "model_type", "n_samples"):
            logger.info("  %-22s: %s", k, v)
    logger.info("=" * 60)
    logger.info("CSV: %s", csv_path)
    return results


def _append_row(csv_path: str, row: dict) -> None:
    """CSV'ye satır ekler; yeni anahtarlar çıkarsa başlığı birleştirerek yeniden yazar."""
    existing = []
    fieldnames = list(row.keys())
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing = list(reader)
            for k in (reader.fieldnames or []):
                if k not in fieldnames:
                    fieldnames.append(k)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in existing:
            writer.writerow(r)
        writer.writerow(row)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Değerlendirme (baseline / LoRA / PRG)")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--num_eval_samples", type=int, default=200)
    parser.add_argument("--output_dir", default="./eval_results")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--no_xrv_fid", action="store_true", help="In-domain XRV FID'i atla")
    parser.add_argument("--model_name", default="model")
    parser.add_argument("--model_type", default=None, choices=["sd", "biomedbert"],
                        help="Belirtilmezse checkpoint'ten otomatik algılanır (baseline/LoRA/PRG = sd)")
    parser.add_argument("--rrg_model_id", default=None, help="Görüntü→rapor HF model id (Tablo 3)")
    parser.add_argument("--generated_reports_file", default=None)
    parser.add_argument("--num_diversity_prompts", type=int, default=20)
    parser.add_argument("--n_repetitions", type=int, default=4)
    parser.add_argument("--eval_split", choices=["test", "train", "all"], default="test",
                        help="Evaluation split. Use test=p19 for reported results.")
    parser.add_argument("--gen_size", type=int, default=256,
                        help="Üretim çözünürlüğü = eğitim çözünürlüğü (256 veya 512).")
    args = parser.parse_args()

    evaluate(
        checkpoint_dir=args.checkpoint_dir, data_root=args.data_root,
        num_eval_samples=args.num_eval_samples, output_dir=args.output_dir,
        num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
        seed=args.seed, device=args.device, skip_fid=args.skip_fid,
        use_xrv_fid=not args.no_xrv_fid, model_name=args.model_name, model_type=args.model_type,
        rrg_model_id=args.rrg_model_id, generated_reports_file=args.generated_reports_file,
        num_diversity_prompts=args.num_diversity_prompts, n_repetitions=args.n_repetitions,
        eval_split=args.eval_split, gen_size=args.gen_size,
    )


if __name__ == "__main__":
    main()
