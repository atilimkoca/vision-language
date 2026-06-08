"""
pathology_reward.py — PRG katkısının çekirdeği
================================================
Pathology Reward-Guided fine-tuning (PRG) için iki parça:

1. CheXpert etiket yükleyici
   MIMIC `mimic-cxr-2.0.0-chexpert.csv` → study_id başına hedef patoloji vektörü.
   (Belirsiz/-1 = pozitif kabul; boş = negatif — makaleyle aynı.)

2. PathologyReward
   Üretilen görüntüyü (x̂₀ → VAE decode → [0,1] RGB) donuk gerçek bir CXR
   sınıflandırıcısına (TorchXRayVision DenseNet-121) sokar ve sınıflandırıcının
   tahminini hedef patolojilere doğru iten DİFERANSİYELLENEBİLİR bir BCE kaybı
   döndürür. Gradyan: classifier → VAE decoder → UNet (LoRA).

Novelty notu: kayıp "bulgu-başına" hesaplanır (her co-occurring bulgu ayrı
ödüllenir) → model "en belirgin bulguyu gösterip gerisini atlama" hilesini
(makalenin belgelediği hata) yapamaz.

ÖNEMLİ varsayımlar (gerçek makinede doğrula):
  • XRV DenseNet çıktısı [0,1] olasılık (xrv pretrained modelleri sigmoid uygular).
  • CheXpert csv kolonları: subject_id, study_id, sonra 14 etiket adı.
  • Görüntü yolundan study_id 's50084553' formatında çıkar (dataset._extract_study_id).
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger(__name__)

# Hedef olarak kullanılan CheXpert etiketleri → XRV pathology adı eşlemesi.
# (XRV'de bulunmayan No Finding / Pleural Other / Support Devices dışarıda.)
CHEXPERT_TO_XRV = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
    "Fracture": "Fracture",
    "Lung Lesion": "Lung Lesion",
    "Lung Opacity": "Lung Opacity",
    "Pleural Effusion": "Effusion",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
}
CHEXPERT_LABELS = list(CHEXPERT_TO_XRV.keys())   # hedef vektör sırası


# ─────────────────────────────────────────────────────────────────────────────
# 1) CheXpert etiket yükleyici
# ─────────────────────────────────────────────────────────────────────────────
def load_chexpert_labels(csv_path: str, uncertain_as: float = 1.0) -> Dict[int, np.ndarray]:
    """
    mimic-cxr-2.0.0-chexpert.csv → {study_id(int): hedef vektör (CHEXPERT_LABELS sırasında)}.

    Args:
        csv_path: CheXpert etiket dosyası yolu.
        uncertain_as: -1 (belirsiz) etiketleri ne kabul edelim (paper: 1.0 = pozitif).
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "study_id" not in df.columns:
        raise ValueError(f"CheXpert csv'de 'study_id' kolonu yok. Kolonlar: {list(df.columns)}")

    labels: Dict[int, np.ndarray] = {}
    for _, row in df.iterrows():
        try:
            sid = int(row["study_id"])
        except (ValueError, TypeError):
            continue
        vec = np.zeros(len(CHEXPERT_LABELS), dtype=np.float32)
        for i, lbl in enumerate(CHEXPERT_LABELS):
            v = row.get(lbl, 0.0)
            if pd.isna(v):
                v = 0.0
            v = float(v)
            if v == -1.0:
                v = uncertain_as
            vec[i] = 1.0 if v == 1.0 else 0.0
        labels[sid] = vec

    logger.info("CheXpert etiketleri yüklendi: %d study, %d patoloji", len(labels), len(CHEXPERT_LABELS))
    return labels


def study_id_to_int(study_id_str: str) -> Optional[int]:
    """'s50084553' → 50084553."""
    s = str(study_id_str).lstrip("s")
    return int(s) if s.isdigit() else None


def labels_for_image_paths(image_paths, chexpert_labels: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Görüntü yolları listesi → (N, L) hedef etiket matrisi.
    study_id eşleşmeyen satırlar sıfır vektör alır (ve uyarı sayılır).
    """
    from dataset import _extract_study_id

    out = np.zeros((len(image_paths), len(CHEXPERT_LABELS)), dtype=np.float32)
    missing = 0
    for i, p in enumerate(image_paths):
        sid_str = _extract_study_id(str(p))
        sid = study_id_to_int(sid_str) if sid_str else None
        if sid is not None and sid in chexpert_labels:
            out[i] = chexpert_labels[sid]
        else:
            missing += 1
    if missing:
        logger.warning("%d/%d görüntü için CheXpert etiketi bulunamadı (sıfır vektör).",
                       missing, len(image_paths))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2) Patoloji ödülü (donuk XRV sınıflandırıcı)
# ─────────────────────────────────────────────────────────────────────────────
class PathologyReward:
    """Üretilen görüntü → XRV sınıflandırıcı → hedef patolojilere BCE ödülü.

    Kayıp (loss) düşükken görüntü prompttaki patolojileri içeriyor demektir.
    Eğitimde toplam kayba `lambda_reward * reward_loss` olarak eklenir.
    """

    def __init__(self, device: torch.device, weights: str = "densenet121-res224-all"):
        import torchxrayvision as xrv

        self.device = device
        self.model = xrv.models.DenseNet(weights=weights).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # CHEXPERT_LABELS → XRV çıktı indeksleri eşlemesi
        xrv_paths = list(self.model.pathologies)
        idx, valid = [], []
        for lbl in CHEXPERT_LABELS:
            xrv_name = CHEXPERT_TO_XRV[lbl]
            if xrv_name in xrv_paths:
                idx.append(xrv_paths.index(xrv_name))
                valid.append(1.0)
            else:
                idx.append(0)      # placeholder; valid=0 ile maskelenir
                valid.append(0.0)
        self.idx = torch.tensor(idx, dtype=torch.long, device=device)
        self.valid = torch.tensor(valid, dtype=torch.float32, device=device)  # (L,)
        logger.info("XRV ödül modeli hazır (%d/%d patoloji eşlendi)",
                    int(self.valid.sum().item()), len(CHEXPERT_LABELS))

    def _preprocess(self, decoded01: torch.Tensor) -> torch.Tensor:
        """[0,1] RGB (B,3,H,W) → XRV girişi: grayscale, 224, [-1024,1024]."""
        g = transforms.functional.rgb_to_grayscale(decoded01, num_output_channels=1)
        g = F.interpolate(g, size=(224, 224), mode="bilinear", align_corners=False)
        # XRV normalizasyonu: [0,1] → [-1024, 1024]. XRV modeli fp32 → fp32'ye çek
        # (gradyan yine decoded01'e akar, autocast bf16 ile uyumlu).
        g = (2048.0 * g - 1024.0).float()
        return g

    def reward_loss(
        self,
        decoded01: torch.Tensor,
        target_labels: torch.Tensor,
        positive_weight: float = 2.0,
    ) -> torch.Tensor:
        """
        Args:
            decoded01:     (B,3,H,W) [0,1], DİFERANSİYELLENEBİLİR (x̂₀'dan VAE decode).
            target_labels: (B, L) hedef CheXpert vektörü (CHEXPERT_LABELS sırasında).
            positive_weight: pozitif (mevcut) bulgulara verilen ağırlık. Sadakat
                             odağı: prompttaki bulguların GÖRÜNMESİ daha önemli.

        Returns:
            Skaler BCE kaybı (bulgu-başına, geçerli etiketler üzerinden ortalama).
        """
        g = self._preprocess(decoded01)
        preds = self.model(g)                       # (B, n_xrv), probabilities or logits by XRV version
        pred_sel = preds[:, self.idx]               # (B, L)

        target = target_labels.to(pred_sel.dtype)
        pred_min = float(pred_sel.detach().min())
        pred_max = float(pred_sel.detach().max())
        if 0.0 <= pred_min and pred_max <= 1.0:
            bce = F.binary_cross_entropy(
                pred_sel.clamp(1e-6, 1.0 - 1e-6), target, reduction="none"
            )
        else:
            bce = F.binary_cross_entropy_with_logits(pred_sel, target, reduction="none")

        # Pozitif bulgulara daha fazla ağırlık (sadakat odağı)
        weight = torch.where(target > 0.5, positive_weight, 1.0)
        bce = bce * weight * self.valid.unsqueeze(0)   # geçersiz patolojileri maskele

        denom = (weight * self.valid.unsqueeze(0)).sum().clamp(min=1.0)
        return bce.sum() / denom


def latents_to_pixels(noise_pred, noisy_latents, timesteps, noise_scheduler, vae, dtype):
    """
    x̂₀ kestirimi → VAE decode → [0,1] RGB (DİFERANSİYELLENEBİLİR).
    PRG ödülü için ortak yardımcı (diffusion x̂₀ rekonstrüksiyonu).
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(noisy_latents.device)
    alpha_bar = alphas_cumprod[timesteps]
    sqrt_ab = alpha_bar.sqrt().view(-1, 1, 1, 1)
    sqrt_1mab = (1.0 - alpha_bar).sqrt().view(-1, 1, 1, 1)

    x0_pred = (noisy_latents - sqrt_1mab * noise_pred) / sqrt_ab
    x0_pred = x0_pred / vae.config.scaling_factor

    decoded = vae.decode(x0_pred.to(vae.dtype)).sample.to(dtype)   # (B,3,H,W) [-1,1]
    decoded01 = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
    return decoded01
