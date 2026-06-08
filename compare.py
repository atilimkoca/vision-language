"""
compare.py — Üç modeli tek komutla değerlendirip birleşik tablo üretir
=======================================================================
baseline / LoRA / PRG modellerini AYNI metriklerle (FID, in-domain XRV FID,
MS-SSIM, CLIP-sim, opsiyonel Tablo 3) çalıştırır; sonuçları tek CSV + Markdown
tabloya yazar.

Her model "ad:checkpoint_dizini" biçiminde verilir.

Kullanım:
  python compare.py --data_root ./data --num_eval_samples 200 \\
      --model baseline_256:./checkpoints/checkpoint-3000 \\
      --model lora_256:./checkpoints_lora_256/checkpoint-3000 \\
      --model prg_256:./checkpoints_prg/checkpoint-3000 \\
      --output_dir ./eval_results

  # Tablo 3 metin metrikleri için (opsiyonel):
  #   --rrg_model_id <hf-image-to-text-model>
"""

import argparse
import logging
import os

from evaluate import evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Tabloda gösterilecek metrik sütunları (varsa) ve yön göstergesi.
METRIC_COLUMNS = [
    ("FID", "↓"),
    ("FID_XRV", "↓"),
    ("MS-SSIM", "↓"),
    ("BLEU-4", "↑"),
    ("ROUGE-L", "↑"),
    ("BERTScore", "↑"),
    ("F1CheXbert", "↑"),
    ("CLIP_sim", "↑"),
]


def _parse_model_spec(spec: str):
    """'ad:checkpoint_yolu' → (ad, yol). Windows sürücü harfi (C:\\) korunur."""
    name, sep, path = spec.partition(":")
    if not sep or not path:
        raise argparse.ArgumentTypeError(
            f"Geçersiz model spec '{spec}'. Biçim: ad:checkpoint_dizini"
        )
    # 'ad:C:\\...' durumunda partition ilk ':' den böler; yol sürücü harfiyse birleştir.
    if len(path) == 1 and path.isalpha():
        # nadiren: ad yoktu, baştaki tek harf sürücü → kullanıcı hatası say
        raise argparse.ArgumentTypeError(f"Geçersiz model spec '{spec}'.")
    return name, path


def render_markdown(rows: list) -> str:
    """Sonuç dict listesinden Markdown karşılaştırma tablosu üretir."""
    present = [(m, arrow) for m, arrow in METRIC_COLUMNS if any(m in r for r in rows)]
    header = ["Model", "Tip"] + [f"{m} {a}" for m, a in present]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        cells = [str(r.get("model", "?")), str(r.get("model_type", "?"))]
        cells += [str(r.get(m, "—")) for m, _ in present]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Çok-modelli karşılaştırma")
    parser.add_argument("--model", action="append", dest="models", required=True,
                        type=_parse_model_spec, metavar="AD:CHECKPOINT",
                        help="ad:checkpoint_dizini (birden fazla için tekrar edin)")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--num_eval_samples", type=int, default=200)
    parser.add_argument("--output_dir", default="./eval_results")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--no_xrv_fid", action="store_true")
    parser.add_argument("--rrg_model_id", default=None)
    parser.add_argument("--num_diversity_prompts", type=int, default=20)
    parser.add_argument("--n_repetitions", type=int, default=4)
    parser.add_argument("--eval_split", choices=["test", "train", "all"], default="test",
                        help="Evaluation split. Use test=p19 for reported results.")
    parser.add_argument("--gen_size", type=int, default=256,
                        help="Üretim çözünürlüğü = eğitim çözünürlüğü (256 veya 512).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for name, ckpt in args.models:
        logger.info("\n" + "#" * 70 + f"\n# Değerlendiriliyor: {name}  ({ckpt})\n" + "#" * 70)
        try:
            res = evaluate(
                checkpoint_dir=ckpt, data_root=args.data_root,
                num_eval_samples=args.num_eval_samples, output_dir=args.output_dir,
                num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                seed=args.seed, device=args.device, skip_fid=args.skip_fid,
                use_xrv_fid=not args.no_xrv_fid, model_name=name,
                rrg_model_id=args.rrg_model_id,
                num_diversity_prompts=args.num_diversity_prompts, n_repetitions=args.n_repetitions,
                eval_split=args.eval_split, gen_size=args.gen_size,
            )
            rows.append(res)
        except Exception as e:  # noqa: BLE001
            logger.error("'%s' değerlendirilemedi: %s", name, e)
            rows.append({"model": name, "model_type": "HATA", "FID": f"HATA: {e}"})

    table = render_markdown(rows)
    md_path = os.path.join(args.output_dir, "comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Model Karşılaştırması\n\n")
        f.write(f"- Örnek sayısı: {args.num_eval_samples}\n")
        f.write(f"- Split: {args.eval_split}\n")
        f.write(f"- Çıkarım adımı: {args.num_inference_steps} | CFG: {args.guidance_scale}\n\n")
        f.write(table + "\n\n")
        f.write("↓ = düşük daha iyi, ↑ = yüksek daha iyi. "
                "Tablo 3 (BLEU/ROUGE/BERTScore/F1CheXbert) için --rrg_model_id gerekir.\n")

    print("\n" + table + "\n")
    logger.info("Markdown tablo: %s", md_path)
    logger.info("Birleşik CSV: %s", os.path.join(args.output_dir, "metrics_summary.csv"))


if __name__ == "__main__":
    main()
