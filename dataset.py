"""
RoentGen MIMIC-CXR dataset helpers.

The original project targets Stable Diffusion fine-tuning on PA chest X-rays.
This module keeps that data flow intact while adding a few lightweight knobs
for concept/demo runs:
  - smaller image sizes
  - optional dataset subsampling
  - one-time caption tokenization instead of tokenizing every sample access
"""

import argparse
import ast
import logging
import platform
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MAX_TOKEN_LENGTH = 77
MIN_IMPRESSION_CHARS = 7
TOKENIZER_ID = "openai/clip-vit-large-patch14"
_DEFAULT_WORKERS = 0 if platform.system() == "Windows" else 4


def _safe_eval_list(value) -> list:
    """Parse Python-list-like CSV cells safely."""
    if pd.isna(value) or str(value).strip() == "[]":
        return []
    try:
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []


def extract_impression(text: str) -> Optional[str]:
    """Extract the Impression section from a report string."""
    match = re.search(r"Impression:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    impression = match.group(1).strip()
    return impression or None


def _extract_study_id(image_path: str) -> Optional[str]:
    """Extract study id such as s50084553 from an image path."""
    for part in image_path.replace("\\", "/").split("/"):
        if part.startswith("s") and part[1:].isdigit():
            return part
    return None


def build_metadata(
    data_root: str,
    csv_path: Optional[str] = None,
    use_augmented: bool = False,
    max_samples: Optional[int] = None,
    sample_seed: int = 42,
) -> pd.DataFrame:
    """
    Build image_path/impression pairs from the preprocessed CSV file.

    Args:
        data_root: base directory that contains the CSV and image tree.
        csv_path: optional explicit CSV path.
        use_augmented: include rows from text_augment when available.
        max_samples: optional random subset size for lightweight runs.
        sample_seed: random seed used with max_samples.
    """
    data_root_path = Path(data_root)
    csv_file = Path(csv_path) if csv_path else data_root_path / "mimic_cxr_aug_train.csv"
    cache_dir = data_root_path / ".cache"
    cache_file = cache_dir / f"{csv_file.stem}_pa_metadata_aug{int(use_augmented)}.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_file}")

    if cache_file.exists():
        logger.info("Loading cached metadata: %s", cache_file)
        metadata = pd.read_csv(cache_file)
    else:
        logger.info("Loading CSV: %s", csv_file)
        df = pd.read_csv(csv_file)
        logger.info("CSV loaded: %s rows", len(df))

        image_root = data_root_path / "official_data_iccv_final"
        rows = []

        for _, row in df.iterrows():
            pa_images = _safe_eval_list(row.get("PA", "[]"))
            if not pa_images:
                continue

            all_images = _safe_eval_list(row.get("image", "[]"))
            seen = set()
            unique_studies = []
            for image_path in all_images:
                study_id = _extract_study_id(image_path)
                if study_id and study_id not in seen:
                    seen.add(study_id)
                    unique_studies.append(study_id)

            texts = _safe_eval_list(row.get("text", "[]"))
            texts_aug = _safe_eval_list(row.get("text_augment", "[]")) if use_augmented else []

            study_text = {}
            study_text_aug = {}
            for index, study_id in enumerate(unique_studies):
                if index < len(texts):
                    study_text[study_id] = texts[index]
                if use_augmented and index < len(texts_aug):
                    study_text_aug[study_id] = texts_aug[index]

            for pa_image in pa_images:
                study_id = _extract_study_id(pa_image)
                if study_id is None or study_id not in study_text:
                    continue

                impression = extract_impression(study_text[study_id])
                if impression is None or len(impression) < MIN_IMPRESSION_CHARS:
                    continue

                full_image_path = image_root / pa_image
                if not full_image_path.exists():
                    continue

                rows.append(
                    {
                        "image_path": str(full_image_path),
                        "impression": impression,
                    }
                )

                if use_augmented and study_id in study_text_aug:
                    augmented_impression = extract_impression(study_text_aug[study_id])
                    if augmented_impression and len(augmented_impression) >= MIN_IMPRESSION_CHARS:
                        rows.append(
                            {
                                "image_path": str(full_image_path),
                                "impression": augmented_impression,
                            }
                        )

        metadata = pd.DataFrame(rows)
        logger.info("PA image/impression matches: %s", len(metadata))

        if metadata.empty:
            raise RuntimeError("No valid PA image/impression pairs were found.")

        tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_ID)
        tokenized = tokenizer(
            metadata["impression"].tolist(),
            truncation=False,
            add_special_tokens=True,
        )["input_ids"]
        metadata["_n_tokens"] = [len(token_ids) for token_ids in tokenized]
        before = len(metadata)
        metadata = metadata[metadata["_n_tokens"] <= MAX_TOKEN_LENGTH].drop(columns=["_n_tokens"])
        logger.info("Dropped %s rows longer than %s tokens", before - len(metadata), MAX_TOKEN_LENGTH)

        metadata = metadata[["image_path", "impression"]].reset_index(drop=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(cache_file, index=False)
        logger.info("Metadata cached at: %s", cache_file)

    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be a positive integer.")
        if len(metadata) > max_samples:
            metadata = metadata.sample(n=max_samples, random_state=sample_seed).reset_index(drop=True)
            logger.info("Using lightweight subset: %s samples", len(metadata))

    logger.info("Final training set size: %s", len(metadata))
    return metadata


class MIMICCXRDataset(Dataset):
    """Return SD-ready image tensors and CLIP token ids."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer_name: str = TOKENIZER_ID,
        image_size: int = 512,
    ):
        if image_size <= 0:
            raise ValueError("image_size must be a positive integer.")

        self.df = dataframe.reset_index(drop=True)
        self.image_size = image_size
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
        self.input_ids = self.tokenizer(
            self.df["impression"].tolist(),
            padding="max_length",
            max_length=MAX_TOKEN_LENGTH,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(row["image_path"]) as image:
            pixel_values = self.image_transform(image.convert("RGB"))

        return {
            "pixel_values": pixel_values,
            "input_ids": self.input_ids[idx],
        }


def create_dataloader(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = _DEFAULT_WORKERS,
    csv_path: Optional[str] = None,
    use_augmented: bool = False,
    image_size: int = 512,
    max_samples: Optional[int] = None,
    sample_seed: int = 42,
) -> DataLoader:
    """Build the end-to-end DataLoader used by train.py."""
    dataframe = build_metadata(
        data_root=data_root,
        csv_path=csv_path,
        use_augmented=use_augmented,
        max_samples=max_samples,
        sample_seed=sample_seed,
    )
    dataset = MIMICCXRDataset(dataframe, image_size=image_size)

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": len(dataset) >= batch_size and batch_size > 1,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(**loader_kwargs)
    logger.info(
        "DataLoader ready: batch_size=%s, image_size=%s, batches=%s",
        batch_size,
        image_size,
        len(loader),
    )
    return loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIMIC-CXR DataLoader smoke test")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--use_augmented", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=_DEFAULT_WORKERS)
    args = parser.parse_args()

    loader = create_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        csv_path=args.csv_path,
        use_augmented=args.use_augmented,
        image_size=args.image_size,
        max_samples=args.max_samples,
    )

    batch = next(iter(loader))
    print(f"pixel_values shape: {batch['pixel_values'].shape}")
    print(f"pixel_values range: [{batch['pixel_values'].min():.2f}, {batch['pixel_values'].max():.2f}]")
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"input_ids sample: {batch['input_ids'][0][:10]}")
