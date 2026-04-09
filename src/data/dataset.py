"""
dataset.py — BraTS data listing, train/val/test splitting, and DataLoader creation.
"""

import json
import random
from pathlib import Path

from monai.data import PersistentDataset, Dataset, DataLoader

from src.data.transforms import get_train_transforms, get_val_transforms


def get_brats_data_list(data_dir: str) -> list[dict]:
    """Scan a BraTS data directory and return a list of dicts with
    'image' (list of 4 modality paths) and 'label' (seg path) keys."""
    data_path = Path(data_dir)
    patient_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    data_list = []

    for p_dir in patient_dirs:
        name = p_dir.name
        data_list.append({
            "image": [
                str(p_dir / f"{name}-t1n.nii.gz"),
                str(p_dir / f"{name}-t1c.nii.gz"),
                str(p_dir / f"{name}-t2w.nii.gz"),
                str(p_dir / f"{name}-t2f.nii.gz"),
            ],
            "label": str(p_dir / f"{name}-seg.nii.gz"),
        })

    return data_list


def _make_dataset(data, transform, cache_dir):
    """Create a PersistentDataset if cache_dir is set, otherwise a plain Dataset."""
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        return PersistentDataset(data=data, transform=transform, cache_dir=cache_path)
    return Dataset(data=data, transform=transform)


def create_data_loaders(cfg: dict) -> dict:
    """Create train/val/test DataLoaders from config.

    Returns:
        dict with keys "train", "val", "test", each containing a DataLoader.
    """
    datalist = get_brats_data_list(cfg["data"]["data_dir"])

    # Shuffle patients before splitting to prevent distribution bias
    # (ensures train/val/test share similar scanner/hospital distributions)
    seed = cfg["project"].get("seed", 42)
    random.seed(seed)
    random.shuffle(datalist)

    train_split = cfg["data"]["train_split"]
    val_split = cfg["data"]["val_split"]

    train_files = datalist[:train_split]
    val_files = datalist[train_split : train_split + val_split]
    test_files = datalist[train_split + val_split : train_split + val_split + 25]

    print(f"  Train : {len(train_files)} patients")
    print(f"  Val   : {len(val_files)} patients")
    print(f"  Test  : {len(test_files)} patients")

    # Save manifest so we always know which patients went where (once only)
    run_dir = cfg.get("_run_dir")
    if run_dir:
        manifest_path = Path(run_dir) / "manifest.json"
        if not manifest_path.exists():
            manifest = {
                "seed": seed,
                "total_patients": len(datalist),
                "train_count": len(train_files),
                "val_count": len(val_files),
                "test_count": len(test_files),
                "train": [Path(f["label"]).parent.name for f in train_files],
                "val":   [Path(f["label"]).parent.name for f in val_files],
                "test":  [Path(f["label"]).parent.name for f in test_files],
            }
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"  Manifest saved to: {manifest_path}")
        else:
            print(f"  Manifest exists: {manifest_path} (skipped)")

    cache_dir = cfg["data"].get("cache_dir", "")

    train_transforms = get_train_transforms(cfg)
    val_transforms = get_val_transforms(cfg)

    train_ds = _make_dataset(train_files, train_transforms, cache_dir)
    val_ds = _make_dataset(val_files, val_transforms, cache_dir)
    test_ds = _make_dataset(test_files, val_transforms, cache_dir)

    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    return {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "val": DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=num_workers
        ),
        "test": DataLoader(
            test_ds, batch_size=1, shuffle=False, num_workers=num_workers
        ),
    }
