"""
dataset.py — BraTS data listing, train/val/test splitting, and DataLoader creation.
"""

from pathlib import Path

from monai.data import PersistentDataset, DataLoader

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


def create_data_loaders(cfg: dict) -> dict:
    """Create train/val/test PersistentDatasets and DataLoaders from config.

    Returns:
        dict with keys "train", "val", "test", each containing a DataLoader.
    """
    datalist = get_brats_data_list(cfg["data"]["data_dir"])

    train_split = cfg["data"]["train_split"]
    val_split = cfg["data"]["val_split"]

    train_files = datalist[:train_split]
    val_files = datalist[train_split : train_split + val_split]
    test_files = datalist[train_split + val_split :]

    print(f"  Train : {len(train_files)} patients")
    print(f"  Val   : {len(val_files)} patients")
    print(f"  Test  : {len(test_files)} patients")

    cache_dir = Path(cfg["data"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_transforms = get_train_transforms(cfg)
    val_transforms = get_val_transforms(cfg)

    train_ds = PersistentDataset(
        data=train_files, transform=train_transforms, cache_dir=cache_dir
    )
    val_ds = PersistentDataset(
        data=val_files, transform=val_transforms, cache_dir=cache_dir
    )
    test_ds = PersistentDataset(
        data=test_files, transform=val_transforms, cache_dir=cache_dir
    )

    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    return {
        "train": DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "val": DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        "test": DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }
