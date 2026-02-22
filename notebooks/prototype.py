import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import wandb

# MONAI and Torch imports 
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, 
    RandGaussianNoised, RandFlipd, MapTransform, 
    EnsureTyped, CropForegroundd, SpatialPadd, RandCropByPosNegLabeld, AsDiscrete
)
from monai.data import PersistentDataset, DataLoader
from monai.utils import set_determinism
from monai.networks.nets import SegResNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- 1. SETUP & REPRODUCIBILITY ---
set_determinism(seed=0) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. DATA LIST GENERATION ---
data_dir = Path("/home/yogipatel/Desktop/Dissertation/data/Prototype_Data")

def get_brats_data_list(data_path):
    patient_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    data_list = []
    for p_dir in patient_dirs:
        p_name = p_dir.name
        data_list.append({
            "image": [
                str(p_dir / f"{p_name}-t1n.nii.gz"),
                str(p_dir / f"{p_name}-t1c.nii.gz"),
                str(p_dir / f"{p_name}-t2w.nii.gz"),
                str(p_dir / f"{p_name}-t2f.nii.gz")
            ],
            "label": str(p_dir / f"{p_name}-seg.nii.gz")
        })
    return data_list

datalist = get_brats_data_list(data_dir)

# --- 3. CUSTOM TRANSFORM: MULTI-CHANNEL LABELS ---
class ConvertToMultiChannelBraTS2023d(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # Channel 1: Whole Tumor (WT) - Labels 1, 2, 3
            result.append(torch.logical_or(torch.logical_or(d[key] == 1, d[key] == 2), d[key] == 3))
            # Channel 2: Tumor Core (TC) - Labels 1, 3
            result.append(torch.logical_or(d[key] == 1, d[key] == 3))
            # Channel 3: Enhancing Tumor (ET) - Label 3
            result.append(d[key] == 3)
            d[key] = torch.cat(result, dim=0).float() 
        return d

# --- 4. PREPROCESSING PIPELINES ---
train_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),
    CropForegroundd(keys=["image", "label"], source_key="image"), 
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ConvertToMultiChannelBraTS2023d(keys="label"),
    SpatialPadd(keys=["image", "label"], spatial_size=[160, 160, 160]),
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=[160, 160, 160], pos=1, neg=1, num_samples=1,
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
])

val_test_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ConvertToMultiChannelBraTS2023d(keys="label"),
    SpatialPadd(keys=["image", "label"], spatial_size=[160, 160, 160]),
])

# --- 5. DATASET SPLITTING & LOADERS ---
cache_dir = Path("./persistent_cache")
cache_dir.mkdir(exist_ok=True)

train_files = datalist[:32]
val_files = datalist[32:40]
test_files = datalist[40:]

train_ds = PersistentDataset(data=train_files, transform=train_transform, cache_dir=cache_dir)
val_ds = PersistentDataset(data=val_files, transform=val_test_transform, cache_dir=cache_dir) 
test_ds = PersistentDataset(data=test_files, transform=val_test_transform, cache_dir=cache_dir)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True) 
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# --- 6. W&B INITIALIZATION ---

os.environ["WANDB_MODE"] = "offline"


wandb.init(
    project="BraTS-Dissertation-Prototype",
    config={
        "learning_rate": 1e-4,
        "epochs": 30,
        "upsample_mode": "deconv",
        "blocks_down": [1, 2, 2, 4],
        "blocks_up": [1,1,1],
        "init_filters": 32, # Increased filters for better feature extraction
        "dropout": 0.2,
        "roi_size": (160, 160, 160),
        "model_architecture": "SegResNet"
    }
)
config = wandb.config

# --- 7. MODEL, LOSS, & OPTIMIZER ---
model = SegResNet(
    spatial_dims=3,
    init_filters=config.init_filters,
    upsample_mode=config.upsample_mode,
    blocks_up=config.blocks_up,
    blocks_down=config.blocks_down,
    in_channels=4,
    out_channels=3,
    dropout_prob=config.dropout,
).to(device)

wandb.watch(model, log_freq=100)

loss_function = DiceLoss(smooth_nr=1e-5, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs) # T_max matches epochs for full annealing

# --- 8. GLOBAL SETTINGS ---
post_trans = AsDiscrete(threshold=0.5)
dice_metric = DiceMetric(include_background=True, reduction="mean")
model_save_path = "prototype-32.pth"
best_metric = -1

# --- 9. TRAINING LOOP ---
if os.path.exists(model_save_path):
    print(f"\n[INFO] Found {model_save_path}. Skipping training...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
else:
    total_pbar = tqdm(total=config.epochs, desc="Total Training Progress", position=0)
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        step_pbar = tqdm(train_loader, desc=f"↳ Epoch {epoch+1}", position=1, leave=False)
        
        for batch_data in step_pbar:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        
        # Validation Phase
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                v_in, v_lab = val_data["image"].to(device), val_data["label"].to(device)
                v_out = monai.inferers.sliding_window_inference(v_in, config.roi_size, 4, model)
                v_out = [post_trans(torch.sigmoid(i)) for i in v_out]
                dice_metric(y_pred=v_out, y=v_lab)
            
            curr_dice = dice_metric.aggregate().item()
            dice_metric.reset()
            
            # Log to W&B
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_dice": curr_dice,
                "lr": optimizer.param_groups[0]['lr']
            })

            if curr_dice > best_metric:
                best_metric = curr_dice
                torch.save(model.state_dict(), model_save_path)
                print(f" -> New Best Dice: {best_metric:.4f}")
        
        total_pbar.update(1)
    total_pbar.close()

# --- 10. EVALUATION & COMPARATIVE ANALYSIS ---
def evaluate_set(loader, name):
    model.eval()
    metric = DiceMetric(include_background=True, reduction="mean_batch")
    with torch.no_grad():
        for data in tqdm(loader, desc=f"Evaluating {name}"):
            inputs, labels = data["image"].to(device), data["label"].to(device)
            outputs = monai.inferers.sliding_window_inference(inputs, config.roi_size, 4, model)
            outputs = [post_trans(torch.sigmoid(i)) for i in outputs]
            metric(y_pred=outputs, y=labels)
    results = metric.aggregate()
    metric.reset()
    return [results[0].item(), results[1].item(), results[2].item()]

print("\n--- Final Comparative Analysis ---")
model.load_state_dict(torch.load(model_save_path))

train_res = evaluate_set(train_loader, "Train")
val_res = evaluate_set(val_loader, "Val")
test_res = evaluate_set(test_loader, "Test")

# Create W&B Table
report_table = wandb.Table(columns=["Region", "Train", "Val", "Test"])
regions = ["Whole Tumor", "Tumor Core", "Enhancing"]
for i, reg in enumerate(regions):
    report_table.add_data(reg, train_res[i], val_res[i], test_res[i])
wandb.log({"Evaluation_Summary": report_table})

# Print console table
print("\n" + "="*50)
print(f"{'Region':<15} | {'Train':<8} | {'Val':<8} | {'Test':<8}")
print("-" * 50)
for i, reg in enumerate(regions):
    print(f"{reg:<15} | {train_res[i]:.4f}   | {val_res[i]:.4f}   | {test_res[i]:.4f}")
print("="*50)

wandb.finish()
print("Process Complete.")
