import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from torchsummary import summary

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
            result.append(torch.logical_or(torch.logical_or(d[key] == 1, d[key] == 2), d[key] == 3))
            result.append(torch.logical_or(d[key] == 1, d[key] == 3))
            result.append(d[key] == 3)
            d[key] = torch.cat(result, dim=0).float() 
        return d

# --- 4. PREPROCESSING PIPELINE ---
train_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),
    CropForegroundd(keys=["image", "label"], source_key="image"), 
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ConvertToMultiChannelBraTS2023d(keys="label"),
    SpatialPadd(keys=["image", "label"], spatial_size=[160, 160, 160]),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=[160, 160, 160], 
        pos=1, neg=1, num_samples=1,
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
])
# Validation/Testing needs consistency (No random crops, no noise)
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

# Split data into train and validation
train_files = datalist[:32] # Use the first 32 for training
val_files = datalist[32:40]# Use the rest for validation
test_files = datalist[40:]

train_ds = PersistentDataset(data=train_files, transform=train_transform, cache_dir=cache_dir)
val_ds = PersistentDataset(data=val_files, transform=train_transform, cache_dir=cache_dir) 
test_ds = PersistentDataset(data=test_files, transform=val_test_transform, cache_dir=cache_dir)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True) 
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)


# --- 6. MODEL, LOSS, & OPTIMIZER ---
model = SegResNet(
    spatial_dims=3,
    init_filters=32,
    in_channels=4,      
    out_channels=3,     
    dropout_prob=0.2,   
).to(device)

loss_function = DiceLoss(smooth_nr=1e-5, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# --- 7. PROGRESS TRACKING & REUSE LOGIC ---
max_epochs = 20
post_trans = AsDiscrete(threshold=0.5)
dice_metric = DiceMetric(include_background=True, reduction="mean")
model_save_path = "prototype.pth"
best_metric = -1
history = {"loss": [], "dice": []}


model_save_path = "prototype.pth"

# Check if a saved model already exists
if os.path.exists(model_save_path):
    print(f"\n[INFO] Found {model_save_path}. Skipping training and loading weights...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
else:
    print(f"\n[INFO] No saved model found. Starting training for {max_epochs} epochs...")
    total_pbar = tqdm(total=max_epochs, desc="Total Training Progress", position=0)
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        step_pbar = tqdm(train_loader, desc=f"↳ Epoch {epoch+1}", position=1, leave=False)
        
        for batch_data in step_pbar:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            loss = loss_function(model(inputs), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                v_in, v_lab = val_data["image"].to(device), val_data["label"].to(device)
                roi_size = (160, 160, 160)
                # Use sliding window for validation to match testing logic
                v_out = monai.inferers.sliding_window_inference(v_in, roi_size, 4, model)
                v_out = [post_trans(torch.sigmoid(i)) for i in v_out]
                dice_metric(y_pred=v_out, y=v_lab)
            
            curr_dice = dice_metric.aggregate().item()
            dice_metric.reset()
            if curr_dice > best_metric:
                best_metric = curr_dice
                torch.save(model.state_dict(), model_save_path)
        
        total_pbar.update(1)
    total_pbar.close()


# --- 8. FINAL TESTING PHASE ---
print("\n--- Running Final Evaluation on Test Set ---")
model.load_state_dict(torch.load(model_save_path)) # Ensure we have the best version
model.eval()
test_metric = DiceMetric(include_background=True, reduction="mean_batch")

with torch.no_grad():
    for test_data in tqdm(test_loader, desc="Testing"):
        t_in, t_lab = test_data["image"].to(device), test_data["label"].to(device)
        # Use sliding window to evaluate the WHOLE brain volume at once
        t_out = monai.inferers.sliding_window_inference(t_in, (160, 160, 160), 4, model)
        t_out = [post_trans(torch.sigmoid(i)) for i in t_out]
        test_metric(y_pred=t_out, y=t_lab)

    final_results = test_metric.aggregate()
    print(f"\nTest Results (Dice):")
    print(f"Whole Tumor: {final_results[0].item():.4f}")
    print(f"Tumor Core:  {final_results[1].item():.4f}")
    print(f"Enhancing:   {final_results[2].item():.4f}")
    test_metric.reset()



def evaluate_set(loader, name):
    model.eval()
    metric = DiceMetric(include_background=True, reduction="mean_batch")
    with torch.no_grad():
        for data in tqdm(loader, desc=f"Evaluating {name}"):
            inputs, labels = data["image"].to(device), data["label"].to(device)
            # Use sliding window for fair comparison across all sets
            outputs = monai.inferers.sliding_window_inference(inputs, (160, 160, 160), 4, model)
            outputs = [post_trans(torch.sigmoid(i)) for i in outputs]
            metric(y_pred=outputs, y=labels)
    results = metric.aggregate()
    metric.reset()
    return [results[0].item(), results[1].item(), results[2].item()]

# Execute evaluations
print("\n--- Final Comparative Analysis ---")
train_results = evaluate_set(train_loader, "Train")
val_results = evaluate_set(val_loader, "Val")
test_results = [final_results[0].item(), final_results[1].item(), final_results[2].item()] # From your previous run

# --- 2. DISPLAY RESULTS TABLE ---
print("\n" + "="*50)
print(f"{'Region':<15} | {'Train':<8} | {'Val':<8} | {'Test':<8}")
print("-" * 50)
regions = ["Whole Tumor", "Tumor Core", "Enhancing"]
for i, region in enumerate(regions):
    print(f"{region:<15} | {train_results[i]:.4f}   | {val_results[i]:.4f}   | {test_results[i]:.4f}")
print("="*50)
