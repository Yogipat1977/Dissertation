import os
import matplotlib.pyplot as plt
import monai
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    CropForeground,
    RandFlip,
    RandRotate90,
    RandScaleIntensity,
    RandGaussianNoise,
    RandSpatialCrop,
    Compose
)

# Set patient directory
patient_idx = "01666"
data_dir = f"/home/yogipatel/Desktop/Dissertation/data/BraTS2023-Training/BraTS-GLI-{patient_idx}-000"
image_path = os.path.join(data_dir, f"BraTS-GLI-{patient_idx}-000-t2w.nii.gz")

# Step 1: Load and get the basic "Original" image
# We'll apply CropForeground as a baseline so there isn't just massive empty space
base_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    CropForeground()
])

original_img = base_transforms(image_path)
# original_img shape is [C, H, W, D], where C=1
# Get middle Z slice for visualization
mid_z = original_img.shape[3] // 2

# Helper to extract the mid slice
def get_slice(img_volume):
    # Shape is [1, H, W, D]. Extract D/2 slice.
    z = img_volume.shape[3] // 2
    # Rotate 90 degrees strictly for visualization so nose is 'up' typical for BraTS
    return img_volume[0, :, :, z].numpy()

# Note: many random transforms expect channel first, which we have.
# 2. Random Flip (e.g. along left-right axis which is spatial_axis=0)
flip_img = RandFlip(prob=1.0, spatial_axis=0)(original_img)

# 3. 90-Degree Rotation
rot_img = RandRotate90(prob=1.0, max_k=1, spatial_axes=(0, 1))(original_img)

# 4. Intensity Scaling
scale_img = RandScaleIntensity(prob=1.0, factors=0.5)(original_img)

# 5. Gaussian Noise
noise_img = RandGaussianNoise(prob=1.0, mean=0.0, std=0.2)(original_img)

# 6. Spatial Cropping (160x160x160)
# Make sure original isn't already smaller than 160. If it is, this will pad it or we can fallback to smaller.
d_shape = original_img.shape[1:]
roi_sizes = [min(160, d) for d in d_shape]
crop_img = RandSpatialCrop(roi_size=roi_sizes, random_size=False)(original_img)


# Set up Matplotlib figure
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

images_to_plot = [
    ("Original (Cropped FG)", original_img),
    ("Random Flip", flip_img),
    ("90-Degree Rotation", rot_img),
    ("Intensity Scaling", scale_img),
    ("Gaussian Noise", noise_img),
    (f"Spatial Cropping ({roi_sizes[0]}³)", crop_img)
]

for idx, (title, img_vol) in enumerate(images_to_plot):
    # For Spatial Crop, the Z-dimension might change! So recalculate the mid_z inside get_slice
    slice_data = get_slice(img_vol)
    
    # We transpose to make the orientation look like a standard axial view in matplotlib
    slice_data = slice_data.T
    
    axes[idx].imshow(slice_data, cmap="gray")
    axes[idx].set_title(title, fontsize=14, pad=10)
    axes[idx].axis("off")

plt.tight_layout()

# Save the figure
out_dir = "/home/yogipatel/Desktop/Dissertation/Final report/Figures"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"patient_{patient_idx}_augmentations.png")

plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"✅ Successfully saved augmentation figure to {out_path}")
