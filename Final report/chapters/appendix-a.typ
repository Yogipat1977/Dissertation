#show raw.where(block: true): set text(size: 9pt)
= Project Administration Forms

== Initial Project Proposal
#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  image("../Figures/Proposal_page-0001.jpg", width: 100%),
  image("../Figures/Proposal_page-0002.jpg", width: 100%)
)

== Final Project Proposal
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  image("../Figures/Final_Proposal_page-0001.jpg", width: 100%),
  image("../Figures/Final_Proposal_page-0002.jpg", width: 100%),
  image("../Figures/Final_Proposal_page-0003.jpg", width: 100%)
)

== Internal Ethical Approval Process
#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  image("../Figures/Ethics_page-00.jpg", width: 100%),
  image("../Figures/Ethics_page-01.jpg", width: 100%),
  image("../Figures/Ethics_page-02.jpg", width: 100%),
  image("../Figures/Ethics_page-03.jpg", width: 100%),
  image("../Figures/Ethics_page-04.jpg", width: 100%)
)

= Full Training Configuration
This section details the complete configuration YAML used for training the SegResNet model on the BraTS 2023 dataset, ensuring reproducibility of the hyperparameters, optimiser settings, and loss functions.

#strong[Listing:] Full Training Configuration (`full_training_segresnet.yaml`)

```yaml
project:
  name: "BraTS-Dissertation-Full-SegResNet"
  seed: 42

data:
  data_dir: "data/BraTS2023-Training"       # full training dataset
  train_split: 1000
  val_split: 125                             # remaining ~125 go to test
  roi_size: [160, 160, 160]                  # 160³ - larger spatial context
  batch_size: 1                              # ~38 GB on single GPU with AMP
  num_workers: 8                             # keeps data pipeline saturated
  num_samples: 4                             # RandCropByPosNegLabeld samples

model:
  architecture: "SegResNet"
  spatial_dims: 3
  in_channels: 4
  out_channels: 3
  init_filters: 32
  dropout_prob: 0.1

training:
  epochs: 35                                 # ~50 epochs typical for full BraTS dataset
  learning_rate: 0.00005                      
  weight_decay: 0.00001
  scheduler: "cosine"
  loss: "DiceFocalLoss"
  loss_params:
    sigmoid: true
    gamma: 2.0                               # Focal factor for hard examples (ET)
```

= Technical "Crown Jewels" Implementation
This section highlights the most critical technical contributions of the framework, focusing on the core XAI logic, novel evaluation metrics, and the immersive visualisation bridge.

== 3D Grad-CAM (Core XAI Logic)
This snippet demonstrates the implementation of 3D Grad-CAM, specifically focusing on the gradient-weighted feature map combination.

```python
def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
    # 1. Forward pass (activations captured by hook)
    output = self.model(input_tensor)
    
    # 2. Compute scalar score (spatially average target class logits)
    score = output[0, target_class].mean()

    # 3. Backward pass (gradients captured by hook)
    self.model.zero_grad()
    score.backward()

    # 4. Global-average-pool gradients to get importance weights (alpha)
    alpha = self._gradients.mean(dim=[2, 3, 4], keepdim=True)

    # 5. Weighted combination of feature maps
    cam = (alpha * self._activations).sum(dim=1, keepdim=True)

    # 6. ReLU and Normalisation to [0, 1]
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # 7. Trilinear Upsampling to input resolution (160^3)
    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="trilinear")
    return cam.squeeze().cpu().numpy()
```

== Weighted Dice (Novel Soft-Membership Metric)
The following code implements the Weighted Dice score, which allows for fair evaluation of coarse saliency maps by treating them as soft membership volumes.

```python
def weighted_dice(saliency: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Formula: 2 * Σ(S · G) / (Σ S + Σ G)
    where S ∈ [0,1] is normalised saliency and G ∈ {0,1} is the GT.
    """
    numerator = 2.0 * (saliency * ground_truth).sum()
    denominator = saliency.sum() + ground_truth.sum()
    if denominator < 1e-8:
        return 0.0
    return float(numerator / denominator)
```

== Uncertainty Quantification (MC Dropout)
This snippet shows the stochastic inference loop used to estimate predictive variance as a proxy for model uncertainty.

```python
def generate_uncertainty(self, input_tensor: torch.Tensor, num_samples: int = 20):
    self._enable_dropout()  # Keep dropout active at inference time
    preds = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Stochastic forward pass
            pred = torch.sigmoid(self.model(input_tensor))
            preds.append(pred)
    
    # [num_samples, C, D, H, W]
    all_preds = torch.stack(preds)
    
    # Calculate per-voxel variance across samples
    uncertainty = all_preds.var(dim=0)
    return uncertainty.squeeze().cpu().numpy()
```

== VR Pipeline: NIfTI Volume Reconstruction
To enable clinicians to view model outputs identically to standard ground truth in 3D Slicer, this logic maps the multi-channel predictions back into clinical BraTS labels.

```python
def convert_multichannel_to_discrete(pred_tensor: torch.Tensor) -> np.ndarray:
    """
    Inverse Logic for VR rendering:
    If ET==1 -> label 3 (Enhancing Tumour)
    Else If TC==1 -> label 1 (Necrotic Core)
    Else If WT==1 -> label 2 (Edema)
    """
    wt = pred_tensor[0] > 0.5
    tc = pred_tensor[1] > 0.5
    et = pred_tensor[2] > 0.5
    
    discrete_mask = torch.zeros_like(wt, dtype=torch.uint8)
    discrete_mask[wt] = 2  # Edema
    discrete_mask[tc] = 1  # Necrotic Core
    discrete_mask[et] = 3  # Enhancing Tumour
    
    return discrete_mask.cpu().numpy()
```
