= Appendices and Supplementary Materials

== Appendix A: Project Administration Forms

=== Initial Project Proposal
#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  image("../Figures/Proposal_page-0001.jpg", width: 100%),
  image("../Figures/Proposal_page-0002.jpg", width: 100%)
)

=== Final Project Proposal
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  image("../Figures/Final_Proposal_page-0001.jpg", width: 100%),
  image("../Figures/Final_Proposal_page-0002.jpg", width: 100%),
  image("../Figures/Final_Proposal_page-0003.jpg", width: 100%)
)

== Appendix B: Model Configuration and Hyperparameters
This section details the complete configuration YAML used for training the SegResNet model on the BraTS 2023 dataset, ensuring reproducibility of the hyperparameters, optimizer settings, and loss functions.

#figure(
  caption: [Full Training Configuration (`full_training_segresnet.yaml`)],
  kind: raw,
  supplement: [Listing],
)[
```yaml
project:
  name: "BraTS-Dissertation-Full-SegResNet"
  seed: 42

data:
  data_dir: "data/BraTS2023-Training"       # full training dataset
  train_split: 1000
  val_split: 125                             # remaining ~125 go to test
  roi_size: [160, 160, 160]                  # 160³ — larger spatial context
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
]

== Appendix C: Volumetric Data Transformations
This snippet demonstrates the data preprocessing pipeline implemented in MONAI, including the custom `ConvertToMultiChannelBraTS2023d` transform that maps raw BraTS labels into the required clinical sub-regions.

#figure(
  caption: [Data Transforms Pipeline (`transforms.py`)],
  kind: raw,
  supplement: [Listing],
)[
```python
import torch
from monai.transforms import MapTransform, Compose, LoadImaged, CropForegroundd

class ConvertToMultiChannelBraTS2023d(MapTransform):
    """
    Groups raw BraTS labels (1, 2, 3) into clinical sub-regions:
        - Channel 0: Whole Tumor (WT) — labels 1, 2, 3
        - Channel 1: Tumor Core (TC) — labels 1, 3
        - Channel 2: Enhancing Tumor (ET) — label 3
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 1, d[key] == 2), d[key] == 3
                )
            )
            result.append(torch.logical_or(d[key] == 1, d[key] == 3))
            result.append(d[key] == 3)
            d[key] = torch.cat(result, dim=0).float()
        return d
```
]

== Appendix D: 3D Explainable AI Implementation (Grad-CAM)
This section provides the core implementation of the 3D Gradient-weighted Class Activation Mapping (Grad-CAM) adapted for volumetric medical imaging. It demonstrates how gradients are hooked from the target convolutional layer to produce a class-discriminative saliency volume.

#figure(
  caption: [3D Grad-CAM Implementation Core (`grad_cam.py`)],
  kind: raw,
  supplement: [Listing],
)[
```python
import torch
import torch.nn.functional as F
import numpy as np

class GradCAM3D:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None

        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)

        # Spatially average the pre-sigmoid logits for the target class channel
        score = output[0, target_class].mean()
        self.model.zero_grad()
        score.backward(retain_graph=False)

        # Importance weights: global-average-pool the gradients
        alpha = self._gradients.mean(dim=[2, 3, 4], keepdim=True)

        # Weighted combination of feature maps & ReLU
        cam = (alpha * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Upsample to input spatial resolution
        spatial_size = input_tensor.shape[2:]
        cam = F.interpolate(cam, size=spatial_size, mode="trilinear", align_corners=False)

        return cam.squeeze().cpu().numpy()
```
]
