# Layer-wise Relevance Propagation (LRP) for 3D CNNs

## 1. What is LRP?
Layer-wise Relevance Propagation (LRP) is an Explainable AI (XAI) technique that operates on the premise of a "conservation principle". Unlike gradient-based methods (which measure sensitivity to change), LRP attributes a relevance score $R$ to each voxel in the input image such that the sum of the relevance scores equals the final prediction score $f(x)$ output by the model. 

Mathematically:
$$ f(x) = \sum_i R_i $$

The process starts with the output prediction score $f(x)$ and propagates it backwards through the network layers distributing it to the lower-level neurons until it reaches the input layer.

## 2. The $\epsilon$-LRP Rule
For standard CNNs, the most common backward propagation rule is the epsilon-rule ($\epsilon$-LRP). It adds a small stabilizer $\epsilon$ to the denominator to prevent numerical instability (division by zero) when the total activation of a neuron is zero or very small.

For a neuron $j$ in layer $l$, the relevance $R_j$ is distributed to a neuron $i$ in layer $l-1$ proportionally to how much neuron $i$ contributed to the activation of neuron $j$:

$$ R_i = \sum_j \frac{z_{ij}}{\sum_k z_{kj} + \epsilon} R_j $$

where $z_{ij} = a_i w_{ij}$ (activation of neuron $i$ multiplied by the weight connecting it to neuron $j$).

## 3. LRP vs. Grad-CAM vs. Guided Backpropagation

| Feature | Grad-CAM | Guided Backprop | LRP |
|---------|----------|-----------------|-----|
| **Core Idea** | Gradient of output w.r.t intermediate feature maps | Modifies ReLU to restrict negative gradients flowing backward | Distributes output prediction score backward based on weights/activations |
| **Resolution** | Coarse (resolution of the chosen target layer) | Native input resolution (sharp edges) | Native input resolution |
| **Interpretation** | "Where is the model looking generally?" | "Which specific pixels would change the prediction if altered?" | "Which specific pixels contributed most to the final prediction score?" |
| **Output Type** | Heatmap blob | High-frequency edge map | High-frequency relevance map (can be positive or negative) |
| **Theoretical Basis** | Partial derivatives | Modified chain rule | Deep Taylor Decomposition / Conservation Principle |

## 4. Implementation Details in PyTorch

Standard PyTorch doesn't natively support LRP because it requires tracking not just gradients, but the forward-pass activations and the specific subset of weights that contributed positively/negatively at every single layer. 

There are two primary ways to implement it:
1.  **Custom Backward Hooks:** Override the backward pass manually at each layer (similar to Guided Backprop, but much more complex due to needing forward activations saved).
2.  **LRP library / Re-implementation:** Re-implement the forward pass using custom layers or use a library that handles the Taylor decomposition.

Given the complexity of `SegResNet` (which includes skip connections, specialized normalization, etc.), writing a manual rule parser for every block type is highly complex.

### 4.1 A Gradient-Based Approximation (The $\alpha \beta$ Rule equivalent via DeepLIFT/GradientxInput)

A common and mathematically sound alternative that strongly approximates LRP (specifically the LRP-$\epsilon$ or linear LRP rules) on networks using ReLUs is **Input $\times$ Gradient**. 

$$ R_i \approx x_i \cdot \frac{\partial f(x)}{\partial x_i} $$

Under certain conditions in ReLU networks (no biases, or biases handled carefully), **Input $\times$ Gradient is mathematically equivalent to $\epsilon$-LRP**. It yields high-resolution maps that indicate which voxels actually 'drive' the score.

For this framework, we will implement this Input $\times$ Gradient (or "Saliency $\times$ Input") as our LRP proxy.

## 5. Script Design

1.  **`src/xai/lrp.py`**: A class that computes the `gradient * input` for a given target class. 
2.  **`scripts/generate_lrp.py`**:
    -   Similar to `generate_gbp.py`
    -   Loads the model and patients.
    -   Passes data through the LRP class to get the 160³ heatmaps.
    -   Saves the results directly to `slicer_export/XAI/LRP/`.
    -   **Skips XAI metric calculation and CSV generation** as requested, focusing only on Slicer visualisation.
