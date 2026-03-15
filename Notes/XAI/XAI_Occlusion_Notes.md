# Occlusion Sensitivity for 3D CNNs

## 1. What is Occlusion Sensitivity?
Occlusion Sensitivity (also known as Sliding Window Occlusion or Perturbation-based Saliency) is a model-agnostic Explainable AI technique. Unlike Grad-CAM or LRP which compute relevance through backpropagation or gradients, Occlusion Sensitivity directly tests how the model behaves when specific parts of the input are hidden (occluded).

By systematically passing a "blank" or "noisy" 3D patch (the occluder) over the entire input volume and recording how the target class prediction changes, we build a spatial map. 

If occluding a specific brain region causes the tumor probability mask to drastically drop, that region is highly salient.

### Pros:
- **Model Agnostic:** Requires no knowledge of the network architecture (no hooking ReLUs, no gradients).
- **Extremely Direct:** It physically tests the model's reliance on specific structures by removing them.
- **Easy to Interpret:** Drop in confidence = Importance.

### Cons:
- **Computationally Expensive:** A single 3D MRI scan evaluated with a stride of 8 voxels might require hundreds or thousands of forward passes.

## 2. Mathematical Definition

Let $x$ be the original 3D input volume and $f_c(x)$ be the scalar prediction score (spatial mean) for target class $c$.

Let $O_{i,j,k}$ be an occluding mask of size $(w, h, d)$ placed at spatial center $(i,j,k)$. When applied to the input, it replaces the values inside the mask with a baseline value $b$ (typically $0$ or the global mean). Let the occluded image be $x_{occluded}$.

The **Occlusion Sensitivity** $S_c$ at location $(i,j,k)$ is the drop in prediction score:

$$ S_c(i,j,k) = f_c(x) - f_c(x_{occluded}) $$

- $S_c > 0$: The occluded region was important (score dropped).
- $S_c < 0$: The occluded region was suppressing the tumor prediction.
- $S_c \approx 0$: The region is irrelevant.

To convert this drop into a standard positive relevance heatmap, we apply ReLU and normalize:

$$ Heatmap_c = \frac{ReLU(S_c)}{\max(ReLU(S_c))} $$

## 3. Most Salient Region (MSR) Accuracy

In perturbation-based techniques, researchers often use validation metrics that test the method's effectiveness. While Pointing Game tests if the *single highest peak* is inside the tumor, **Most Salient Region (MSR) Accuracy** measures whether the *most important local patch* aligns with the true anomaly.

Given a ground truth mask $G$ and an occlusion saliency map $S$:

1. Find the 3D position $(i_{max}, j_{max}, k_{max})$ that yielded the highest occlusion sensitivity drop.
2. Check if the center of that occluding patch falls inside the Ground Truth tumor $G$.

$$ MSR_{Accuracy} = \begin{cases} 1 & \text{if } G(i_{max}, j_{max}, k_{max}) == 1 \\ 0 & \text{otherwise} \end{cases} $$

MSR Accuracy is essentially the Pointing Game, but applied specifically to the block-wise nature of Occlusion Sensitivity. Since Saliency $S$ is generated at a lower resolution (due to the sliding window stride), MSR provides a fairer assessment of localization than raw Pointing Game.

## 4. Implementation Details

We will implement a 3D sliding window using a `window_size` (e.g., 16x16x16) and a `stride` (e.g., 8).

1.  **Baseline Score:** Compute the model's prediction score $f_c$ on the unaltered input.
2.  **Grid Generation:** Generate a 3D grid of center points based on the stride over the 160x160x160 input.
3.  **Forward Passes:** For each point, zero out the 16³ block in the input MRI modalities, run the model, and calculate the new score.
4.  **Reconstruction:** Assign the score drop $f_c - f_{occluded}$ to the center voxel in the output heatmap map.
5.  **Upsampling:** Because the stride is 8, the resulting heatmap is 20x20x20. We will upsample this back to 160x160x160 using trilinear interpolation to overlay on the original MRI.
