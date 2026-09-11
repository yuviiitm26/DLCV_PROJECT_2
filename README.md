# Image Colorization Model & Failure Analysis

## Model Implementation

To satisfy the project specifications, we have deployed **DDColor**, an open-source image colorization model published in a peer-reviewed venue (ICCV) in 2023. DDColor utilizes a dual-decoder architecture, leveraging a pixel decoder to maintain spatial resolution and a color decoder that uses multi-scale features to optimize learned color queries.

## Empirical Failure Analysis & Taxonomy

The deployed DDColor architecture was evaluated across four challenging visual regimes using a standardized benchmark subject ("Nano Banana"). Below is the systematic failure taxonomy and root-cause analysis:

| Stress Condition | Primary Visual Artifact | Architectural / Theoretical Mechanism |
| :--- | :--- | :--- |
| **1. Complex Lighting** | Specular Chrominance Entanglement | Illuminant-reflectance ambiguity in CIELAB |
| **2. Fine Textures** | Low-Frequency Chromatic Bleed | Downsampling bottleneck in pixel decoder |
| **3. Multi-Object Semantic Boundary** | Global Chromatic Flooding | Spatial attention diffusion & loss of edges |
| **4. Out-of-Domain Inputs** | Mean-Mode Collapse & Patchy Artifacts | Severe distribution shift from ImageNet |

---

### 1. Complex Lighting: Specular Chrominance Entanglement

*   **Visual Evidence:** In the stress-test image below, the high-intensity overhead lamp cast and specular reflections along the tabletop are coated in deep, saturated yellow and amber tones.
*   **Failure Analysis:** In natural image training sets, high luminance ($L \to 100$) strongly correlates with warm incandescent or solar light sources. The model cannot mathematically separate intrinsic surface reflectance (albedo) from extrinsic illumination:

$$\mathbf{I}(x,y) = \mathbf{R}(x,y) \cdot \mathbf{L}(x,y)$$

Because DDColor conditions color prediction solely on the luminance map $L$, extreme luminance values trick the color queries into predicting heavy chromaticity ($a^*, b^*$) onto uncolored metal surfaces.

### 2. Fine Textures: Chromatic Aliasing & Local Bleed

*   **Visual Evidence:** The dense woven fabric background exhibits an artificial split-tone effect (slate-blue drifting into warm earthen tones). The device's color spills over the high-frequency weave boundaries.
*   **Failure Analysis:** The pixel decoder undergoes spatial downsampling to construct multi-scale feature pyramids. In regions of high spatial frequency $\omega_{\text{high}}$, the feature maps lose exact phase and edge alignment. Consequently, the low-frequency chrominance output cannot conform to individual micro-edges, producing chromatic blur.

### 3. Multi-Object Semantic Boundaries: Chromatic Diffusion

*   **Visual Evidence:** The model suffers a catastrophic breakdown. The entire uniform gray background is flooded with an intense red/orange wash, and the color boundaries between the subjects completely dissolve.
*   **Failure Analysis:** The transformer-based color decoder uses cross-attention between learned color queries and visual feature maps:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

When foreground objects and background share near-identical luminance and low edge gradients ($\nabla L \approx 0$), the attention map flattens. Unable to identify semantic occlusion contours, the network's color queries diffuse uniformly across the spatial domain.

### 4. Out-of-Domain Inputs: Manifold Shift & Mode Collapse

*   **Visual Evidence:** The technical engineering schematic is rendered with an artificial sepia-yellow paper wash, accompanied by an arbitrary yellow smear localized around the central blueprint.
*   **Failure Analysis:** Technical line drawings occupy a completely disjoint subspace from natural photographic manifolds. Because the network encounters zero natural texture priors, the softmax distribution over the quantized color bins flattens:

$$\mathcal{H}(P) = -\sum_{q} P(q) \log P(q) \to \text{maximum}$$

To minimize expected cross-entropy loss under high uncertainty, the model collapses toward the dataset's empirical mean mode (aged sepia/parchment tone).
