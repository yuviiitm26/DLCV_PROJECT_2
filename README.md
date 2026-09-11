# Image Colorization Model & Failure Analysis

## Model Implementation

To satisfy the project specifications, we have deployed **DDColor**, an open-source image colorization model published in a peer-reviewed venue (ICCV) in 2023. DDColor utilizes a dual-decoder architecture, leveraging a pixel decoder to maintain spatial resolution and a color decoder that uses multi-scale features to optimize learned color queries.

## Failure Analysis & Taxonomy

As per the project scope, this section catalogs and stress-tests specific failure conditions. While DDColor achieves state-of-the-art results on standard benchmarks, it exhibits distinct vulnerabilities under the following conditions:

*   **Out-of-Domain Inputs:** The model struggles significantly with out-of-domain inputs, such as historical line art, non-photorealistic sketches, or corrupted archival footage. It tends to default to desaturated or sepia tones due to a lack of relevant priors in its training manifold.
*   **Complex Lighting:** When subjected to complex lighting conditions, the architecture fails to accurately separate illumination from reflectance. Extreme luminance intensities are frequently misclassified as intrinsic chrominance, leading to localized color distortion.
*   **Fine Textures:** The network struggles to process fine textures, often generating noticeable chromatic aliasing and Moiré patterns.
*   **Multi-Object Semantic Boundaries:** The model exhibits severe "color bleeding" across multi-object semantic boundaries. This failure mode is triggered when foreground and background occlusion edges share similar low-level contrast distributions.

## Theoretical Failure Mechanisms

From a deep learning perspective, these failures are mathematically tied to the spatial downsampling bottlenecks and the loss formulation of the network. Deep colorization models typically predict the $a$ and $b$ channels of the CIELAB color space from the luminance channel $L$.

DDColor treats colorization as a classification problem over a quantized color space, optimizing an objective similar to cross-entropy drift:

$$L_c = -\sum_{h,w}\sum_{q} Y_{h,w,q} \log(P_{h,w,q})$$

When the model encounters out-of-domain data or complex lighting, the uncertainty flattens the predicted probability distribution $P_{h,w,q}$, pushing the expected value toward the mean (gray/desaturated). Furthermore, the attention mechanism's receptive field can smooth spatial gradients excessively, causing the predicted chrominance $\nabla P_{h,w}$ to diffuse across boundaries where semantic alignment fails.

## Deliveries

This README serves as the complete report describing where the model fails and in what ways.
