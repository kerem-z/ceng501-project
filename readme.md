# On the Faithfulness of Vision Transformer Explanations

## 1. Introduction

### 1.1 Background: Vision Transformers and Their Rise
Vision Transformers (ViTs) have emerged as a groundbreaking architecture in deep learning for image classification tasks. Unlike Convolutional Neural Networks (CNNs), which rely on local receptive fields to capture spatial patterns, ViTs leverage a **self-attention mechanism** to model global relationships across the entire input image. This key difference allows ViTs to:
- Capture long-range dependencies between different regions of the image.
- Adapt flexibly to different input scales and structures.
- Achieve superior performance on various vision benchmarks when trained on large datasets.

While ViTs have achieved remarkable success, they are considered "black-box" models due to the complexity of the attention mechanism. Understanding **why** and **how** these models make specific predictions is crucial, particularly for safety-critical applications, such as medical imaging or autonomous driving.

### 1.2 Explainable AI and Vision Transformers
**Explainable AI (XAI)** refers to methods and techniques that make machine learning models more interpretable, ensuring that their decisions are understandable to humans. In the case of ViTs, interpretability is essential for:
- Building trust in model predictions.
- Diagnosing model failures.
- Ensuring fairness and transparency in AI systems.

To achieve this, **post-hoc explanation methods** are employed to generate **salience maps** that highlight the most important regions of an input image, showing which parts of the image influenced the model’s decision the most. These methods can be divided into two primary categories:
1. **Gradient-based methods**: These methods compute salience scores by calculating gradients with respect to the input image. Examples include:
   - Integrated Gradients
   - Grad-CAM
   - SmoothGrad
2. **Attention-based methods**: These methods use attention weights from the ViT model to determine the relative importance of different tokens (parts of the image). Examples include:
   - Raw Attention
   - Rollout

While these techniques help explain model predictions, their **faithfulness**—the degree to which the salience map accurately reflects the model's true decision-making process—remains an open challenge.

### 1.3 Paper Context and Research Problem
The paper that introduces the **Salience-Guided Faithfulness Coefficient (SaCo)** addresses the problem of **faithfulness** in post-hoc explanations for ViTs. Specifically, it focuses on the fact that:
- Current evaluation metrics, such as those based on cumulative perturbation, fail to properly assess the individual contributions of pixel groups with different salience levels.
- These metrics also overlook the absolute values of salience scores, focusing only on their relative rankings, which can lead to misleading interpretations.

**SaCo** offers a new approach by introducing a pair-wise evaluation of pixel groups based on their salience scores and comparing their impact on the model's predictions. The contributions of the paper include:
- Proposing a more accurate and **faithful** method for evaluating post-hoc explanations in ViTs.
- Providing a **robust framework** to assess the true influence of different pixel groups on model predictions, setting a new standard for explainability in ViTs.
- Demonstrating that **existing methods** often fail to distinguish between meaningful explanations and random attributions, highlighting the need for more reliable evaluation techniques.

### 1.4 My Goal
The goal of this project is to:
- **Reproduce the SaCo metric** as described in the paper to verify its reproducibility and reliability.
- **Explore its application** across different datasets and model architectures to test its generalizability.
- **Identify potential improvements** to make ViT explanations more faithful, transparent, and useful for real-world applications.

---

### 1.5 Challenges and Motivation

#### 1.5.1 Faithfulness in Explanations
The central challenge in explainability for ViTs is ensuring that the **salience scores** truly reflect the impact of different parts of the image on the model’s predictions. Existing explanation methods have significant limitations:
- **Cumulative perturbation** techniques fail to isolate the effects of individual salience levels, blending the influences of various pixel groups.
- **Absolute salience values** are ignored, and only relative rankings are considered, which can distort the actual impact of different image regions.

These limitations result in unreliable and unfaithful explanations, leading to potential misinterpretations of the model's decision-making process. This is particularly concerning for real-world applications where understanding the model's behavior is critical.
---

## Visuals and Examples

### 1.6 ViT Architecture Overview
Below is a simplified diagram of the Vision Transformer architecture, showing how the self-attention mechanism processes input image patches to form a global representation.

![ViT Architecture](path/to/vit-architecture-image.png)

### 1.7 Post-hoc Explanation Methods

#### Gradient-based Explanation Example
This heatmap demonstrates how a **Gradient-based method** (e.g., Integrated Gradients) highlights the regions of an image most relevant to the model's decision.

![Gradient-Based Explanation](path/to/gradient-example.png)

#### Attention-based Explanation Example
This image shows the salience map generated by using **attention weights** from a ViT model. The map indicates which image tokens (patches) the model paid attention to the most when making a prediction.

![Attention-Based Explanation](path/to/attention-example.png)

### 1.8 Comparison of Explanation Methods

| Method          | Approach           | Strengths                  | Weaknesses                      |
|-----------------|--------------------|----------------------------|----------------------------------|
| Gradient-based  | Uses gradients     | Precise, detailed          | Computationally expensive       |
| Attention-based | Uses attention     | Intuitive for ViTs         | May not always align with true model behavior|

---

## 2. Challenges and Motivation

### 2.1 Faithfulness in Explanations
The central challenge in explainability for ViTs is ensuring that the **salience scores** truly reflect the impact of different parts of the image on the model’s predictions. Existing explanation methods have significant limitations:
- **Cumulative perturbation** techniques fail to isolate the effects of individual salience levels, blending the influences of various pixel groups.
- **Absolute salience values** are ignored, and only relative rankings are considered, which can distort the actual impact of different image regions.

These limitations result in unreliable and unfaithful explanations, leading to potential misinterpretations of the model's decision-making process. This is particularly concerning for real-world applications where understanding the model's behavior is critical.

## 2. The Method and Our Interpretation

### 2.1. The Original Method

#### Salience-guided Faithfulness Coefficient (SaCo)

SaCo is a metric designed to evaluate whether salience maps accurately represent the influence of input pixels on model predictions. The key steps of SaCo are outlined below:

1. **Salience Map Creation**  
   A salience map $M(x)$ is generated for an input image $x$, assigning each pixel $(i, j)$ a salience score $m_{ij}$:
```math
M(x) = \{ m_{ij} \mid m_{ij} \in \mathbb{R}, \, \forall (i, j) \in x \}.
```

Here, $m_{ij}$ represents the importance of pixel $(i, j)$ to the model’s output.

2. **Pixel Partitioning**  
Pixels are ranked by their salience scores and divided into $K$ subsets $G_k$, each containing pixels with similar scores:
```math
G_k = \{ (i, j) \mid (k-1)\frac{HW}{K} \leq \text{rank}(m_{ij}) < k\frac{HW}{K} \}
```
where $HW$ is the total number of pixels in the image, and $\text{rank}(m_{ij})$ is the position of $m_{ij}$ in the sorted salience scores.

4. **Perturbation Impact**  
   The influence of each subset $G_k$ is measured by perturbing its pixels and calculating the resulting change in the model’s confidence for the predicted class:
```math
\Delta_k = p(y \mid x) - p(y \mid R_p(x, G_k))
```
   Here, $R_p(x, G_k)$ represents the image where all pixels in $G_k$ are replaced by their mean value.

5. **Faithfulness Coefficient**  
   SaCo compares all pairs of subsets $(G_i, G_j)$ to test whether their salience scores align with their actual impact on the model:
```math
F = \frac{\sum_{i, j} w_{ij} \cdot \text{sign}(\Delta_i - \Delta_j)}{\sum_{i, j} |w_{ij}|}
```
   The weight $w_{ij}$ is defined as the difference in total salience scores between the two subsets:
```math
w_{ij} = s(G_i) - s(G_j), \quad \text{where } s(G_i) = \sum_{p \in G_i} M(x)_p
```
   The faithfulness coefficient $F$ ranges from $-1$ (completely unfaithful) to $+1$ (perfectly faithful).

---

### 2.2. Our Interpretation

#### Clarifications and Potential Improvements

1. **Perturbation Strategy**  
   The original paper perturbs pixels by replacing them with their mean value. While simple, this approach may introduce bias. I might suggest alternative strategies
   - **Gaussian Noise**: Replace pixels with noise sampled from a Gaussian distribution.
   - **Blackout Perturbation**: Mask pixels entirely by setting their values to zero.

2. **Subset Size $K$**  
   The choice of $K$ impacts the resolution of the evaluation:
   - Smaller $K$: Coarser evaluation, less computational effort.
   - Larger $K$: Finer evaluation, more computational cost.
   Dynamically determining $K$ based on the complexity of the salience map could balance efficiency and accuracy.

3. **Cross-layer Aggregation**  
   Attention-based explanation methods perform better when salience scores incorporate information from multiple layers. Aggregating attention maps and gradients across layers provides a more comprehensive view of the model’s reasoning process.

#### Connections to Related Metrics

- **Inspired by Kendall $\tau$**  
   SaCo’s pairwise comparisons are similar to the Kendall $\tau$ statistic, which measures rank correlation. This ensures robustness in assessing salience map quality.

- **Scale Invariance**  
   SaCo avoids issues with normalized or scaled salience maps. Unlike older metrics such as AOPC or AUC, it maintains consistency regardless of transformations applied to salience scores.

#### Key Strengths of SaCo

1. **Direct Evaluation**  
   SaCo assesses the true impact of individual pixel subsets, avoiding the biases of cumulative perturbation.

2. **Noise Sensitivity**  
   Mismatches between salience and influence are penalized more strongly for larger salience differences, enhancing its sensitivity to poor explanations.

3. **Broad Applicability**  
   SaCo is effective across datasets and models, making it a versatile tool for evaluating salience-based explanations.

This interpretation highlights areas for improvement while emphasizing the robustness and relevance of SaCo in faithfulness evaluations.




