# On the Faithfulness of Vision Transformer Explanations

## 1. Introduction

### Vision Transformers
Vision Transformers (ViTs) were introduced by Dosovitskiy et al. in 2021 in the paper *"An Image is Worth 16x16 Words"*. They are different from Convolutional Neural Networks (CNNs) because they use self-attention mechanisms, which were first developed for natural language processing. ViTs break an image into small parts (patches) and process these parts as sequences, similar to how text is processed in language models. 

ViTs are powerful and achieve excellent results in tasks like image classification, object detection, and segmentation. However, they are like "black boxes" because it is hard to explain why they make specific decisions. This is a problem for applications like healthcare or self-driving cars, where people need to trust the model's decisions.

### Why Explanation is Important
Post-hoc explanation techniques are used to understand how ViTs make decisions. One common method is **salience maps**, which assign importance scores to pixels to show which parts of an image influence the model's prediction. 

But there is a problem: these salience maps are often not **faithful**. This means they do not accurately show the true importance of each pixel. Faithfulness is very important because:
- **Trust**: If the explanations are wrong, people cannot trust the model.
- **Debugging**: Faithful explanations help fix issues in the model.
- **Accountability**: Faithful explanations are needed in areas like healthcare or law to justify decisions.

### The Paper’s Goal
This paper, *"On the Faithfulness of Vision Transformer Explanations"*, focuses on the problem of faithfulness in salience maps. It introduces a new metric called **Salience-guided Faithfulness Coefficient (SaCo)**, which measures how well salience maps align with the actual behavior of the model. 

### Research Questions
The paper explores:
1. Are current metrics for evaluating salience maps good enough to measure faithfulness in ViTs?
2. How can we define and measure faithfulness in a clear and reliable way?
3. What changes can make current explanation methods more faithful?

### Project Goal
In this project, we will analyze the methods and results in the paper. We will focus on understanding the SaCo metric, comparing it with existing metrics, and exploring how well it works in practice.

---

## 1.1 Paper Summary

### Main Contributions of the Paper
1. **A New Metric (SaCo)**:
   - SaCo measures how well salience scores represent the true influence of pixels.
   - It avoids common problems with older metrics like AUC and AOPC by comparing individual pixel groups instead of cumulative effects.

2. **Experimental Findings**:
   - The paper tests SaCo on popular ViT models (ViT-B, ViT-L, and DeiT-B) and datasets (CIFAR-10, CIFAR-100, and ImageNet).
   - Results show that:
     - Current metrics often fail to differentiate between good explanations and random noise.
     - Explanation methods that combine attention information with gradients and multi-layer analysis perform better.

3. **Theoretical Basis**:
   - SaCo is based on comparing pixel subsets using statistical techniques like the Kendall τ correlation.
   - It ensures results are reliable regardless of how salience scores are scaled or ordered.

### Key Observations
- Many current explanation methods do not meet the faithfulness standard.
- Random explanations often score as well as real explanation methods under older metrics, which is a problem.
- Attention-based explanation methods can be improved by combining gradient information and analyzing multiple layers.

---
### 1.2. Related Work

#### Post-hoc Explanation Methods
1. **Gradient-based Methods**: 
   - *Integrated Gradients* (Sundararajan et al., 2017) and *Grad-CAM* (Selvaraju et al., 2017) use gradients to assign salience scores.
   - **Limitations**: Dependence on local gradients, which may not reflect the model's global reasoning.
2. **Attribution-based Methods**:
   - *Layer-wise Relevance Propagation* (Binder et al., 2016) propagates relevance scores through layers.
3. **Attention-based Methods**:
   - Methods like *Raw Attention*, *Rollout*, and *ATTCAT* utilize attention weights as explanations.

#### Faithfulness Evaluation Metrics
- **AUC**: Measures model performance under cumulative pixel removal.
- **AOPC**: Quantifies output variations after perturbation.
- **Comprehensiveness**: Measures importance of pixels with lower salience.
- **Limitations**: These metrics do not evaluate the magnitude differences between salience subsets, failing to validate faithfulness.

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




