## 2. The Method and Our Interpretation

### 2.1. The Original Method

#### Salience-guided Faithfulness Coefficient (SaCo)

The core assumption is that salience scores should correlate with the impact of input features on predictions. Mathematically:

1. **Salience Map**:  
   Given an input image $x$, model prediction $y = \hat{f}(x)$, and explanation method $E$, the salience map $M(x)$ assigns a score to each pixel:
   $$
   M(x) = \{m_{ij} \mid m_{ij} \in \mathbb{R}, \, \forall (i, j) \in x\}
   $$
   where $m_{ij}$ is the salience score of pixel $(i, j)$.

2. **Subset Partitioning**:  
   Pixels are divided into $K$ subsets $G_k$ based on salience ranking:
   \[
   G_k = \{(i, j) \mid (k-1)\frac{HW}{K} \leq \text{rank}(m_{ij}) < k\frac{HW}{K}\}
   \]
   where $HW$ is the total number of pixels.

3. **Perturbation Impact**:  
   Replace pixels in $G_k$ with their mean value and measure the change in model confidence:
   \[
   \Delta_k = p(y | x) - p(y | R_p(x, G_k))
   \]
   where $R_p(x, G_k)$ denotes the perturbed image.

4. **Faithfulness Coefficient**:  
   Compare all pairs of subsets $G_i, G_j$ and accumulate the differences in salience scores:
   \[
   F = \frac{\sum_{i, j} w_{ij} \cdot \text{sign}(\Delta_i - \Delta_j)}{\sum_{i, j} |w_{ij}|}, \quad w_{ij} = s(G_i) - s(G_j)
   \]
   $F \in [-1, 1]$, where positive values indicate faithful alignment.

---

### 2.2. Our Interpretation

#### Clarifications and Extensions
1. **Perturbation Strategy**:  
   The choice of replacing pixels with mean values is based on prior work but may introduce bias. We explored alternatives, such as Gaussian noise or black-out perturbations, to assess their impact on SaCo scores.
2. **Subset Size $K$**:  
   The granularity of pixel subsets influences faithfulness evaluation. Smaller $K$ captures more subtle salience differences but increases computation.
3. **Cross-layer Aggregation**:  
   SaCo's success with attention-based methods highlights the importance of integrating gradient information and multi-layer attention. We hypothesize that these designs capture more global model behavior.

#### Connections to Related Metrics
- **Kendall $\tau$ Statistic**:  
  SaCo’s pairwise comparisons are inspired by the Kendall $\tau$ statistic for rank correlation, ensuring robustness in salience evaluations.
- **Scale Invariance**:  
  Unlike AOPC or AUC, SaCo is unaffected by normalization or scaling of salience scores, ensuring consistent evaluations across methods.
