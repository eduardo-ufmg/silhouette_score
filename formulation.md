This function calculates a custom metric derived from the well-known **Silhouette Score** to evaluate class separation. It first computes the silhouette score for each sample and then uses the mean and standard deviation of these scores to produce a final value.

***

### 1. The Silhouette Score for a Single Point 🧍

The process begins by treating each row of the input data as a point in an $N$-dimensional space. For each individual point, $\mathbf{p}_i$, a score is calculated to measure how well it fits within its assigned class compared to neighboring classes. This involves two key quantities:

* **Mean Intra-Cluster Distance ($a(i)$)**: This is the average Euclidean distance between the point $\mathbf{p}_i$ and all other points belonging to the **same class**. A small value indicates the point is close to the other members of its own class.

* **Mean Nearest-Cluster Distance ($b(i)$)**: This is the average Euclidean distance from the point $\mathbf{p}_i$ to all the points in the **nearest neighboring class**. To find it, the average distance from $\mathbf{p}_i$ to every *other* class is calculated first; $b(i)$ is the minimum of these values.

The **silhouette score** for the single point $\mathbf{p}_i$, denoted $s(i)$, is then calculated with the following formula:

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

This score ranges from +1 (indicating the point is far from neighboring classes and close to its own) down to -1 (indicating the point is closer to a neighboring class than to its own).

***

### 2. Aggregation and Final Custom Metric 📈

After computing the individual score $s(i)$ for every point in the dataset, the results are aggregated to produce a final custom metric.

* **Statistical Measures**: The **mean** ($\mu_s$) and **standard deviation** ($\sigma_s$) of the set of all individual silhouette scores $\{s(1), s(2), \dots, s(M)\}$ are calculated.

* **Penalty Term**: A penalty factor is computed based on a given scalar input, $f_k$. This penalty is defined as:
    
    $$
    \text{penalty} = 1 - (2 f_k (1 - f_k))
    $$
    
   

* **Final Score**: The final output is not the standard mean silhouette score. Instead, it's a custom formula that combines the mean, its deviation from a target of 0.5, the standard deviation, and the penalty factor.
    
    $$
    \text{Final Score} = - ((\mu_s - 0.5)^2 + \sigma_s) \cdot \text{penalty}
    $$
    
   