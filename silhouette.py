import numpy as np
from scipy.spatial.distance import pdist, squareform

def silhouette(Q: np.ndarray, y: np.ndarray, factor_h: float, factor_k: int) -> float:
    """
    Calculates the Silhouette Score of all samples.

    The Silhouette Score is a measure of how similar a sample is to its own
    class compared to other classes. The score is bounded between -1 for
    incorrectly clustered samples and +1 for highly dense, well-separated classes.
    Scores around zero indicate overlapping classes.

    Each row of the input matrix Q is treated as a point in an n-classes
    dimensional space, and the distances are the Euclidean distances between
    these points.

    Parameters:
        Q (np.ndarray): A 2D numpy array of shape (n_samples, n_classes). Q[i, j]
                        is the similarity of sample `i` to class `j`.
        y (np.ndarray): A 1D numpy array of shape (n_samples,) containing the true
                        class labels for each sample.
        factor_h (float): A scaled factor from the RBF kernel bandwidth parameter.
                          This is used to adjust the Silhouette Score.
        factor_k (int): A scaled factor from the number of nearest neighbors used in
                        the sparse RBF kernel. This is used to adjust the Silhouette Score.

    Raises:
        TypeError: If Q or y cannot be converted to numpy arrays.
        ValueError: If Q is not a 2D array, y is not a 1D array, or if the number of samples in Q and y do not match.
        ValueError: If the number of unique classes in y is less than 2.
        MemoryError: If the pairwise distance matrix is too large to fit in memory.

    Returns:
        float: The mean Silhouette Score over all samples. Returns 0.0 if there
               is only one class.
    """
    # --- Input Validation and Edge Cases ---
    try:
        Q = np.asanyarray(Q, dtype=np.float64)
        y = np.asanyarray(y, dtype=int)
    except (ValueError, TypeError):
        raise TypeError("Inputs Q and y must be convertible to numpy arrays.")

    if Q.ndim != 2:
        raise ValueError("Input Q must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("Input y must be a 1D array.")
    if Q.shape[0] != y.shape[0]:
        raise ValueError("The number of samples in Q and y must be the same.")

    n_samples = Q.shape[0]
    unique_labels, y_indices = np.unique(y, return_inverse=True)
    n_classes = len(unique_labels)

    if n_samples < 2 or n_classes < 2:
        return 0.0

    # --- Core Calculation ---
    # For performance, compute all pairwise distances at once.
    # This is much faster than Python loops but uses O(n_samples^2) memory.
    try:
        pairwise_dists = squareform(pdist(Q, metric='euclidean'))
    except MemoryError:
        raise MemoryError(
            f"Failed to create a {n_samples}x{n_samples} pairwise distance matrix. "
            "The input array is too large for this memory-intensive approach."
        )

    # --- Vectorized calculation of a(i) and b(i) for all samples ---
    # a(i): Mean distance from sample i to all other points in the same class.
    # b(i): Mean distance from sample i to all points in the nearest other class.

    # 1. Calculate mean inter-cluster distances for all samples to all classes
    mean_inter_cluster_dists = np.zeros((n_samples, n_classes), dtype=np.float64)
    for c_idx, label in enumerate(unique_labels):
        mask_c = (y == label)
        # Mean distance from all samples to the samples in class c
        mean_inter_cluster_dists[:, c_idx] = np.mean(pairwise_dists[:, mask_c], axis=1)

    # 2. Extract a(i) and b(i)
    # a is the mean distance to a sample's own cluster
    a = mean_inter_cluster_dists[np.arange(n_samples), y_indices]

    # For b, set the distance to a sample's own cluster to infinity
    mean_inter_cluster_dists[np.arange(n_samples), y_indices] = np.inf
    # b is the minimum of the mean distances to all other clusters
    b = np.min(mean_inter_cluster_dists, axis=1)

    # 3. Calculate silhouette scores for each sample
    # s(i) = (b(i) - a(i)) / max(a(i), b(i))
    denominator = np.maximum(a, b)
    s = np.zeros_like(denominator) # Initialize scores to 0
    # Use a mask to avoid division by zero. If the denominator is 0, score is 0.
    mask_denom_ne_zero = denominator != 0
    s[mask_denom_ne_zero] = (b[mask_denom_ne_zero] - a[mask_denom_ne_zero]) / denominator[mask_denom_ne_zero]

    return float(np.mean(s) - np.std(s)) * factor_h * factor_k
