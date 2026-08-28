import numpy as np

# Written by Claude - Performs Correlation Pruning
def correlation_pruning(X, threshold):
    corr_matrix = X.corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    x_pruned = X.drop(columns=to_drop)

    return x_pruned

# Written by Claude - Performs Variance Pruning
def variance_pruning(X, threshold=0.01):
    variances = X.var()
    keep = variances[variances > threshold].index
    X_pruned = X[keep]

    return X_pruned