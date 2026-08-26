import numpy as np

# This Function was Written by Claude
def correlation_pruning(X, threshold):
    corr_matrix = X.corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    x_pruned = X.drop(columns=to_drop)

    return x_pruned