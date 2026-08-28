from pathlib import Path
import pandas as pd
from preprocessing.pruning import correlation_pruning, variance_pruning


def get_static_data():
    # Locates Static Dataset and Saves as Dataframe
    base = Path(__file__).parent.parent
    static_file_path = base/"datasets"/"static_ransom_data.csv"
    static_ransom_data = pd.read_csv(static_file_path)

    # Configure Target and Unneeded Columns
    target_column = "Benign"
    drop_columns = ["FileName", "md5Hash"]

    # Drop Unneeded Columns
    static_ransom_data = static_ransom_data.dropna()
    static_ransom_data = static_ransom_data.drop(columns=[c for c in drop_columns if c in static_ransom_data.columns])

    # Sets X
    X = static_ransom_data.drop(columns=target_column)

    # Ensures Static and Dynamic use Same Target Value - 1 for Ransom, 0 for Benign
    y = 1 - static_ransom_data[target_column]

    # Performs Variance and Correlation Pruning on X
    X = variance_pruning(X, 0.01)
    X = correlation_pruning(X, 0.95)
    return X, y