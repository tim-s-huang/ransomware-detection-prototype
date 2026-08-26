### Static Data Preprocessing ###

from pathlib import Path
import pandas as pd
from preprocessing.correlation_pruning import correlation_pruning


def get_static_data():
    # Create Static Dataset
    base = Path(__file__).parent.parent
    static_file_path = base/"datasets"/"static_ransom_data.csv"

    static_ransom_data = pd.read_csv(static_file_path)

    # Configure Target and Drop Excess Columns
    target_column = "Benign"
    drop_columns = ["FileName", "md5Hash"]

    static_ransom_data = static_ransom_data.dropna()
    static_ransom_data = static_ransom_data.drop(columns=[c for c in drop_columns if c in static_ransom_data.columns])

    X = static_ransom_data.drop(columns=target_column)
    y = static_ransom_data[target_column]

    # Ensures Static and Dynamic use Same Targets
    y = 1 - static_ransom_data[target_column]

    X = correlation_pruning(X, 0.95)
    return X, y