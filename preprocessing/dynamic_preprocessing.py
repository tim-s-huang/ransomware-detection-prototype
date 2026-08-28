from pathlib import Path
import pandas as pd
from preprocessing.pruning import correlation_pruning, variance_pruning

def get_dynamic_data():
    # Locates Dynamic Dataset and Saves as Dataframe
    base = Path(__file__).parent.parent
    dynamic_file_path = base / "datasets" / "dynamic_ransom_data.csv"
    dynamic_ransom_data = pd.read_csv(dynamic_file_path, header=None)

    # Debugged with Claude - Assigns Column Names based on Dataset Documentation
    n_cols = dynamic_ransom_data.shape[1]
    dynamic_ransom_data.columns = (
            ["ID", "Ransomware", "Ransomware_Family"]
            + [f"feature_{i}" for i in range(n_cols - 3)]
    )

    # Configure Target and Unneeded Columns
    target_column = "Ransomware"
    drop_columns = ["ID", "Ransomware_Family"]

    # Drop Unneeded Columns
    dynamic_ransom_data = dynamic_ransom_data.dropna()
    dynamic_ransom_data = dynamic_ransom_data.drop(columns=[c for c in drop_columns if c in dynamic_ransom_data.columns])

    # Sets X and Y
    X = dynamic_ransom_data.drop(columns=target_column)
    y = dynamic_ransom_data[target_column]

    # Performs Variance and Correlation Pruning on X
    X = variance_pruning(X, 0.01)
    X = correlation_pruning(X, 0.95)

    return X, y
