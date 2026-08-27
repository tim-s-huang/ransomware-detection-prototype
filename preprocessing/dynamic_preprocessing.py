from pathlib import Path
import pandas as pd
from preprocessing.pruning import correlation_pruning, variance_pruning

def get_dynamic_data():
    base = Path(__file__).parent.parent
    dynamic_file_path = base / "datasets" / "dynamic_ransom_data.csv"

    dynamic_ransom_data = pd.read_csv(dynamic_file_path, header=None)

    n_cols = dynamic_ransom_data.shape[1]
    dynamic_ransom_data.columns = (
            ["ID", "Ransomware", "Ransomware_Family"]
            + [f"feature_{i}" for i in range(n_cols - 3)]
    )

    # Configure Target and Drop Excess Columns
    target_column = "Ransomware"
    drop_columns = ["ID", "Ransomware_Family"]

    dynamic_ransom_data = dynamic_ransom_data.dropna()
    dynamic_ransom_data = dynamic_ransom_data.drop(columns=[c for c in drop_columns if c in dynamic_ransom_data.columns])

    X = dynamic_ransom_data.drop(columns=target_column)
    y = dynamic_ransom_data[target_column]

    X = variance_pruning(X, 0.01)
    X = correlation_pruning(X, 0.95)

    return X, y
