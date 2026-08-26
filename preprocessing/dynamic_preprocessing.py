from pathlib import Path
import pandas as pd
from preprocessing.correlation_pruning import correlation_pruning

def get_dynamic_data():
    base = Path(__file__).parent.parent
    dynamic_file_path = base / "datasets" / "dynamic_ransom_data.csv"

    dynamic_ransom_data = pd.read_csv(dynamic_file_path)

    # Configure Target and Drop Excess Columns
    target_column = "malware"
    drop_columns = ["hash"]

    dynamic_ransom_data = dynamic_ransom_data.dropna()
    dynamic_ransom_data = dynamic_ransom_data.drop(columns=[c for c in drop_columns if c in dynamic_ransom_data.columns])

    ## Correlation Pruning

    X = dynamic_ransom_data.drop(columns=target_column)
    y = dynamic_ransom_data[target_column]

    X = correlation_pruning(X, 0.95)
    return X, y
