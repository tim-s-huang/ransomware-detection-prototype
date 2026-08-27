from preprocessing.static_preprocessing import get_static_data
from preprocessing.dynamic_preprocessing import get_dynamic_data
from models import get_models
from train import run_model
from evaluate import calculate_metrics
from results import display_results
import pickle
from pathlib import Path
Path("saved_models").mkdir(exist_ok=True)

# Debugged with Claude
def save_testing_data(X_static, y_static, n_per_class=10):
    ransomware_idx = y_static[y_static == 1].sample(n_per_class, random_state=42).index
    benign_idx = y_static[y_static == 0].sample(n_per_class, random_state=42).index
    demo_idx = ransomware_idx.union(benign_idx)

    Path("demo_samples").mkdir(exist_ok=True)

    for i, idx in enumerate(demo_idx):
        sample = X_static.loc[[idx]]
        label = "ransomware" if y_static.loc[idx] == 0 else "benign"
        sample.to_csv(f"demo_samples/{label}_{i:02d}.csv", index=False)

    return X_static.drop(demo_idx), y_static.drop(demo_idx)

# Calculates Minority / Majority classes
def generate_ratio(y_static):
    counts = y_static.value_counts()
    return counts.max() / counts.min()

# Get Static File Metadata
X_static, y_static = get_static_data()
X_static, y_static = save_testing_data(X_static, y_static)

# Save Feature List for Agent
with open("saved_models/static_features.pk1", "wb") as f:
    pickle.dump(list(X_static.columns), f)

# Calculate Ratio for XGBoost Weighting Static
ratio_static = generate_ratio(y_static)

static_model_results = {}

# Loop through ML models for Static Analysis
for name, model in get_models(ratio_static).items():
    val_x, val_y, preds, probs = run_model(X_static, y_static, model)

    # Save Metrics as a Dictionary for Plotting
    model_results = calculate_metrics(model, val_x, val_y, preds, probs)
    static_model_results[name] = model_results

    # Save Models for Detection Agent
    filename = f"saved_models/static_{name.replace(' ', '_').lower()}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


# Get Dynamic API Call Data
X_dynamic, y_dynamic = get_dynamic_data()

# Calculate Ratio for XGBoost Weighting Dynamic
ratio_dynamic = generate_ratio(y_dynamic)

dynamic_model_results = {}

# Loop through ML models for Dynamic Analysis
for name, model in get_models(ratio_dynamic).items():
    val_x, val_y, preds, probs = run_model(X_dynamic, y_dynamic, model)

    # Print Accuracy and Time Metrics
    model_results = calculate_metrics(model, val_x, val_y, preds, probs)
    dynamic_model_results[name] = model_results

    # Save Models for Detection Agent (Full System Requirement)
    filename = f"saved_models/dynamic_{name.replace(' ', '_').lower()}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

# Display Model Performance and Time Metrics as Graphs
display_results(static_model_results, dynamic_model_results)