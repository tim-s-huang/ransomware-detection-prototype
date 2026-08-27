import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay

Path("figures").mkdir(exist_ok=True)

# Plots Accuracy Graph
def plot_bar_chart(results, y_axis_name, title, filename, log_scale):
    # Turn Model Names and Values Into Lists
    names_list = list(results)
    y_axis_values = [results[name][y_axis_name] for name in results]

    # Creates Figure and Axes
    fig, ax = plt.subplots(figsize=(9, 4))

    # Creates Bar Objects
    bars = ax.bar(names_list, y_axis_values, width=0.6, color="#4C72B0", edgecolor="black", linewidth=0.6)

    # Set Y Scale
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(min(y_axis_values) * 0.5, max(y_axis_values) * 3)
    else:
        ax.set_ylim(0, 1.08)

    # Loops Over Each Bar and Adds its Value - Debugged with Claude
    for bar, y_val in zip(bars, y_axis_values):
        label_y = bar.get_height() * 1.1 if log_scale else bar.get_height() + 0.005
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{y_val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    # Adds Axis Label and Title
    ax.set_ylabel(y_axis_name)
    ax.set_title(title)

    # Adds Dashed Horizontal Grid Lines
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Saves PNG and closes Graph from Memory
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Debugged with Calude
def plot_grouped_bar_chart(static_results, dynamic_results, metric, title, filename):
    # Turn Model names and Values into Lists
    model_names = list(static_results)
    static_values = [static_results[name][metric] for name in model_names]
    dynamic_values = [dynamic_results[name][metric] for name in model_names]

    # Set X Position for Each Group
    x = np.arange(len(model_names))
    bar_width = 0.35

    # Creates Figure and Axes
    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Offsets Bars
    static_bars = ax.bar(
        x - bar_width / 2, static_values, bar_width,
        label="Static", color="#4C72B0", edgecolor="black", linewidth=0.6
    )
    dynamic_bars = ax.bar(
        x + bar_width / 2, dynamic_values, bar_width,
        label="Dynamic", color="#DD8452", edgecolor="black", linewidth=0.6
    )

    # Set Y Scale
    ax.set_ylim(0, max(static_values + dynamic_values) * 1.2)

    # Loops Over Each Bar and Adds its Value
    for bars in (static_bars, dynamic_bars):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8
            )

    # Adds Y Axis Label and Title
    ax.set_ylabel(metric)
    ax.set_title(title)

    # Adds X Axis Labels
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    # Adds Dashed Horizontal Grid Lines
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Saves PNG and closes Graph from Memory
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_pr_curves(results, title, filename):
    # Creates Figure and Axes
    fig, ax = plt.subplots(figsize=(7, 6))

    line_types = ["-", "--", "-.", ":"]

    # Plots Y-Value to Probabilities to Create Curves
    for i, name in enumerate(results):
        display = PrecisionRecallDisplay.from_predictions(
            results[name]["Y-Values"],
            results[name]["Probabilities"],
            name=name,
            ax=ax
        )

        # Make Each Curve Distinguishable
        display.line_.set_linestyle(line_types[i % len(line_types)])
        display.line_.set_alpha(0.8)

    # Add Dashed Grid Lines
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title(title)

    # Saves PNG and closes Graph from Memory
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_confusion_matrix(results, model_name, title, filename):
    # Creates Figure and Axes
    fig, ax = plt.subplots(figsize=(5, 4))

    # Plot Confusion Matrix with Benign/Ransomware Labels
    display = ConfusionMatrixDisplay(
        confusion_matrix=results[model_name]["Confusion Matrix"],
        display_labels=["Benign", "Ransomware"]
    )
    display.plot(cmap="Blues", ax=ax, values_format="d", colorbar=False)

    # Adds Title, Saves PNG and closes Graph from Memory
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Save Results to CSV - Debugged with Claude
def export_results_table(results, filename):
    exclude = ("Probabilities", "Y-Values", "Confusion Matrix")
    clean = {name: {k: v for k, v in metrics.items() if k not in exclude}
             for name, metrics in results.items()}
    pd.DataFrame(clean).T.to_csv(filename)

def display_results(static_models_results, dynamic_models_results):
    # Plot Bar Charts
    bar_chart_metrics = ["Accuracy", "MCC", "False Positive Rate", "False Negative Rate", "Precision", "Recall", "F1-score", "Inference Time Per Sample", "Model Size"]

    pipelines = {
        "Static": static_models_results,
        "Dynamic": dynamic_models_results
    }

    # Loop Metric for each Pipeline to Plot Bar Charts
    for pipeline_name, results in pipelines.items():
        for metric in bar_chart_metrics:
            filename = f"figures/{pipeline_name.lower()}_{metric.replace(' ', '_').lower()}.png"
            title = f"{pipeline_name} Pipeline: {metric} by Model"

            # Choose Scale Type and Plot Bar Charts
            if metric == "Inference Time Per Sample" or metric == "Model Size":
                plot_bar_chart(results, metric, title, filename, True)
            else:
                plot_bar_chart(results, metric, title, filename, False)

    # Plot Grouped Bar Charts
    for metric in ["MCC", "False Positive Rate"]:
        filename = f"figures/comparison_{metric}.png"
        title = f"{metric} by Model: Static vs Dynamic Pipeline"

        plot_grouped_bar_chart(static_models_results, dynamic_models_results, metric, title, filename)

    # Plot PR-AUC Line Chart
    for pipeline_name, results in pipelines.items():
        filename = f"figures/{pipeline_name.lower()}_auc.png"
        title = f"{pipeline_name} Pipeline: AUC-PR by Model"

        plot_pr_curves(results, title, filename)

    # Plot Confusion Matrices
    for pipeline_name, results in pipelines.items():
        for model_name in ["Random Forest", "XGBoost", "Logistic Regression", "Ensemble"]:
            filename = f"figures/{pipeline_name.lower()}_{model_name.replace(' ', '_').lower()}_matrix.png"
            title = f"{pipeline_name} Pipeline: {model_name} Confusion Matrix"

            plot_confusion_matrix(results, model_name, title, filename)

    # Saves Results to CSV Files
    for pipeline_name, results in pipelines.items():
        export_results_table(results, f"figures/{pipeline_name.lower()}_results_table.csv")