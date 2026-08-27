from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    average_precision_score
)
import time
import psutil
import os
import pickle

def calculate_metrics(model, val_x, val_y, predictions, probabilities):

    # Calculates Confusion Matrix
    matrix_score = confusion_matrix(val_y, predictions)
    tn, fp, fn, tp = matrix_score.ravel()

    # Calculates False Positive Rate and False Negative Rate
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    # Access Current Python Process
    current_process = psutil.Process(os.getpid())

    # Record CPU Baseline and Start Time before Classification
    cpu_before = current_process.cpu_percent(interval=None)
    start_time = time.perf_counter()

    # Prediction Output Discarded
    preds = model.predict(val_x)

    # Record CPU Usage and End Time after Classification
    end_time = time.perf_counter()
    cpu_after = current_process.cpu_percent(interval=None)

    # Calculate Total Runtime
    total_runtime = end_time - start_time
    per_sample_runtime = (total_runtime / len(val_x)) * 1000

    # Measure Model Size as a Proxy - Debugged by Claude
    model_size = len(pickle.dumps(model)) / 1024 / 1024

    return {
        "Accuracy":  round(accuracy_score(val_y, predictions), 4),
        "Precision": round(precision_score(val_y, predictions), 4),
        "Recall": round(recall_score(val_y, predictions), 4),
        "F1-score": round(f1_score(val_y, predictions), 4),
        "Confusion Matrix": matrix_score,
        "False Positive Rate": round(fpr, 4),
        "False Negative Rate": round(fnr, 4),
        "MCC": round(matthews_corrcoef(val_y, predictions) ,4),
        "PR-AUC": round(average_precision_score(val_y, probabilities), 4),
        "Total Inference Time": round(total_runtime, 4),
        "Inference Time Per Sample": round(per_sample_runtime, 4),
        "CPU Usage": round(cpu_after / psutil.cpu_count(), 4),
        "Model Size": round(model_size, 4),
        "Probabilities": probabilities,
        "Y-Values": val_y
    }
