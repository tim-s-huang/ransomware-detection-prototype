# Ransomware Detection with Machine Learning

A Python tool that classifies Windows executables as **ransomware or benign** using machine learning, with a simple desktop app to scan samples. Built for my MSc Cyber Security dissertation at Cardiff University, 2026.

<img width="839" height="533" alt="detection agent" src="https://github.com/user-attachments/assets/bd46e0d5-0f61-453f-a941-d175457fce5d" />

## What it does
- **Trains and compares four models**: Logistic Regression, Random Forest, XGBoost and a soft-voting ensemble
- **Uses two kinds of data**: *static* features (PE header metadata) and *dynamic* features (behaviour recorded in a sandbox)
- **Measures what matters for real users**: precision, false positives and how fast each model runs per sample
- **Includes a desktop scanner** (tkinter): upload a sample, pick a model, get a verdict with a confidence score

## Results
| Model | Static accuracy | Static false positive rate | Dynamic accuracy |
|---|---|---|---|
| Logistic Regression | 89.71% | 10.47% | 97.05% |
| Random Forest | 99.67% | 0.35% | 98.03% |
| XGBoost | 99.68% | 0.41% | 97.38% |
| Ensemble | 99.68% | 0.46% | 98.36% |

Found XGBoost and Ensemble performed the best on the static features and Ensemble performed the best on the dynamic features. 
Logistic Regression was found to have the highest trade-off of inference time versus classification metrics, but on the behavioural pipeline.

## Use the Agent
Pre-trained models and demo samples are included, so no datasets are needed.
```bash
pip install -r requirements.txt
python detection_agent.py
```
Click **Upload CSV**, choose a file from `demo_samples/` (each is labelled benign or ransomware so you can check the answer), pick a model and click **Scan File**.

## Retrain the models
1. Download the two datasets (links below) into `datasets/` as `static_ransom_data.csv` and `dynamic_ransom_data.csv`
2. Run `python main.py`. It takes under 5 minutes and regenerates the models, demo samples and all result charts in `figures/`.

## How it works
`preprocessing/` cleans the data and prunes low-variance and highly correlated features → `models.py` sets up the four models → `train.py` runs an 80/20 stratified split → `evaluate.py` calculates the metrics → `results.py` draws the charts → `detection_agent.py` runs the scanner.

## Limitations
The scanner classifies **pre-extracted** PE header features rather than monitoring live programs, so it's a proof of concept, not real-time protection. Dynamic models are trained and evaluated but not wired into the scanner.

## Built with
Python 3.12 · scikit-learn · XGBoost · pandas · NumPy · matplotlib · tkinter

## Datasets and credits
- **Static:** [Kaggle Ransomware Detection Dataset](https://www.kaggle.com/datasets/amdj3dax/ransomware-detection-data-set) (Apache 2.0). The 20 files in `demo_samples/` are rows taken from it.
- **Dynamic:** [EldeRAN](https://github.com/rissgrouphub/ransomwaredataset2016). Sgandurra, D., Muñoz-González, L., Mohsen, R. and Lupu, E.C. (2016). *Automated Dynamic Analysis of Ransomware: Benefits, Limitations and use for Detection.* arXiv:1609.03020.
