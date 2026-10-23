# ransomware-detection-prototype

## Ransomware Detection Using AI

## Timothy Shimman Huang 22059057

## Script Description
- main.py - runs the static and dynamic pipelines producing serialised models, figures and samples for the demo [FILE TO RUN]
- preprocessing/dynamic_preprocessing.py - converts CSV to a dataframe, removes unnecessary columns and sets X and Y
- preprocessing/pruning.py - performs variance and correlation pruning
- preprocessing/static_preprocessing.py - converts CSV to a dataframe, removes unnecessary columns and sets X and Y
- models.py - configures Random Forest, XGBoost, Logistic Regression and Soft-Voting Ensemble
- train.py - runs the stratified split and generates predictions and probabilities
- evaluate.py - calculates all classification and efficiency metrics for a model, outputting them as a dictionary
- results.py - creates the figures found in the report and saves them as PNGs in the figures folder
- detection_agent.py - runs the prototype detection agent using tkinter [FILE TO RUN]

## Python Version and Dependencies
This program was written in Python 3.12 using PyCharm 2026.2.1, scikit-learn 1.9.0 and xgboost 3.4.1
The dependencies required can be found in requirements.txt

## Datasets
Two datasets are required but not included due to file size limitations

Static - Kaggle Ransomware Detection Dataset
Download from: https://www.kaggle.com/datasets/amdj3dax/ransomware-detection-data-set
and save as: `datasets/static_ransom_data.csv`

Dynamic — EldeRAN
Download from: https://github.com/rissgrouphub/ransomwaredataset2016
and save as: `datasets/dynamic_ransom_data.csv`

Note: this file has no header row; column 0 is ID, column 1 is the Ransomware label,
column 2 is Ransomware_Family, and columns 3+ are features.

## Model Training Instructions**
1. Install the dependencies using `pip install -r requirements.txt`
2. Download both datasets and save them in datasets/ under the exact filenames given above
3. From the project root, run: `python main.py`

main.py creates saved_models/, demo_samples/ and figures/ automatically if they do not already exist and writes:
- saved_models/ - the 8 serialised models (4 static, 4 dynamic) plus static_features.pkl, the feature list the agent validates against
- demo_samples/ - 10 benignware and 10 ransomware labelled single-sample CSVs held out before the train/test split. Running main.py will overwrite demo_samples/; this is intended 
- figures/ - all bar charts, PR curves and confusion matrices as PNGs, plus static_results_table.csv and dynamic_results_table.csv

The expected runtime should be under 5 minutes.

## Agent Classification Instructions**
1. The agent requires the trained static models. Either use those included in saved_models/, or run main.py first to generate them
2. From the project root, run: `python detection_agent.py`
3. Click "Upload CSV" and select a file from demo_samples/
4. Select one of the four models from the list
5. Click "Scan File"

A messagebox reports the classification and the model's confidence. Each demo file is labelled so classifications can be verified.

The agent classifies pre-extracted PE header metadata rather than performing live behavioural monitoring. Section 3.2.2 of the report explains this scope reduction and Section 5.5 documents the agent's tested edge cases and known limitations.