import pickle
import pandas as pd
import tkinter
from tkinter import filedialog
from tkinter import messagebox

TITLE_FONT = ("arial", 12, "bold")
SUBTITLE_FONT = ("arial", 10, "normal")
STANDARD_FONT = ("arial", 8, "normal")
MODEL_FILES = {
    "Random Forest": "static_random_forest.pkl",
    "XGBoost": "static_xgboost.pkl",
    "Logistic Regression": "static_logistic_regression.pkl",
    "Ensemble": "static_ensemble.pkl"
}
MODELS = list(MODEL_FILES.keys())

agent_model = None

# Loads List of Expected Features for Uploaded CSV
with open("saved_models/static_features.pkl", "rb") as f:
    loaded_expected_features = pickle.load(f)

# Debugged with Claude
def select_file():

    # Allows User to Select CSV
    user_csv_path = filedialog.askopenfilename(
        title="Select PE feature CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    # Sets Selected User File and Updates File Label
    if user_csv_path:
        selected_file_path.set(user_csv_path)
        file_label.config(text=f"Selected: {user_csv_path.split('/')[-1]}")
    else:
        file_label.config(text="No file selected")

# Debugged with Claude
def load_csv(path, expected_features):
    user_file_csv = pd.read_csv(path)

    # Drops Identifier and Target Columns
    user_file_csv = user_file_csv.drop(columns=["FileName", "md5Hash", "Benign"], errors="ignore")

    # Checks CSV has Required Features
    missing_features = set(expected_features) - set(user_file_csv.columns)
    if missing_features:
        return None, f"Missing features: {missing_features}"

    # Checks CSV is a Singular Row
    if len(user_file_csv) != 1:
        return None, f"Expected a single sample, found {len(user_file_csv)} rows"

    # Arranges CSV into Expected Header Order
    user_file_csv = user_file_csv[expected_features]

    return user_file_csv, None

def load_model(event):
    global agent_model

    # Gets Model Selection from Listbox Widget
    selection = model_select_listbox.curselection()

    # Loads the Correct ML Model from Saved Models
    if selection:
        name = MODELS[selection[0]]
        selected_model_name.set(name)
        with open(f"saved_models/{MODEL_FILES[name]}", "rb") as f:
            agent_model = pickle.load(f)
    else:
        return

# Inputs Provided File to Selected Model and returns Errorbox if Invalid
def scan_file():
    path = selected_file_path.get()

    # Shows Errorbox if File not Selected
    if not path:
        messagebox.showwarning(title="File Select Error", message="No File Selected!")
        return

    # Shows Errorbox if Model not Selected
    if agent_model is None:
        messagebox.showwarning(title="Model Select Error", message="No Model Selected!")
        return

    # Show Errorbox if CSV of PE Header Features is Invalid
    sample, error = load_csv(path, loaded_expected_features)
    if error:
        messagebox.showwarning(title="File Features Error", message="PE Header Features Format Invalid! ")
        return

    # Predicts File Classification and Probability
    pred = int(agent_model.predict(sample)[0])
    prob = agent_model.predict_proba(sample)
    prob = float(prob[0][pred])

    # Show Alertbox if CSV is Ransomware
    if pred == 1:
        messagebox.showinfo(title="Ransomware Detected!!!", message=f"{selected_file_path.get().split('/')[-1]} has been classified as Ransomware by {selected_model_name.get()} ({prob:.1%} confidence)")
    else:
        # Show Alertbox if CSV is Benignware - Prototype only functionality
        messagebox.showinfo(title="Benignware...",message=f"{selected_file_path.get().split('/')[-1]} has been classified as Benign using {selected_model_name.get()} ({prob:.1%} confidence)")


# Create Tkinter Interface
window = tkinter.Tk()
window.title("Ransomware Detection with AI Prototype Agent")
window.minsize(400, 260)
window.config(padx=30, pady=5)

# Tkinter Variables for Selected File and Model
selected_file_path = tkinter.StringVar(value="")
selected_model_name = tkinter.StringVar(value="")

# Instruction Label for CSV File Selection
input_csv_label = tkinter.Label(text="Select a CSV file containing PE header features", font=TITLE_FONT)
input_csv_label.grid(column=1, row=0, pady=5)

# Button for CSV File Selection
select_file_button = tkinter.Button(text="Upload CSV", command=select_file)
select_file_button.grid(column=1, row=1, pady=0)

# Info Label of File Currently Selected
file_label = tkinter.Label(text="No File Selected", font=STANDARD_FONT)
file_label.grid(column=1, row=2, pady=(0, 10))

# Instruction Label for Model Selection
info_label = tkinter.Label(text="Select the model for classification", font=TITLE_FONT)
info_label.grid(column=1, row=3, pady=(10, 0))

# Listbox for Model Selection
model_select_listbox = tkinter.Listbox(height=4)
for model in MODELS:
    model_select_listbox.insert(MODELS.index(model), model)
model_select_listbox.bind("<<ListboxSelect>>", load_model)
model_select_listbox.grid(column=1, row=4, pady=5)

# Button for Running File Classification
scan_button = tkinter.Button(text="Scan File", command=scan_file)
scan_button.grid(column=1, row=5, pady=(10, 2))

window.mainloop()







