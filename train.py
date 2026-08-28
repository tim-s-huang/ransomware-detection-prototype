from sklearn.model_selection import train_test_split

def run_model(X, y, model_select):
    # Stratified Train-Test Split
    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    # Fit ML model with Training Data
    model_select.fit(train_x, train_y)

    # Generate Predictions for Testing Data
    predictions = model_select.predict(val_x)

    # Generate Probabilities of Model Accuracy
    probabilities = model_select.predict_proba(val_x)[:,1]

    return val_x, val_y, predictions, probabilities





