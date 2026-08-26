from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Debugged with Claude
def get_models(ratio):
    rf = RandomForestClassifier(
        n_estimators = 200,
        random_state = 0,
        class_weight="balanced"
    )

    xgb = XGBClassifier(
        n_estimators = 200,
        random_state = 0,
        eval_metric = "logloss",
        scale_pos_weight = ratio
    )

    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter = 5000,
            solver= "liblinear",
            random_state = 0,
            class_weight = "balanced"
        )
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lr", lr)],
        voting="soft",
    )

    return {"Random Forest" : rf, "XGBoost" : xgb, "Logistic Regression": lr, "Ensemble": ensemble}

