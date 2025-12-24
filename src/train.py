from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from joblib import dump

from preprocessing import load_and_prepare_data, build_preprocessor

X_train, X_test, y_train, y_test = load_and_prepare_data(
    "../data/raw/telco_churn.csv"
)

preprocessor = build_preprocessor(X_train)

baseline_model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

roc_auc = cross_val_score(
    baseline_model,
    X_train,
    y_train,
    cv=5,
    scoring="roc_auc"
)

print("Baseline ROC-AUC:", roc_auc.mean())

baseline_model.fit(X_train, y_train)
dump(baseline_model, "../models/baseline_model.pkl")
