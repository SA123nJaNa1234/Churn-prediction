from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from joblib import dump

from preprocessing import DATA_PATH, load_and_prepare_data, build_preprocessor

X_train, X_test, y_train, y_test = load_and_prepare_data(
    DATA_PATH
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



from sklearn.ensemble import RandomForestClassifier

final_model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42
    ))
])

roc_auc = cross_val_score(
    final_model,
    X_train,
    y_train,
    cv=5,
    scoring="roc_auc"
)

print("Final Model ROC-AUC:", roc_auc.mean())

final_model.fit(X_train, y_train)
dump(final_model, "../models/best_model.pkl")

