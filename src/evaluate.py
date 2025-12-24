from joblib import load
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from preprocessing import load_and_prepare_data

model = load("../models/best_model.pkl")

X_train, X_test, y_train, y_test = load_and_prepare_data(
    "../data/raw/telco_churn.csv"
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
