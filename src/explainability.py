import pandas as pd
from joblib import load

model = load("../models/best_model.pkl")

preprocessor = model.named_steps["preprocess"]
classifier = model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()
importances = classifier.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("Top churn drivers:")
print(importance_df.head(10))
