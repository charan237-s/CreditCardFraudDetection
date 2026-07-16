import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay
)

from imblearn.over_sampling import SMOTE

# ==========================================
# CREATE FOLDERS
# ==========================================

os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ==========================================
# LOAD DATASET
# ==========================================

print("=" * 60)
print("LOADING DATASET...")
print("=" * 60)

df = pd.read_csv("data/creditcard.csv")

print("\nDataset Loaded Successfully!")

# ==========================================
# BASIC INFORMATION
# ==========================================

print("\nFirst 5 Rows")
print(df.head())

print("\nShape")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows :", df.duplicated().sum())

# ==========================================
# REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates")
print(df.shape)

# ==========================================
# STATISTICS
# ==========================================

print("\nStatistical Summary")
print(df.describe())

# ==========================================
# FRAUD DISTRIBUTION
# ==========================================

print("\nFraud Distribution")
print(df["Class"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x="Class", data=df)
plt.title("Fraud vs Genuine Transactions")
plt.savefig("images/fraud_distribution.png")
plt.show()

# ==========================================
# AMOUNT DISTRIBUTION
# ==========================================

plt.figure(figsize=(10,5))
sns.histplot(df["Amount"], bins=50)
plt.title("Transaction Amount Distribution")
plt.savefig("images/amount_distribution.png")
plt.show()

# ==========================================
# CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(18,14))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("images/correlation_heatmap.png")
plt.show()

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop("Class", axis=1)
y = df["Class"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed")

# ==========================================
# SMOTE
# ==========================================

print("\nBefore SMOTE")
print(y_train.value_counts())

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE")
print(pd.Series(y_train).value_counts())

# ==========================================
# MODEL COMPARISON
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )
}

results = []

for name, model in models.items():

    print("\n" + "="*50)
    print(f"Training {name}")
    print("="*50)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    roc = roc_auc_score(y_test, y_pred)

    print("Accuracy :", accuracy)

    print("ROC AUC :", roc)

    results.append([name, accuracy, roc])

best_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

best_model.fit(X_train, y_train)

joblib.dump(best_model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# RESULTS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred)

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(f"Accuracy : {accuracy:.4f}")
print(f"ROC AUC  : {roc:.4f}")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ==========================================
# CONFUSION MATRIX GRAPH
# ==========================================

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()

plt.title("Confusion Matrix")

plt.savefig("images/confusion_matrix.png")

plt.show()

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,8))

sns.barplot(
    data=importance.head(15),
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Important Features")

plt.savefig("images/feature_importance.png")

plt.show()

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel Saved Successfully!")

print("models/fraud_model.pkl")
print("models/scaler.pkl")

print("\nProject Completed Successfully!")
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "ROC AUC"]
)

print("\n")
print(results_df)
plt.figure(figsize=(8,5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy"
)

plt.title("Model Comparison")

plt.savefig("images/model_comparison.png")

plt.show()