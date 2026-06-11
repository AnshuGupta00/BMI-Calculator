# ============================================================
# BMI Health Risk Classification — ML Model
# Uses scikit-learn to classify health risk from BMI data
# Author: Anshu Gupta
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD & INSPECT DATA ────────────────────────────────────────────────────
print("=" * 60)
print("  BMI Health Risk Classification — ML Pipeline")
print("=" * 60)

# Load the existing bmi.csv from the repo
df = pd.read_csv('bmi.csv')
print(f"\n✅ Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nNull values:\n{df.isnull().sum()}")

# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
# WHO standard BMI risk categories (4 classes)
def assign_risk_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25.0:
        return 'Normal'
    elif bmi < 30.0:
        return 'Overweight'
    else:
        return 'Obese'

# Calculate BMI if not already present
if 'BMI' not in df.columns and 'bmi' not in df.columns:
    # Assumes columns for height (m) and weight (kg) exist
    height_col = [c for c in df.columns if 'height' in c.lower()][0]
    weight_col = [c for c in df.columns if 'weight' in c.lower()][0]
    df['BMI'] = df[weight_col] / (df[height_col] ** 2)
    print(f"\n✅ BMI calculated from {height_col} and {weight_col}")

# Standardize BMI column name
bmi_col = 'BMI' if 'BMI' in df.columns else 'bmi'

# Assign WHO-standard 4-class risk labels
df['Risk_Category'] = df[bmi_col].apply(assign_risk_category)
print(f"\n✅ WHO Risk Categories assigned:")
print(df['Risk_Category'].value_counts())

# ── 3. PREPARE FEATURES & TARGET ─────────────────────────────────────────────
# Use BMI as primary feature; add derived features for better model performance
df['BMI_squared']  = df[bmi_col] ** 2
df['BMI_log']      = np.log(df[bmi_col])

feature_cols = [bmi_col, 'BMI_squared', 'BMI_log']

# Add any other numeric columns as features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
extra_features = [c for c in numeric_cols
                  if c not in feature_cols + ['Risk_Category']
                  and 'risk' not in c.lower()]
feature_cols += extra_features

print(f"\n✅ Features used: {feature_cols}")

X = df[feature_cols].fillna(df[feature_cols].median())
y = df['Risk_Category']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\n✅ Classes: {list(le.classes_)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 4. TRAIN / TEST SPLIT ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"\n✅ Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── 5. MODEL 1 — RANDOM FOREST ───────────────────────────────────────────────
print("\n" + "─" * 60)
print("  MODEL 1: Random Forest Classifier")
print("─" * 60)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)

print(f"\n✅ Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
print(f"\nClassification Report:\n")
print(classification_report(y_test, rf_preds, target_names=le.classes_))

# Cross-validation
rf_cv = cross_val_score(rf_model, X_scaled, y_encoded, cv=5, scoring='accuracy')
print(f"5-Fold Cross-Validation: {rf_cv.mean()*100:.2f}% ± {rf_cv.std()*100:.2f}%")

# ── 6. MODEL 2 — LOGISTIC REGRESSION ────────────────────────────────────────
print("\n" + "─" * 60)
print("  MODEL 2: Logistic Regression")
print("─" * 60)

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
    multi_class='multinomial'
)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_preds)

print(f"\n✅ Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")
print(f"\nClassification Report:\n")
print(classification_report(y_test, lr_preds, target_names=le.classes_))

lr_cv = cross_val_score(lr_model, X_scaled, y_encoded, cv=5, scoring='accuracy')
print(f"5-Fold Cross-Validation: {lr_cv.mean()*100:.2f}% ± {lr_cv.std()*100:.2f}%")

# ── 7. MODEL COMPARISON ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MODEL COMPARISON SUMMARY")
print("=" * 60)
print(f"  Random Forest      : {rf_accuracy * 100:.2f}%  (CV: {rf_cv.mean()*100:.2f}%)")
print(f"  Logistic Regression: {lr_accuracy * 100:.2f}%  (CV: {lr_cv.mean()*100:.2f}%)")
best = "Random Forest" if rf_accuracy >= lr_accuracy else "Logistic Regression"
print(f"\n  ✅ Best Model: {best}")

# ── 8. FEATURE IMPORTANCE (Random Forest) ────────────────────────────────────
print("\n" + "─" * 60)
print("  FEATURE IMPORTANCE")
print("─" * 60)
importances = rf_model.feature_importances_
for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {feat:<20} {bar} {imp:.4f}")

# ── 9. VISUALIZATIONS ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('BMI Health Risk Classification — ML Analysis', fontsize=14, fontweight='bold')

# Plot 1: Risk category distribution
category_counts = df['Risk_Category'].value_counts()
colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']
axes[0].bar(category_counts.index, category_counts.values, color=colors)
axes[0].set_title('WHO Risk Category Distribution')
axes[0].set_xlabel('Risk Category')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=15)

# Plot 2: Confusion Matrix — Random Forest
cm = confusion_matrix(y_test, rf_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=axes[1], colorbar=False, cmap='Blues')
axes[1].set_title(f'Random Forest\nAccuracy: {rf_accuracy*100:.1f}%')
axes[1].tick_params(axis='x', rotation=15)

# Plot 3: Model accuracy comparison
models = ['Random Forest', 'Logistic Regression']
accuracies = [rf_accuracy * 100, lr_accuracy * 100]
bar_colors = ['#2563EB', '#1A2B4A']
bars = axes[2].bar(models, accuracies, color=bar_colors, width=0.4)
axes[2].set_ylim(0, 110)
axes[2].set_title('Model Accuracy Comparison')
axes[2].set_ylabel('Accuracy (%)')
for bar, acc in zip(bars, accuracies):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('bmi_ml_results.png', dpi=150, bbox_inches='tight')
print("\n✅ Visualization saved as 'bmi_ml_results.png'")
plt.show()

# ── 10. PREDICTION DEMO ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  LIVE PREDICTION DEMO")
print("=" * 60)

def predict_risk(bmi_value, model=rf_model, scaler=scaler, le=le):
    bmi_sq  = bmi_value ** 2
    bmi_log = np.log(bmi_value)
    features = np.array([[bmi_value, bmi_sq, bmi_log] +
                          [0] * (len(feature_cols) - 3)])
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)
    prob = model.predict_proba(features_scaled)[0]
    category = le.inverse_transform(pred)[0]
    confidence = max(prob) * 100
    return category, confidence

test_bmis = [17.2, 22.5, 27.8, 34.1]
print(f"\n  {'BMI':<10} {'Predicted Risk':<20} {'Confidence'}")
print(f"  {'─'*45}")
for bmi_val in test_bmis:
    cat, conf = predict_risk(bmi_val)
    print(f"  {bmi_val:<10} {cat:<20} {conf:.1f}%")

print("\n" + "=" * 60)
print("  ✅ Pipeline complete. Model ready for deployment.")
print("=" * 60)
