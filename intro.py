import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# === Load the Iris Dataset ===
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Show dataset information
print("🔹 Feature names:", feature_names)
print("🔹 Target classes:", target_names)

# Create DataFrame for analysis
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[i] for i in y]

# === Save Pairplot ===
sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.suptitle("Iris Dataset Feature Pairplot", y=1.02)
plt.tight_layout()
plt.savefig("iris_pairplot.png")
print("📊 Pairplot saved as iris_pairplot.png")

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# === Train RandomForest Classifier ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate Model ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy:.4f}")
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# === Show Feature Importances ===
importances = model.feature_importances_
print("🌟 Feature Importances:")
for name, importance in zip(feature_names, importances):
    print(f"{name:25s}: {importance:.4f}")

# === Save Model to File ===
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\n💾 Model saved to iris_model.pkl")

# === Save Metadata (optional) ===
with open("iris_metadata.pkl", "wb") as f:
    pickle.dump({
        "feature_names": feature_names,
        "target_names": target_names.tolist()
    }, f)
print("💾 Metadata saved to iris_metadata.pkl")

