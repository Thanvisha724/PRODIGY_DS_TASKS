import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("customer_data.csv")
df.columns = df.columns.str.strip()

print(df.head())
print(df.columns.tolist())

# Determine target column
target_col = "Purchased" if "Purchased" in df.columns else df.columns[-1]

# Encode any non-numeric (object) columns except the target
encoders = {}
for col in df.select_dtypes(include=[object]).columns.tolist():
    if col == target_col:
        # encode target separately below
        continue
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Encode target if it's non-numeric
if df[target_col].dtype == object:
    target_encoder = LabelEncoder()
    df[target_col] = target_encoder.fit_transform(df[target_col].astype(str))
else:
    target_encoder = None

# Features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print(classification_report(y_test, y_pred))

# Plot Decision Tree
plt.figure(figsize=(12, 7))
plot_tree(
    model,
    feature_names=X.columns.tolist(),
    class_names=[str(c) for c in model.classes_],
    filled=True
)

plt.title("Decision Tree Classifier")
plt.tight_layout()
plt.savefig("Decision_Tree_Output.png")
plt.show()