# ============================================================
# PRODIGY INFOTECH - DATA SCIENCE INTERNSHIP
# Task 02 : Data Cleaning and Exploratory Data Analysis (EDA)
# Dataset : Titanic Dataset
# ============================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Style
plt.style.use("ggplot")
sns.set_style("whitegrid")

# ============================================================
# STEP 1 : Load Dataset
# ============================================================
df = pd.read_csv("titanic_sample.csv")

print("=" * 60)
print("TITANIC DATASET ANALYSIS")
print("=" * 60)

# ============================================================
# STEP 2 : Basic Information
# ============================================================
print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

# ============================================================
# STEP 3 : Missing Values Analysis
# ============================================================
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing Age values with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Fare values with mean
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ============================================================
# STEP 4 : Descriptive Statistics
# ============================================================
print("\nDescriptive Statistics:")
print(df.describe())

# ============================================================
# STEP 5 : Correlation Analysis
# ============================================================
numeric_df = df.select_dtypes(include=np.number)

print("\nCorrelation Matrix:")
print(numeric_df.corr())

# ============================================================
# STEP 6 : Data Visualization
# ============================================================

plt.figure(figsize=(18, 12))

# ---------------------------------------
# Graph 1 : Survival Count
# ---------------------------------------
plt.subplot(2, 3, 1)
sns.countplot(data=df, x="Survived", palette="Set2")
plt.title("Survival Count")
plt.xlabel("Survived")
plt.ylabel("Number of Passengers")

# ---------------------------------------
# Graph 2 : Survival by Gender
# ---------------------------------------
plt.subplot(2, 3, 2)
sns.countplot(data=df, x="Sex", hue="Survived",
              palette="Set1")
plt.title("Survival by Gender")

# ---------------------------------------
# Graph 3 : Age Distribution
# ---------------------------------------
plt.subplot(2, 3, 3)
sns.histplot(df["Age"],
             bins=10,
             kde=True,
             color="skyblue")
plt.title("Age Distribution")

# ---------------------------------------
# Graph 4 : Passenger Class Distribution
# ---------------------------------------
plt.subplot(2, 3, 4)
sns.countplot(data=df,
              x="Pclass",
              palette="viridis")
plt.title("Passenger Class Distribution")

# ---------------------------------------
# Graph 5 : Fare Distribution
# ---------------------------------------
plt.subplot(2, 3, 5)
sns.boxplot(data=df,
            x="Survived",
            y="Fare",
            palette="coolwarm")
plt.title("Fare Distribution by Survival")

# ---------------------------------------
# Graph 6 : Correlation Heatmap
# ---------------------------------------
plt.subplot(2, 3, 6)
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="Blues"
)
plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("EDA_Output.png", dpi=300)
plt.show()

# ============================================================
# STEP 7 : Insights
# ============================================================
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

print("1. Survival count shows how many passengers survived.")
print("2. Female passengers have higher survival rates.")
print("3. Most passengers belong to the age group 20-40.")
print("4. Passenger class influences survival chances.")
print("5. Fare paid also has some relationship with survival.")
print("6. Heatmap shows correlations between numerical features.")

print("\nEDA Completed Successfully.")
print("Output saved as EDA_Output.png")