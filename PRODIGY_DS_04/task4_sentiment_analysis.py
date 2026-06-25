"""
Sentiment Analysis on Customer Reviews
---------------------------------------
This script trains a simple Naive Bayes classifier to predict whether a
review is Positive, Negative (or Neutral, depending on your dataset)
based purely on the words used in the text.

The pipeline is:
    1. Load the data
    2. Clean it up a little
    3. Turn the text into numbers (Bag of Words)
    4. Split into train/test sets
    5. Train a Multinomial Naive Bayes model
    6. Evaluate it
    7. Visualize a few things so we can actually *see* what's going on

Nothing fancy here, just a clear, step-by-step walkthrough.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ------------------------------------------------------------------
# Step 1: Load the dataset
# ------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Read the CSV and do a quick sanity check before moving on."""
    print(f"Loading data from '{path}'...")
    data = pd.read_csv(path)

    print(f"  -> Loaded {len(data)} rows.")
    print(f"  -> Columns found: {list(data.columns)}")

    # Drop any rows that are missing a review or a sentiment label.
    # No point training on empty text.
    before = len(data)
    data = data.dropna(subset=["Review", "Sentiment"])
    after = len(data)
    if before != after:
        print(f"  -> Dropped {before - after} rows with missing values.")

    return data


# ------------------------------------------------------------------
# Step 2: A little bit of text cleanup
# ------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Basic cleaning: lowercase everything and strip extra whitespace.
    CountVectorizer already handles punctuation/tokenizing reasonably
    well, but lowercasing first avoids treating "Good" and "good" as
    two different words.
    """
    text = str(text).lower().strip()
    return text


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning review text...")
    data["Review_Clean"] = data["Review"].apply(clean_text)
    return data


# ------------------------------------------------------------------
# Step 3 & 4: Vectorize the text and split into train/test sets
# ------------------------------------------------------------------
def vectorize_and_split(data: pd.DataFrame, test_size: float = 0.3, seed: int = 42):
    print("Converting text into numerical features (Bag of Words)...")
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(data["Review_Clean"])
    y = data["Sentiment"]

    print(f"  -> Vocabulary size: {len(vectorizer.vocabulary_)} unique words.")

    print(f"Splitting data into train/test sets ({int((1-test_size)*100)}/{int(test_size*100)})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    return X_train, X_test, y_train, y_test, vectorizer


# ------------------------------------------------------------------
# Step 5: Train the model
# ------------------------------------------------------------------
def train_model(X_train, y_train) -> MultinomialNB:
    print("Training the Naive Bayes classifier...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    print("  -> Done training.")
    return model


# ------------------------------------------------------------------
# Step 6: Evaluate the model
# ------------------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model performance...\n")
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions) * 100
    print(f"Accuracy: {round(accuracy, 2)}%\n")

    print("Classification Report:")
    print(classification_report(y_test, predictions))

    return predictions


# ------------------------------------------------------------------
# Step 7: Visualizations
# ------------------------------------------------------------------
def plot_sentiment_distribution(data: pd.DataFrame, output_path: str = "Sentiment_Output.png"):
    """Bar chart showing how many reviews fall into each sentiment class."""
    print(f"Saving sentiment distribution chart to '{output_path}'...")
    sentiment_counts = data["Sentiment"].value_counts()

    plt.figure(figsize=(6, 4))
    sentiment_counts.plot(kind="bar", color="#4C72B0", edgecolor="black")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_confusion_matrix(y_test, predictions, model, output_path: str = "Confusion_Matrix.png"):
    """Heatmap of actual vs predicted sentiment, so we can see *where* the model messes up."""
    print(f"Saving confusion matrix to '{output_path}'...")
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, predictions, labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Sentiment")
    plt.ylabel("Actual Sentiment")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def plot_top_words(vectorizer, model, n: int = 15, output_path: str = "Top_Words.png"):
    """
    Quick peek at which words the model leans on most heavily for its
    *first* class label, just for curiosity's sake.
    """
    print(f"Saving top influential words chart to '{output_path}'...")
    feature_names = vectorizer.get_feature_names_out()
    class_index = 0  # first class in model.classes_
    log_probs = model.feature_log_prob_[class_index]

    top_indices = log_probs.argsort()[-n:]
    top_words = [feature_names[i] for i in top_indices]
    top_scores = [log_probs[i] for i in top_indices]

    plt.figure(figsize=(7, 5))
    plt.barh(top_words, top_scores, color="#55A868")
    plt.title(f"Top {n} Words Influencing '{model.classes_[class_index]}' Predictions")
    plt.xlabel("Log Probability")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


# ------------------------------------------------------------------
# Putting it all together
# ------------------------------------------------------------------
def main():
    data_path = "sentiment_data.csv"

    data = load_data(data_path)
    data = preprocess(data)

    X_train, X_test, y_train, y_test, vectorizer = vectorize_and_split(data)

    model = train_model(X_train, y_train)
    predictions = evaluate_model(model, X_test, y_test)

    # A few visuals to make the results easier to digest
    plot_sentiment_distribution(data)
    plot_confusion_matrix(y_test, predictions, model)
    plot_top_words(vectorizer, model)

    print("\nAll done! Check the saved PNG files for the visualizations.")


if __name__ == "__main__":
    main()