import pandas as pd
import joblib
import os
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ------------------------------------------------
# Download NLTK resources (first run only)
# ------------------------------------------------
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ------------------------------------------------
# Text preprocessing
# ------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ------------------------------------------------
# 1. Load dataset
# ------------------------------------------------
data = pd.read_csv("dataset/product_review.csv")

# ------------------------------------------------
# 🔹 DATASET ANALYSIS (ADDED — NO LOGIC CHANGED)
# ------------------------------------------------
print("\n📊 Dataset Analysis")
print("-" * 30)

total_samples = len(data)
sentiment_counts = data['Sentiment'].str.lower().value_counts()

print(sentiment_counts)
print("\nSummary:")
print(f"Total samples    : {total_samples}")
print(f"Positive samples : {sentiment_counts.get('positive', 0)}")
print(f"Negative samples : {sentiment_counts.get('negative', 0)}")
print(f"Neutral samples  : {sentiment_counts.get('neutral', 0)}")

# ------------------------------------------------
# 2. Combine Summary + Review
# ------------------------------------------------
data['text'] = data['Summary'].fillna('') + " " + data['Review'].fillna('')
data['Sentiment'] = data['Sentiment'].str.lower()

# ------------------------------------------------
# 3. Clean text
# ------------------------------------------------
data['text'] = data['text'].apply(clean_text)

# ------------------------------------------------
# 4. Balance dataset
# ------------------------------------------------
positive = data[data['Sentiment'] == "positive"]
negative = data[data['Sentiment'] == "negative"]
neutral  = data[data['Sentiment'] == "neutral"]

min_size = min(len(positive), len(negative), len(neutral))

positive = resample(positive, n_samples=min_size, random_state=42)
negative = resample(negative, n_samples=min_size, random_state=42)
neutral  = resample(neutral,  n_samples=min_size, random_state=42)

balanced_data = pd.concat([positive, negative, neutral])

X = balanced_data['text']
y = balanced_data['Sentiment']

# ------------------------------------------------
# 5. Train-test split
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------
# 6. Logistic Regression model
# ------------------------------------------------
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=40000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        max_df=0.9,
        min_df=2
    )),
    ('clf', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        n_jobs=-1
    ))
])

# ------------------------------------------------
# 7. Train model
# ------------------------------------------------
model.fit(X_train, y_train)

# ------------------------------------------------
# 8. Evaluate model
# ------------------------------------------------
y_pred = model.predict(X_test)

print("\n🔍 Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ------------------------------------------------
# 9. Save trained model
# ------------------------------------------------
os.makedirs("saved_model", exist_ok=True)
joblib.dump(model, "saved_model/nb_sentiment.pkl")

print("✅ Logistic Regression model trained and saved successfully!")
