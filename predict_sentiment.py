import joblib
import numpy as np

# Load trained Logistic Regression model
model = joblib.load("saved_model/nb_sentiment.pkl")

def confidence_to_stars(sentiment, confidence):
    """
    Convert confidence score to star rating
    """
    if sentiment == "positive":
        stars = 3 + confidence * 2       # 3 to 5 stars
    elif sentiment == "negative":
        stars = 1 + (1 - confidence)     # 1 to 2 stars
    else:
        stars = 2 + confidence           # 2 to 3 stars

    return round(stars * 2) / 2          # half-star support

def render_stars(stars):
    full = int(stars)
    half = 1 if stars - full >= 0.5 else 0
    empty = 5 - full - half
    return "⭐" * full + "⭐️" * half + "☆" * empty

print("🔍 Logistic Regression Sentiment Predictor")
print("Type 'exit' to stop\n")

while True:
    text = input("Enter product review: ")

    if text.lower() == "exit":
        print("Exiting...")
        break

    # Predict probabilities
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_

    max_index = np.argmax(probabilities)
    sentiment = classes[max_index]
    confidence = probabilities[max_index]

    stars_value = confidence_to_stars(sentiment, confidence)
    stars_display = render_stars(stars_value)

    print("\n--- Prediction Result ---")
    print(f"Sentiment  : {sentiment.upper()}")
    print(f"Confidence : {confidence * 100:.2f}%")
    print(f"Rating     : {stars_display} ({stars_value} stars)\n")
