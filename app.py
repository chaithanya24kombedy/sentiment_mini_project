from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load trained pipeline model
model = joblib.load("saved_model/nb_sentiment.pkl")

def confidence_to_stars(confidence):
    if confidence >= 0.8:
        return "⭐⭐⭐⭐⭐"
    elif confidence >= 0.6:
        return "⭐⭐⭐⭐"
    elif confidence >= 0.4:
        return "⭐⭐⭐"
    elif confidence >= 0.2:
        return "⭐⭐"
    else:
        return "⭐"

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        return redirect(url_for("review"))
    return render_template("login.html")

@app.route("/review", methods=["GET", "POST"])
def review():
    if request.method == "POST":
        # ✅ FIX HERE
        text = request.form["review"]

        probs = model.predict_proba([text])[0]
        classes = model.classes_
        idx = np.argmax(probs)

        sentiment = classes[idx].upper()
        confidence = probs[idx]
        stars = confidence_to_stars(confidence)

        return render_template(
            "result.html",
            sentiment=sentiment,
            confidence=f"{confidence*100:.2f}%",
            stars=stars
        )

    return render_template("review.html")

if __name__ == "__main__":
    app.run(debug=True)
