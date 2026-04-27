Sentiment Analysis Web App

A simple and interactive Sentiment Analysis Web Application built using Flask, Machine Learning, and Natural Language Processing (NLP).
This project predicts whether a product review is Positive, Negative, or Neutral along with a confidence score and star rating.

🚀 Features

🔐 User Login Page
📝 Review Input Page
🤖 Sentiment Prediction (Positive / Negative / Neutral)
📊 Confidence Score Display
⭐ Star Rating Visualization
💡 Clean and Simple UI
⚡ Fast predictions using ML model

🛠️ Tech Stack

Frontend: HTML, CSS
Backend: Python, Flask
Machine Learning:
Logistic Regression
TF-IDF Vectorization

Libraries Used:
Flask
Scikit-learn
Pandas
NumPy
Joblib / Pickle

📂 Project Structure

sentiment_project/
│
├── app.py
├── train_nb.py
├── saved_model/
│   └── nb_sentiment.pkl
│
├── dataset/
│   └── product_review.csv
│
├── templates/
│   ├── login.html
│   ├── review.html
│   └── result.html
│
├── static/
│   └── style.css
│
└── README.md


⚙️ Installation & Setup

1️⃣ Clone the repository
git clone https://github.com/your-username/sentiment-analysis-app.git
cd sentiment-analysis-app
2️⃣ Install dependencies
python -m pip install flask pandas numpy scikit-learn joblib
3️⃣ Train the model
python train_nb.py
4️⃣ Run the application
python app.py
5️⃣ Open in browser
http://127.0.0.1:5000

🧠 How It Works

User logs in
Enters a product review
Text is converted using TF-IDF Vectorizer
Model predicts sentiment using Machine Learning
Output displayed with:
Sentiment (Positive / Neutral / Negative)
Confidence Score
Star Rating ⭐
