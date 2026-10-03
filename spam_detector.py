"""
Beginner Spam Detector (emails / SMS)
-------------------------------------
Steps:
  1. Load labeled messages (spam or ham = not spam)
  2. Turn text into numbers (TF-IDF)
  3. Train a Naive Bayes classifier
  4. Check accuracy on messages the model has never seen
  5. Let you type your own messages to test

Install:  pip install scikit-learn pandas
Run:      python spam_detector.py

For better results, download the free "SMS Spam Collection" dataset
(Kaggle / UCI) and save it as spam.csv next to this file.
It has columns: v1 (ham/spam) and v2 (the message text).
"""

import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# ---------- 1. LOAD DATA ----------
def load_data():
    if os.path.exists("spam.csv"):
        print("Loading spam.csv ...")
        df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
        df.columns = ["label", "text"]
        return df

    print("spam.csv not found -> using a tiny built-in sample (add spam.csv for real accuracy).")
    spam = [
        "WINNER!! You have won a $1000 gift card. Click here to claim now",
        "Congratulations! You've been selected for a free iPhone. Reply YES",
        "URGENT: Your bank account is suspended. Verify your password at this link",
        "Free entry in our weekly prize draw! Text WIN to 80085",
        "Get cheap loans approved instantly, no credit check needed",
        "Claim your free vacation to the Bahamas. Limited time offer!!!",
        "You have been pre-approved for a credit card. Apply now and get cash back",
        "Lose 10kg in 7 days with this miracle pill. Order today",
        "Your parcel is held. Pay a small fee at this link to release it",
        "Earn $5000 a week working from home. No experience needed",
        "Final notice: you owe unpaid taxes. Pay immediately to avoid arrest",
        "Hot singles in your area are waiting. Click to chat free",
        "Exclusive deal! 90% off luxury watches. Buy now before it ends",
        "Your account has won a lottery prize of 500,000 dollars. Send details",
        "Act now! Free trial, cancel anytime, just enter your card number",
        "Cheap meds online without prescription. Fast delivery, order now",
    ]
    ham = [
        "Hey, are we still meeting for lunch tomorrow?",
        "Can you send me the notes from today's class?",
        "Don't forget to pick up milk on your way home",
        "The meeting has been moved to 3 pm in the main conference room",
        "Happy birthday! Hope you have a wonderful day",
        "I'll call you after work, is that okay?",
        "Please find the project report attached. Let me know your thoughts",
        "Dinner at our place on Saturday? Mom is making pasta",
        "Thanks for helping me with the assignment yesterday",
        "Reminder: dentist appointment on Monday at 10 am",
        "Running late, will be there in 15 minutes",
        "Did you watch the match last night? It was amazing",
        "Can we reschedule our call to Thursday afternoon?",
        "Here are the photos from our trip, check them out",
        "I finished the code review, just a few small comments",
        "Good luck with your exam today, you'll do great",
    ]
    return pd.DataFrame(
        {"label": ["spam"] * len(spam) + ["ham"] * len(ham), "text": spam + ham}
    )


df = load_data()
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
print(f"Loaded {len(df)} messages ({(df['label_num'] == 1).sum()} spam)\n")

# ---------- 2. SPLIT INTO TRAIN / TEST ----------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label_num"], test_size=0.25, random_state=42, stratify=df["label_num"]
)

# ---------- 3. BUILD + TRAIN THE MODEL ----------
model = Pipeline(
    [
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
        ("classifier", MultinomialNB()),
    ]
)
model.fit(X_train, y_train)

# ---------- 4. EVALUATE ----------
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print(classification_report(y_test, predictions, target_names=["ham", "spam"], zero_division=0))


# ---------- 5. TRY IT YOURSELF ----------
def check_message(text):
    prob_spam = model.predict_proba([text])[0][1]
    verdict = "SPAM" if prob_spam >= 0.5 else "NOT SPAM"
    print(f"  -> {verdict} (spam probability: {prob_spam:.0%})\n")


if __name__ == "__main__":
    print("Type a message to check it (or 'quit' to exit).")
    while True:
        try:
            msg = input("Message: ").strip()
        except EOFError:
            break
        if msg.lower() in {"quit", "exit", "q"}:
            break
        if msg:
            check_message(msg)
