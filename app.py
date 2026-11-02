# Spam Email Classifier
# A beginner-friendly ML project using TF-IDF and Multinomial Naive Bayes.

import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# 1. Small example dataset
# In a larger project, this would come from a CSV file.
# -------------------------------------------------
emails = [
    ("Hey, are we still meeting for lunch today?", "Not Spam"),
    ("Please find the meeting notes attached.", "Not Spam"),
    ("Can you call me when you reach home?", "Not Spam"),
    ("Your order has been shipped and is on the way.", "Not Spam"),
    ("Reminder: your appointment is tomorrow at 10 AM.", "Not Spam"),
    ("Thanks for helping me with the assignment.", "Not Spam"),
    ("The project presentation is scheduled for Monday.", "Not Spam"),
    ("Are you free to discuss the report this evening?", "Not Spam"),
    ("I will send you the photos after class.", "Not Spam"),
    ("Your electricity bill is ready to view.", "Not Spam"),
    ("Please review the document before the deadline.", "Not Spam"),
    ("Happy birthday! Have a wonderful day.", "Not Spam"),
    ("The train is delayed by fifteen minutes.", "Not Spam"),
    ("Let's catch up over coffee this weekend.", "Not Spam"),
    ("Your package was delivered successfully.", "Not Spam"),
    ("I have shared the class notes with you.", "Not Spam"),
    ("The team meeting starts at 3 PM.", "Not Spam"),
    ("Could you please send me the invoice?", "Not Spam"),
    ("Dinner is ready, come downstairs.", "Not Spam"),
    ("Your password was changed successfully.", "Not Spam"),
    ("We are planning a trip next month.", "Not Spam"),
    ("The library will remain closed on Sunday.", "Not Spam"),
    ("I received your email, thank you.", "Not Spam"),
    ("Please confirm your attendance for the event.", "Not Spam"),
    ("Your exam timetable has been published.", "Not Spam"),

    ("Congratulations! You have won a free prize. Claim now!", "Spam"),
    ("URGENT! You have been selected for a cash reward.", "Spam"),
    ("Win a brand new phone. Click here to claim your gift.", "Spam"),
    ("You are a lucky winner. Send your bank details now.", "Spam"),
    ("Get a loan instantly with no paperwork. Apply today!", "Spam"),
    ("Limited offer! Buy now and get 90 percent discount.", "Spam"),
    ("You have won a lottery. Call this number immediately.", "Spam"),
    ("Claim your free gift card before the offer expires.", "Spam"),
    ("Earn money from home. Guaranteed income every day.", "Spam"),
    ("Your account is blocked. Click this link to verify.", "Spam"),
    ("Exclusive deal! Get cheap medicines without prescription.", "Spam"),
    ("You have been chosen for a free vacation. Claim now.", "Spam"),
    ("Act now! This investment will double your money.", "Spam"),
    ("Free entry in our prize draw. Text WIN to enter.", "Spam"),
    ("Congratulations, you won cash. Reply with your details.", "Spam"),
    ("Lowest price ever! Buy now before stock runs out.", "Spam"),
    ("Urgent notice: verify your account to avoid suspension.", "Spam"),
    ("Get rich quickly with this secret online opportunity.", "Spam"),
    ("You are eligible for a free reward. Click to redeem.", "Spam"),
    ("Winner! Collect your cash prize by calling now.", "Spam"),
    ("Special promotion: claim your free shopping voucher.", "Spam"),
    ("Make thousands weekly with no experience required.", "Spam"),
    ("Your mobile number has won a huge cash prize.", "Spam"),
    ("Act fast to receive your exclusive bonus today.", "Spam"),
    ("Free vacation waiting for you. Send details to claim.", "Spam"),
]

df = pd.DataFrame(emails, columns=["message", "label"])

# -------------------------------------------------
# 2. Train the machine learning model
# TF-IDF converts text into numerical features.
# Naive Bayes learns patterns from labelled examples.
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# -------------------------------------------------
# 3. Create a SQLite database for prediction history
# -------------------------------------------------
def save_prediction(message, prediction):
    connection = sqlite3.connect("spam_history.db")
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            prediction TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute(
        "INSERT INTO history (message, prediction, created_at) VALUES (?, ?, ?)",
        (message, prediction, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )

    connection.commit()
    connection.close()


def get_history():
    connection = sqlite3.connect("spam_history.db")
    try:
        # Create the table on the first run, even before any prediction is saved.
        connection.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT NOT NULL,
                prediction TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        connection.commit()
        history = pd.read_sql_query(
            "SELECT message, prediction, created_at FROM history ORDER BY id DESC",
            connection
        )
    finally:
        connection.close()
    return history


# -------------------------------------------------
# 4. Streamlit user interface
# -------------------------------------------------
st.set_page_config(page_title="Spam Email Classifier", page_icon="📧")
st.title("📧 Spam Email Classifier")
st.write("Enter an email or message below to check whether it looks like spam.")

st.caption("Demo project trained on a small example dataset. Predictions are not guaranteed.")

message = st.text_area(
    "Enter your email message:",
    placeholder="Example: Congratulations! You have won a free prize..."
)

if st.button("Classify Message", type="primary"):
    if message.strip():
        prediction = model.predict([message])[0]
        probability = model.predict_proba([message]).max()

        save_prediction(message, prediction)

        if prediction == "Spam":
            st.error(f"🚨 This message looks like SPAM (confidence: {probability:.1%})")
        else:
            st.success(f"✅ This message looks like NOT SPAM (confidence: {probability:.1%})")
    else:
        st.warning("Please enter a message first.")

st.divider()

st.subheader("Model Information")
col1, col2 = st.columns(2)
col1.metric("Test accuracy", f"{accuracy:.1%}")
col2.metric("Training examples", len(X_train))

st.caption("Accuracy is measured on a small held-out sample and should not be treated as real-world performance.")

st.subheader("Prediction History")
history = get_history()

if history.empty:
    st.info("No predictions yet. Classify a message to create your first history record.")
else:
    st.dataframe(history, use_container_width=True, hide_index=True)
