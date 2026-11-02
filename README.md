# Spam Email Classifier

A beginner-friendly machine learning project that predicts whether a message is **Spam** or **Not Spam**.

## Technologies used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes
- Streamlit
- SQLite

## How it works

1. The project starts with a small labelled example dataset containing spam and normal messages.
2. The dataset is split into training and testing data.
3. TF-IDF converts message text into numerical features.
4. Multinomial Naive Bayes learns from the training messages.
5. The model predicts whether a new message is Spam or Not Spam.
6. SQLite stores the message, prediction and date/time.
7. Streamlit provides a simple interface to use the classifier.

## Project structure

```text
Spam-Email-Classifier/
├── app.py
├── requirements.txt
└── README.md
```

The SQLite database (`spam_history.db`) is created automatically when the app runs. It does not need to be uploaded to GitHub.

## Run the project locally

### 1. Download or clone this repository

```bash
git clone YOUR_REPOSITORY_URL
cd Spam-Email-Classifier
```

### 2. Install the required libraries

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The app will open in your browser.

## Important note

This is an educational demo trained on a small example dataset included in `app.py`. Its accuracy is only a basic demonstration and it is not suitable for filtering real inboxes. For a stronger portfolio version, train and evaluate it on a larger public spam dataset and test it on messages that were not used during training.

## What I learned

- Text preprocessing and text classification
- Converting text into numerical features using TF-IDF
- Training a Naive Bayes classifier
- Splitting data into training and testing sets
- Evaluating a model using accuracy
- Building a small Streamlit application
- Saving application data using SQLite
