import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
nltk.download('stopwords')
df = pd.read_csv(r"E:\Email classifier dataset\archive\spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df.head()
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text
df['cleaned_message'] = df['message'].apply(clean_text)
df.head()
X = df['cleaned_message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)
y_pred = model.predict(X_test_vectorized)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
def predict_spam(message):
    message_cleaned = clean_text(message)
    message_vectorized = vectorizer.transform([message_cleaned])
    prediction = model.predict(message_vectorized)
    return "Spam" if prediction == 1 else "Not Spam"
print(predict_spam("You have won a free iPhone! Click here to claim now."))
print(predict_spam("Hey, let’s meet up for lunch tomorrow."))
