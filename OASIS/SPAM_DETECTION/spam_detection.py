
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("spam.csv", encoding="latin-1")

data = data[['v1', 'v2']]
data.columns = ['label', 'message']


data['label'] = data['label'].map({'ham': 0, 'spam': 1})


vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X = vectorizer.fit_transform(data['message'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

def predict_email(email_text):
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)
    return "SPAM 🚫" if prediction[0] == 1 else "NOT SPAM ✅"

test_email_1 = "Congratulations! You have won a free prize. Click here to claim now."
test_email_2 = "Hello, let's meet tomorrow to discuss the assignment."

print("\nCustom Email Tests:")
print("Email 1:", predict_email(test_email_1))
print("Email 2:", predict_email(test_email_2))

