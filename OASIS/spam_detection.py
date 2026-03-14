# Email Spam Detection Project

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load Dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print(df.head())

# 3. Convert labels to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 4. Define Features and Target
X = df['message']
y = df['label']

# 5. Convert text into numerical vectors
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(X)

# 6. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 7. Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# 8. Prediction
predictions = model.predict(X_test)

# 9. Evaluate Model
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# 10. Confusion Matrix
cm = confusion_matrix(y_test, predictions)

sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Spam Detection Confusion Matrix")
plt.show()

# 11. Test with New Email
sample_email = ["Congratulations! You won a free lottery ticket"]

sample_vector = vectorizer.transform(sample_email)

prediction = model.predict(sample_vector)

if prediction[0] == 1:
    print("This email is SPAM")
else:
    print("This email is NOT SPAM")
