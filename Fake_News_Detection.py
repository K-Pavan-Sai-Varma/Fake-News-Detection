import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset from local CSV
df = pd.read_csv("fake_or_real_news.csv")

# Show dataset shape and head
print("Dataset shape:", df.shape)
print(df.head())

# Show label counts
print("\nLabel distribution:")
print(df['label'].value_counts())

# --- Show 10 Random Fake and Real News Articles ---
print("\n🔹 Random FAKE News Samples:")
print(df[df['label'] == 'FAKE']['text'].sample(10, random_state=42))

print("\n🔹 Random REAL News Samples:")
print(df[df['label'] == 'REAL']['text'].sample(10, random_state=42))

# --- Visualization: Make REAL bar smaller on purpose ---
fake_df = df[df['label'] == 'FAKE']
real_df = df[df['label'] == 'REAL'].sample(n=int(len(fake_df) * 0.5), random_state=42)  # 50% for visual effect

df_visual = pd.concat([fake_df, real_df])
label_counts = df_visual['label'].value_counts()

# Bar chart
plt.figure(figsize=(6, 4))
sns.barplot(x=label_counts.index, y=label_counts.values, palette=['red', 'green'])
plt.title("Fake vs Real News (Adjusted for Visualization)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# --- Model Training ---
labels = df['label']
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Model
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Prediction and Accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {round(score*100, 2)}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print("\nConfusion Matrix:\n", conf_matrix)
