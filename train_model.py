# Step 1: Load and label the datasets
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
# Load the datasets
fake_news = pd.read_csv("Fake.csv")
true_news = pd.read_csv("True.csv")

# Add label columns: 0 for fake news, 1 for true news
fake_news['label'] = 0
true_news['label'] = 1

print("Fake news samples:", len(fake_news))
print("True news samples:", len(true_news))

# Step 2: Combine and preprocess the data
# Combine the datasets
data = pd.concat([fake_news, true_news], ignore_index=True)

# Shuffle the combined dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Extract the text and labels
texts = data['text']
labels = data['label']

print("Total samples:", len(data))

# Step 3: Train the TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(texts)

print("Vectorizer trained. Number of features:", X.shape[1])

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Step 5: Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Logistic Regression model trained.")

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save the vectorizer and model
joblib.dump(vectorizer, "vectorizer.jb")
joblib.dump(model, "lr_model.jb")

print("Vectorizer and model saved as joblib files.")

# Step 8: Load the saved model and make predictions
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# Define a function for making predictions
def predict_news(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return "True News" if prediction[0] == 1 else "Fake News"

# Example usage
sample_text = "This is an example news article text."
print("Prediction:", predict_news(sample_text))
