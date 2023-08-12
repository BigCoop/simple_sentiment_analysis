import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# Load the dataset directly from the local machine
data = pd.read_csv('Tweets.csv')


# Dropping columns not essential for sentiment analysis
data = data[['text', 'airline_sentiment']]

# Text cleaning (removing special characters, converting to lowercase)
data['clean_text'] = data['text'].str.replace("[^a-zA-Z#]", " ").str.lower()

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=2500, stop_words='english')
X = vectorizer.fit_transform(data['clean_text'])

y = data['airline_sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

predictions = pd.DataFrame({
    'Original_Sentiment': y_test,
    'Predicted_Sentiment': y_pred
})

# Save predictions to a new CSV file
predictions.to_csv('predictions.csv', index=False)

print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

