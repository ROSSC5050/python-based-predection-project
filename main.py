import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the dataset (you might have to adjust the file path)
data = pd.read_csv("fake_or_real_news.csv")

# Data preprocessing: Drop any rows with missing values
data.dropna(inplace=True)

# Feature extraction using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_tfidf = vectorizer.fit_transform(data["text"])

# Train the classifier (Multinomial Naive Bayes in this example)
classifier = MultinomialNB()
classifier.fit(X_tfidf, data["label"])


def predict_fake_news(news_text):
    # Convert the user input to TF-IDF vector using the same vectorizer
    news_tfidf = vectorizer.transform([news_text])

    # Make prediction using the trained classifier
    prediction = classifier.predict(news_tfidf)[0]

    return "fake" if prediction == "FAKE" else "real"


# Example usage
user_input = input("Enter a news text: ")
prediction = predict_fake_news(user_input)
print(f"The news is predicted to be {prediction}.")
