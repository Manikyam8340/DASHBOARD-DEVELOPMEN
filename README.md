import pandas as pd
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Sample dataset (Replace this with actual text data)
data = {'Review': [
    "The product is amazing! I love it.",
    "Worst experience ever. Completely disappointed.",
    "It's okay, not the best but not the worst either.",
    "Absolutely fantastic! Highly recommended!",
    "I hate it. Waste of money."
]}

df = pd.DataFrame(data)

# Function to get sentiment using TextBlob
def get_sentiment_textblob(text):
    return TextBlob(text).sentiment.polarity

df['Sentiment_TextBlob'] = df['Review'].apply(get_sentiment_textblob)

# Function to get sentiment using VADER
sia = SentimentIntensityAnalyzer()
def get_sentiment_vader(text):
    return sia.polarity_scores(text)['compound']

df['Sentiment_VADER'] = df['Review'].apply(get_sentiment_vader)

# Classify Sentiments
conditions = [
    (df['Sentiment_VADER'] > 0),
    (df['Sentiment_VADER'] < 0),
    (df['Sentiment_VADER'] == 0)
]
choices = ['Positive', 'Negative', 'Neutral']
df['Sentiment_Label'] = pd.np.select(conditions, choices)

# Display results
print(df)
