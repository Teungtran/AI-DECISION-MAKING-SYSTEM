import torch
import streamlit as st
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import opinion_lexicon
nltk.download('opinion_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import joblib as jb
from tensorflow.keras.preprocessing.sequence import pad_sequences
# varibles
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]') # identify comma
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]') # identify special symbol
NEGATIVE_WORDS = set(opinion_lexicon.negative())
POSITIVE_WORDS = set(opinion_lexicon.positive())
MODEL_SENTIMENT = os.getenv("MODEL_SENTIMENT")
MODEL_RATING = os.getenv("MODEL_RATING")
TOKENIZER_SENTIMENT = os.getenv("TOKENIZER_SENTIMENT")
TOKENIZER_RATING = os.getenv("TOKENIZER_RATING")



model_sentiment = jb.load(MODEL_SENTIMENT)
model_rating = jb.load(MODEL_RATING)
tokenizer_rating = jb.load(TOKENIZER_RATING)
tokenizer_sentiment = jb.load(TOKENIZER_SENTIMENT)
#TEXT CLEANING FUNCTIONS

def expand_contractions(text: str) -> str:
    contractions = {
        "don't": 'do not', "can't": 'cannot', "won't": 'will not', "isn't": 'is not',
        "aren't": 'are not', "wasn't": 'was not', "weren't": 'were not', "hasn't": 'has not',
        "haven't": 'have not', "hadn't": 'had not', "doesn't": 'does not', "didn't": 'did not',
        "shouldn't": 'should not', "couldn't": 'could not', "wouldn't": 'would not', "mightn't": 'might not',
        "mustn't": 'must not', "needn't": 'need not',"not":"not"
    }
    
    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)
    return text

def preprocess_sentiment_text(review):
    """Preprocess text for sentiment analysis."""
    # Convert to string and lowercase
    review = str(review).lower()
    
    # Expand contractions
    review = expand_contractions(review)
    
    # Remove special characters and replace with space
    review = REPLACE_BY_SPACE_RE.sub(' ', review)
    review = BAD_SYMBOLS_RE.sub('', review)
    
    # Remove URLs, mentions, hashtags, and HTML tags
    review = re.sub(r'https*\S+', ' ', review)  # Remove URLs
    review = re.sub(r'[@#]\S+', ' ', review)    # Remove mentions and hashtags
    review = re.sub('<.*?>', '', review)        # Remove HTML tags
    
    # Tokenize the text
    tokens = word_tokenize(review)
    
    # Remove stopwords but keep sentiment-related words
    stop_words = set(stopwords.words('english')) - NEGATIVE_WORDS - POSITIVE_WORDS
    tokens = [token for token in tokens if token not in stop_words]
    
    # POS tagging
    tagged_tokens = pos_tag(tokens)
    
    # Lemmatization based on POS tags
    lemmatizer = WordNetLemmatizer()
    processed_tokens = []
    
    for word, tag in tagged_tokens:
        if tag.startswith('JJ'):
            pos = 'a'  # adjective
        elif tag.startswith('VB'):
            pos = 'v'  # verb
        elif tag.startswith('NN'):
            pos = 'n'  # noun
        elif tag.startswith('RB'):
            pos = 'r'  # adverb
        else:
            pos = 'n'  # default to noun
            
        lemmatized = lemmatizer.lemmatize(word, pos=pos)
        processed_tokens.append((lemmatized, tag))
    
    # Capitalize words based on requirements
    final_tokens = []
    for word, tag in processed_tokens:
        # Capitalize if it's a sentiment word (positive or negative)
        if word in NEGATIVE_WORDS or word in POSITIVE_WORDS:
            final_tokens.append(word.upper())
        # Capitalize adjectives, adverbs, and certain verb forms
        elif tag in ('JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VBD', 'VBG'):
            final_tokens.append(word.upper())
        else:
            final_tokens.append(word)
    
    # Join tokens back into a string
    processed_review = ' '.join(final_tokens)
    
    return processed_review

def preprocess_rating_text(review):
    # remove noise data
    review = str(review).lower()
    review = expand_contractions(review)
    review = REPLACE_BY_SPACE_RE.sub(' ', review)
    review = BAD_SYMBOLS_RE.sub('', review)
    review = re.sub(r'https*\S+', ' ', review)  # Remove URLs
    review = re.sub(r'[@#]\S+', ' ', review)   # Remove mentions and hashtags
    review = re.sub('<.*?>', '', review)       # Remove HTML tags
    tokenizer = word_tokenize(review)
    stop_words = set(stopwords.words('english')) - NEGATIVE_WORDS - POSITIVE_WORDS
    tokens = [token for token in tokenizer if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tagged_tokens = pos_tag(tokens)
    # Extract key words
    processed_tokens = [
        word.lower() if tag in ('JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VBD', 'VBG', 'RBS') else word
        for word, tag in tagged_tokens
        if not (tag == ('VBZ','VBG'))
    ]
    
    processed_review = ' '.join(processed_tokens)
    return processed_review
# MODEL RELATED FUNTION
def calculate_rating(ratings):
    return [min(5.0, max(0.5, round(r[0] * 10) / 2)) for r in ratings]
def map_to_binary_label(predicted_id):
    if predicted_id <= 2:
        return "NEGATIVE"
    elif predicted_id == 3:
        return "NEUTRAL"
    else: 
        return "POSITIVE"
    
# PREDICTION RELATED FUNCTIONS
def predict_sentiment(text):
    token = tokenizer_sentiment.tokenize(text)
    tokens = tokenizer_sentiment.convert_tokens_to_ids(token)
    tokens = torch.tensor(tokens).unsqueeze(0)
    output = model_sentiment(tokens)
    predicted_id = torch.argmax(output.logits, dim=1).item()
    predicted_label = map_to_binary_label(predicted_id)
    return predicted_label
def analyze_single_text(text):
    results = predict_sentiment(text)
    sequences = tokenizer_rating.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=200)
    ratings = model_rating.predict(padded_sequences)
    processed_ratings = calculate_rating(ratings)
    
    return results, processed_ratings
def analyze_customer_reviews(df):
    if df is None or df.empty:
        raise ValueError("No data available for analysis")
    df.columns = df.columns.map(str.strip)
    df.columns = df.columns.map(str.lower)
    df['processed_review'] = df['review'].apply(lambda x: preprocess_sentiment_text(x))
    df['sentiment'] = df['processed_review'].apply(
        lambda x: predict_sentiment(x)
    )
    df['processed_rating_review'] = df['review'].apply(lambda x: preprocess_rating_text(x))
    sequences = tokenizer_rating.texts_to_sequences(df['processed_rating_review'].tolist())
    padded_sequences = pad_sequences(sequences, maxlen=200)
    ratings = model_rating.predict(padded_sequences)
    df['rating'] = calculate_rating(ratings)
    return df

# VISUALIZING
def word_cloud(df_sentiment_report):
    stopwords = set(STOPWORDS)
    unique_sentiments = sorted(df_sentiment_report['sentiment'].unique())
    fig_cloud = plt.figure(figsize=(36, 12))
    for index, sentiment in enumerate(unique_sentiments, start=1):
        plt.subplot(1, 3, index)
        df_filtered = df_sentiment_report[df_sentiment_report['sentiment'] == sentiment]
        data = " ".join(df_filtered['review'].astype(str))
        wordcloud = WordCloud(
            background_color='white',
            stopwords=stopwords,
            max_words=500,
            max_font_size=60,
            scale=6,
            width=1000,
            height=600).generate(data)
        
        plt.xticks([])
        plt.yticks([])
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.title(f"Sentiment: {sentiment.upper()}", fontsize=24, pad=20)
    plt.tight_layout()
    st.pyplot(fig_cloud)
def sentiment_distribution(df_sentiment_report):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sentiment_colors = ['#2ecc71', '#3498db', '#e74c3c']
    rating_colors = plt.cm.YlOrRd(np.linspace(0.3, 0.8, 5))
    # Sentiment Distribution 
    sentiment_counts = df_sentiment_report['sentiment'].value_counts()
    patches, texts, autotexts = ax1.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        colors=sentiment_colors,
        startangle=90,
        pctdistance=0.85,
        wedgeprops={'width': 0.7}  
    )
    plt.setp(autotexts, size=9, weight="bold")
    plt.setp(texts, size=10)
    ax1.set_title("Sentiment Distribution", pad=20, fontsize=12, fontweight='bold')
    # Rating Distribution 
    ax2 = plt.gca()
    bars = df_sentiment_report['rating'].value_counts().sort_index().plot(
        kind='bar',
        color=rating_colors,
        edgecolor='black',
        ax=ax2
    )
    for rect in bars.patches:
        height = rect.get_height()
        ax2.text(
            rect.get_x() + rect.get_width()/2.,
            height,
            f'{int(height):,}',
            ha='center',
            va='bottom'
        )
    plt.title('Ratings Distribution')
    plt.xlabel('Ratings')
    plt.ylabel('Count')
    plt.tight_layout(pad=3.0)
    st.pyplot(fig)