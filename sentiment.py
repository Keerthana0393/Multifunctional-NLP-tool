from transformers import pipeline
import streamlit as st

# Cache the Sentiment Analysis pipeline
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis")

def analyze_sentiment(input_text):
    # Use the cached pipeline
    sentiment_pipeline = load_sentiment_pipeline()
    return sentiment_pipeline(input_text)[0]
