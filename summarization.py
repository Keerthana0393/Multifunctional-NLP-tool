from transformers import pipeline
import streamlit as st

# Cache the Summarization pipeline
@st.cache_resource
def load_summarization_pipeline():
    return pipeline("summarization", model="t5-small")

def summarize_text(input_text):
    # Use the cached pipeline
    summarization_pipeline = load_summarization_pipeline()
    return summarization_pipeline(input_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
