from transformers import pipeline
import streamlit as st

# Cache the NER pipeline
@st.cache_resource
def load_ner_pipeline():
    return pipeline("ner", grouped_entities=True)

def recognize_entities(input_text):
    # Use the cached pipeline
    ner_pipeline = load_ner_pipeline()
    return ner_pipeline(input_text)
