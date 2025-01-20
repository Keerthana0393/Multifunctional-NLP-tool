from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_qa_pipeline():
    return pipeline("question-answering")

def answer_question(context, question):
    qa_pipeline = load_qa_pipeline()
    return qa_pipeline({"context": context, "question": question})
