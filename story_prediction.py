from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st

@st.cache_resource
def load_story_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

def predict_story(input_text):
    model, tokenizer = load_story_model()
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    output = model.generate(
        input_ids,
        max_length=100,  # Adjust max story length
        repetition_penalty=2.0,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)
