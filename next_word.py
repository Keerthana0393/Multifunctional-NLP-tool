from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st

# Cache the GPT-2 model and tokenizer
@st.cache_resource
def load_next_word_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer

def predict_next_word(input_text):
    model, tokenizer = load_next_word_model()
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    output = model.generate(
        input_ids,
        max_length=50,  # Limit the output length
        repetition_penalty=2.0,  # Penalize repeated phrases
        temperature=0.7,  # Introduce randomness
        top_k=50,         # Consider top 50 options
        top_p=0.95        # Nucleus sampling
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)
