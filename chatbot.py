import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st

@st.cache_resource
def load_chatbot_model():
    print("Loading chatbot model...")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    print("Chatbot model loaded.")
    return model, tokenizer

def chat_with_bot(input_text, history=[]):
    model, tokenizer = load_chatbot_model()
    new_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    history_ids = new_input_ids if not history else torch.cat([history, new_input_ids], dim=-1)
    bot_output = model.generate(history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(bot_output[:, history_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, bot_output
