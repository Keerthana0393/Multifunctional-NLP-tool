import torch
print(torch.cuda.is_available())  # Should return True if GPU is enabled
import streamlit as st #Ensure Streamlit is imported

from diffusers import StableDiffusionPipeline
@st.cache_resource
def load_image_model():
    return StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float32
    ).to("cpu")


def generate_image(prompt):
    model = load_image_model()
    return model(prompt, num_inference_steps=20).images[0]






