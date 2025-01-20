from transformers import pipeline

# Pre-download the models
def download_models():
    print("Downloading Sentiment Analysis model...")
    pipeline("sentiment-analysis")  # Download sentiment-analysis model
    print("Downloading Text Summarization model...")
    pipeline("summarization", model="facebook/bart-large-cnn")  # Download summarization model
    print("Downloading Next Word Prediction model...")
    pipeline("text-generation", model="gpt2")  # Download next-word prediction model
    print("All models downloaded and cached locally!")

if __name__ == "_main_":
    download_models()