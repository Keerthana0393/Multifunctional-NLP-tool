import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import streamlit as st
from app.tasks.summarization import summarize_text
from app.tasks.sentiment import analyze_sentiment
from app.tasks.next_word import predict_next_word
from app.tasks.named_entity_recognition import recognize_entities
from app.tasks.question_answering import answer_question
from app.tasks.story_prediction import predict_story
from app.tasks.chatbot import chat_with_bot
from app.tasks.image_generation import generate_image

# App Title
st.title("Multifunctional NLP Tool")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Summarization", "Sentiment", "Next Word", "NER",
    "QA", "Story Prediction", "Chatbot", "Image Generation"
])

# Tab 1: Text Summarization
with tab1:
    st.header("Text Summarization")
    input_text = st.text_area("Enter text to summarize:")
    if st.button("Summarize"):
        if not input_text.strip():
            st.warning("Please enter some text to summarize!")
        else:
            with st.spinner("Summarizing..."):
                summary = summarize_text(input_text)
                st.write("Summary:", summary)


# Tab 2: Sentiment Analysis
with tab2:
    st.header("Sentiment Analysis")
    input_text = st.text_area("Enter text for sentiment analysis:")
    if st.button("Analyze Sentiment"):
        if not input_text.strip():
            st.warning("Please enter some text for sentiment analysis!")
        else:
            with st.spinner("Analyzing sentiment..."):
                sentiment = analyze_sentiment(input_text)
                st.write("Sentiment:", sentiment)


# Tab 3: Next Word Prediction
with tab3:
    st.header("Next Word Prediction")
    input_text = st.text_area("Enter text:")
    if st.button("Predict"):
        if not input_text.strip():
            st.warning("Please enter some text for prediction!")
        else:
            with st.spinner("Predicting next word..."):
                prediction = predict_next_word(input_text)
                st.write("Predicted Text:", prediction)

# Tab 4: Named Entity Recognition (NER)
with tab4:
    st.header("Named Entity Recognition (NER)")
    input_text = st.text_area("Enter text for NER:")
    if st.button("Identify Entities"):
        if not input_text.strip():
            st.warning("Please enter some text for NER!")
        else:
            with st.spinner("Identifying entities..."):
                entities = recognize_entities(input_text)
                st.write("Entities Identified:")
                for entity in entities:
                    st.write(f"{entity['entity_group']}: {entity['word']}")

# Tab 5: Question Answering
with tab5:
    st.header("Question Answering")
    context = st.text_area("Enter context:")
    question = st.text_input("Enter your question:")
    if st.button("Find Answer"):
        if not context.strip() or not question.strip():
            st.warning("Please provide both context and a question!")
        else:
            with st.spinner("Finding answer..."):
                answer = answer_question(context, question)
                st.write("Answer:", answer["answer"])

# Tab 6: Story Prediction
with tab6:
    st.header("Story Prediction")
    input_text = st.text_area("Enter text to predict story:")
    if st.button("Generate Story"):
        if not input_text.strip():
            st.warning("Please enter some text to generate a story!")
        else:
            with st.spinner("Generating story..."):
                story = predict_story(input_text)
                st.write("Generated Story:", story)

# Tab 7: Chatbot
with tab7:
    st.header("Chat with Bot")
    user_input = st.text_input("Enter your message:")
    if st.button("Send"):
        if not user_input.strip():
            st.warning("Please enter a message to chat!")
        else:
            with st.spinner("Chatting with bot..."):
                response, _ = chat_with_bot(user_input)
                st.write("Chatbot:", response)

# Tab 8: Image Generation
with tab8:
    st.header("Image Generation")
    prompt = st.text_input("Enter an image description:")
    if st.button("Generate Image"):
        if not prompt.strip():
            st.warning("Please enter a description for the image!")
        else:
            with st.spinner("Generating image..."):
                from app.tasks.image_generation import generate_image
                image = generate_image(prompt)
                st.image(image, caption="Generated Image")
