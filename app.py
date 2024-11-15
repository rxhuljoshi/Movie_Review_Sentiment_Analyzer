import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Load the sentiment analysis model
model = tf.keras.models.load_model("sentiment_analysis_model.h5")

# Function to predict sentiment
def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    return sentiment, confidence

# Streamlit app
st.title("Sentiment Analysis App")
st.write("This app uses a pre-trained LSTM model to classify the sentiment of a movie review as Positive or Negative.")

# Input from user
user_input = st.text_area("Enter a movie review below:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment, confidence = predict_sentiment(user_input)
        st.subheader("Sentiment Analysis Result")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")
    else:
        st.warning("Please enter a review to analyze.")

# Footer
st.markdown("---")
st.write("Developed using Streamlit, TensorFlow, and Keras")