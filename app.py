import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import tempfile
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("emotion_recognition_model.keras")

model = load_model()

# -----------------------------
# Load Label Encoder
# -----------------------------
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# -----------------------------
# MFCC Feature Extraction
# -----------------------------
def extract_mfcc(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        duration=3,
        offset=0.5
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc


# -----------------------------
# Prediction Function
# -----------------------------
def predict_emotion(file_path):

    features = extract_mfcc(file_path)

    features = np.array(features)
    features = features.reshape(1, 40, 1)

    prediction = model.predict(
        features,
        verbose=0
    )

    predicted_class = np.argmax(prediction)

    emotion = encoder.inverse_transform(
        [predicted_class]
    )[0]

    confidence = np.max(prediction) * 100

    return emotion, confidence


# -----------------------------
# User Interface
# -----------------------------
st.title("🎙️ Speech Emotion Recognition")

st.write(
    "Upload a speech audio file and the AI model "
    "will predict the emotion expressed in the speech."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload your audio file",
    type=["wav", "mp3", "ogg"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button("🔍 Predict Emotion"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_file:

            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        try:

            emotion, confidence = predict_emotion(temp_path)

            st.success(
                f"🎯 Predicted Emotion: {emotion.upper()}"
            )

            st.info(
                f"Confidence: {confidence:.2f}%"
            )

        except Exception as e:

            st.error(
                f"Error processing audio: {e}"
            )

        finally:

            if os.path.exists(temp_path):
                os.remove(temp_path)