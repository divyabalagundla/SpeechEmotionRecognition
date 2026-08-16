import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import os

# Load trained files
model = joblib.load("models/emotion_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a speech audio file and the AI model will predict the emotion.")

st.info(
    "Supported emotions: Neutral, Calm, Happy, Sad, Angry, "
    "Fearful, Disgust, and Surprised."
)

uploaded_file = st.file_uploader(
    "Upload a WAV audio file",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button("Predict Emotion"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_file:

            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        try:
            # Load audio
            audio, sample_rate = librosa.load(
                temp_path,
                sr=16000
            )

            # Extract MFCC
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=40
            )

            mfcc_mean = np.mean(mfcc, axis=1)

            # Scale
            features = scaler.transform(
                mfcc_mean.reshape(1, -1)
            )

            # Predict
            prediction = model.predict(features)[0]

            emotion = label_encoder.inverse_transform(
                [prediction]
            )[0]

            # Probability
            probabilities = model.predict_proba(features)[0]

            confidence = np.max(probabilities) * 100

            st.success(
                f"Predicted Emotion: {emotion.upper()}"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        except Exception as e:

            st.error(f"Error processing audio: {e}")

        finally:

            if os.path.exists(temp_path):
                os.remove(temp_path)