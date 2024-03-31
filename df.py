import streamlit as st
import librosa
import numpy as np
from keras.models import load_model

# Load the saved model
model = load_model("https://github.com/roshangeorge97/deepfake_audio_detection/blob/main/model.h5")

# Define a function to preprocess the audio file
def preprocess_audio(audio_file, max_length=500):
    try:
        # Read audio file
        audio, _ = librosa.load(audio_file, sr=16000)
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
        # Pad or trim MFCCs to ensure a consistent shape
        if mfccs.shape[1] < max_length:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :max_length]
        # Normalize MFCC features
        mfccs_normalized = (mfccs - np.mean(mfccs)) / np.std(mfccs)
        return mfccs_normalized
    except Exception as e:
        print(f"Error encountered while processing file: {audio_file}")
        return None

# Define the Streamlit app
def main():
    st.title("Deep Fake Prediction App")

    # File uploader
    audio_file = st.file_uploader("Upload an audio file (.wav, .mp3)", type=["wav","mp3"])

    if audio_file is not None:
        # Preprocess the uploaded audio file
        features = preprocess_audio(audio_file)
        if features is not None:
            # Perform prediction
            prediction = model.predict(np.expand_dims(features, axis=0))[0][0]
            if prediction >= 0.5:
                st.markdown(f"<h1 style='text-align: center;'>Prediction: Fake Voice </h1>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h1 style='text-align: center;'>Prediction: Real Voice</h1>", unsafe_allow_html=True)

    st.write("Developed by Surendar S")


if __name__ == "__main__":
    main()
