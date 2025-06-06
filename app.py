import streamlit as st
from pydub import AudioSegment
import numpy as np
import helper
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Sidebar Title
st.sidebar.title("UrbanSound8K Audio Classifier")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Insert File", type=['wav', 'mp3'])

if uploaded_file is not None:
    # Convert MP3 to WAV if needed
    if uploaded_file.name.endswith('.mp3'):
        sound = AudioSegment.from_mp3(uploaded_file)
        sound.export("audio.wav", format="wav")
        uploaded_file_path = "audio.wav"
    else:
        uploaded_file_path = uploaded_file

    # Preprocessing
    raw, sample_rate = helper.preprocess(uploaded_file_path)

    # Play Audio
    st.title("Play Uploaded File & Waveplot")
    st.audio(uploaded_file, format='audio/wav')
    
    fig, ax = plt.subplots()
    librosa.display.waveshow(raw, sr=sample_rate, ax=ax)
    st.pyplot(fig)

    # Prediction
    st.title("Predicted Class")
    prediction = helper.audio_to_result(uploaded_file_path)
    le.classes_ = np.load('classes.npy')
    pred_class = le.inverse_transform(prediction)
    st.success(f"Predicted Class: {pred_class[0]}")

    # Spectrogram
    st.title('Spectrogram')
    X_db = helper.Spectrogram(raw)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(X_db, sr=sample_rate, x_axis="time", y_axis="hz", ax=ax)
    plt.colorbar(img, ax=ax)
    plt.title("Input Audio's Spectrogram")
    st.pyplot(fig)

    # MFCC
    st.title('Mel-Frequency Cepstral Coefficients (MFCC)')
    mfcc = helper.MFCC(raw, sample_rate)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfcc, x_axis="time", ax=ax)
    plt.colorbar(img, ax=ax)
    plt.title("MFCC")
    st.pyplot(fig)

    # Zero Crossing Rate
    st.title('Zero Crossing Rate')
    zero_crossing = helper.ZCR(raw)
    fig, ax = plt.subplots()
    ax.plot(raw[4700:5500])
    plt.title("Waveform Segment")
    st.pyplot(fig)
    st.write("Total Number of Zero Crossings:", int(np.sum(zero_crossing)))

    # Spectral Centroid & Roll-Off
    st.title('Spectral Centroid & Roll-Off')
    C, R = helper.Spectral(raw, sample_rate)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.semilogy(C.T, color='r')
        plt.ylabel("Hz")
        plt.title("Spectral Centroid")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.semilogy(R.T, color='g')
        plt.ylabel("Hz")
        plt.title("Spectral Roll-Off")
        st.pyplot(fig)

    # Chroma Feature
    st.title('Chroma Feature')
    chroma = helper.Chroma(raw, sample_rate)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    plt.colorbar(img, ax=ax)
    plt.title("Chromagram")
    st.pyplot(fig)

    # RMSE & Log Power Spectrogram
    st.title('RMSE & Log Power Spectrogram')
    S, RMSEn, times = helper.RMSE(raw)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.semilogy(times, RMSEn[0])
        plt.title("Root Mean Squared Energy")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), 
                                       y_axis='log', x_axis='time', ax=ax)
        plt.colorbar(img, ax=ax)
        plt.title("Log Power Spectrogram")
        st.pyplot(fig)
