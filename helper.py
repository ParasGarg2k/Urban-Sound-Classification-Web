import librosa
import librosa.display
import numpy as np
import IPython.display as ipd
import tensorflow as tf
import tensorflow.keras.models as models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Flatten, Dense, Dropout
import cv2

# -------- Model Building Function --------
def build_model(num_labels):
    model = Sequential([
        Input(shape=(50, 1)), 
        Conv1D(64, 3, activation='relu'), 
        MaxPooling1D(2),
        BatchNormalization(),

        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        BatchNormalization(),

        Conv1D(256, 3, activation='relu'),
        MaxPooling1D(2),
        BatchNormalization(),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_labels, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------- Feature Extraction Helpers --------
def preprocess(data, max_length_sec=4):
    # Load audio data
    audio_data, sample_rate = librosa.load(data, res_type='kaiser_fast')
    
    # Calculate the maximum number of samples for 4 seconds
    max_samples = max_length_sec * sample_rate
    
    # Trim or pad the audio to 4 seconds
    if len(audio_data) > max_samples:
        audio_data = audio_data[:max_samples]  # Trim if longer than 4 seconds
    else:
        audio_data = np.pad(audio_data, (0, max_samples - len(audio_data)), mode='constant')  # Pad if shorter than 4 seconds
    
    return audio_data, sample_rate


def play_audio(raw, sr):
    return ipd.Audio(raw, rate=sr)

def Spectrogram(raw):
    X = librosa.stft(raw)
    X_db = librosa.amplitude_to_db(abs(X))
    return X_db

def MFCC(raw, sr):
    return librosa.feature.mfcc(y=raw, sr=sr)

def ZCR(raw):
    return librosa.zero_crossings(raw)

def Spectral(raw, sr):
    spec_cent = librosa.feature.spectral_centroid(y=raw, sr=sr)
    spec_roll = librosa.feature.spectral_rolloff(y=raw, sr=sr)
    return spec_cent, spec_roll

def Chroma(raw, sr):
    return librosa.feature.chroma_stft(y=raw, sr=sr)

def RMSE(raw):
    S = librosa.magphase(librosa.stft(y=raw, window=np.ones, center=False))[0]
    RMSEn = librosa.feature.rms(S=S)
    times = librosa.times_like(RMSEn)
    return S, RMSEn, times

def audio_to_result(filename):
    num_labels = 10  # Replace with the actual number of classes
    model = build_model(num_labels)
    model.load_weights('cnnzeropad.weights.h5')  # Load model weights

    raw, sr = preprocess(filename, max_length_sec=4)  # Preprocess the audio to 4 seconds

    # Extract MFCC features (e.g., 40 MFCC coefficients)
    mfcc = librosa.feature.mfcc(y=raw, sr=sr, n_mfcc=40)

    # Optionally, average across the time dimension (axis=1) to reduce the shape to (40,)
    mfcc_mean = np.mean(mfcc, axis=1)

    # If the number of MFCC features is less than 50, pad to make it 50
    if len(mfcc_mean) < 50:
        pad_width = 50 - len(mfcc_mean)
        mfcc_mean = np.pad(mfcc_mean, (0, pad_width), mode='constant')
    else:
        mfcc_mean = mfcc_mean[:50]  # Trim if too long

    # Reshape to match input shape of (1, 50, 1)
    X_input = mfcc_mean.reshape(1, 50, 1)

    # Predict the class
    pred = model.predict(X_input)
    prediction = np.argmax(pred, axis=-1)
    
    return prediction



