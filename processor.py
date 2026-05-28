import librosa
import numpy as np
import matplotlib.pyplot as plt
import io

def extract_features(audio_path):
    """
    Extract MFCC, Chroma, Mel Spectrogram, and Zero Crossing Rate features.
    """
    try:
        # Load audio file
        data, sample_rate = librosa.load(audio_path, duration=2.5, offset=0.6)
        
        # Result array
        result = np.array([])
        
        # ZCR
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
        result = np.hstack((result, zcr))
        
        # Chroma
        stft = np.abs(librosa.stft(data))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
        
        # MFCC
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mfcc))
        
        # Mel
        mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
        
        return result
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def get_audio_visualizations(audio_path):
    """
    Generate waveform and spectrogram visualizations.
    """
    data, sample_rate = librosa.load(audio_path)
    
    # Waveform
    fig_wave, ax_wave = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(data, sr=sample_rate, ax=ax_wave)
    ax_wave.set_title('Waveform')
    
    # Spectrogram
    fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='hz', ax=ax_spec)
    ax_spec.set_title('Spectrogram')
    plt.colorbar(ax_spec.collections[0], ax=ax_spec)
    
    return fig_wave, fig_spec

def predict_emotion(features):
    """
    Mock prediction logic for demo purposes.
    In a real app, this would load a pre-trained model.
    """
    emotions = ['Happy', 'Sad', 'Angry', 'Fear', 'Neutral', 'Calm', 'Surprise']
    # Use the sum of features as a seed for consistent mock prediction for the same file
    seed = int(np.sum(features) * 1000) % len(emotions)
    prediction = emotions[seed]
    confidence = 0.65 + (seed % 30) / 100.0
    
    return prediction, confidence
