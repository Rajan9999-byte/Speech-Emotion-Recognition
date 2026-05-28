# EmotiVoiceAI — Speech Emotion Recognition System

## Project Overview

EmotiVoiceAI is an AI-powered Speech Emotion Recognition System that detects human emotions from voice/audio recordings using Machine Learning and Deep Learning techniques.

The system analyzes speech patterns, tone, pitch, frequency, and audio features to identify emotions such as:
- Happy
- Sad
- Angry
- Fear
- Neutral
- Calm
- Surprise

## Problem Statement

Human emotions play a major role in communication, but traditional systems cannot naturally understand emotional states from speech. Organizations need intelligent systems that can analyze customer emotions, monitor stress levels, and improve human-computer interaction.

EmotiVoiceAI solves this by using AI models to automatically recognize emotions from speech/audio input.

## Core Features

- **Audio Upload Support**: Upload WAV or MP3 files for analysis
- **Speech Emotion Detection**: AI-powered emotion classification
- **Audio Preprocessing**: Noise reduction and normalization
- **Feature Extraction**: MFCC, Chroma, Mel Spectrogram, Zero Crossing Rate
- **Real-time Emotion Prediction**: Instant emotion classification with confidence scores
- **Emotion Visualization Dashboard**: Interactive charts and graphs
- **Voice Waveform Visualization**: Visual representation of audio signals
- **Spectrogram Analysis**: Frequency-time domain analysis
- **Model Performance Metrics**: Accuracy and F1-score analysis

## Tech Stack

### Programming Language
- Python 3.11

### ML/DL Libraries
- **Librosa**: Audio processing and feature extraction
- **Scikit-learn**: Machine learning models
- **NumPy & Pandas**: Data processing
- **Matplotlib & Seaborn**: Visualization

### Deployment/UI
- **Streamlit**: Interactive web dashboard

## Project Workflow

1. User uploads audio/speech file
2. Audio preprocessing performed (normalization, noise reduction)
3. Features extracted from speech (MFCC, Chroma, Mel, ZCR)
4. ML/DL model predicts emotion
5. Emotion result displayed with confidence score
6. Graphs and analytics generated
7. Dashboard visualizes predictions and model performance

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone or navigate to project directory
cd emotivoice_ai

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Dashboard

1. **Navigate to Dashboard**: Select "Dashboard" from the sidebar
2. **Upload Audio**: Click "Upload an audio file" and select a WAV or MP3 file
3. **View Results**: The system will display:
   - Detected emotion and confidence score
   - Emotion probability distribution
   - Waveform visualization
   - Spectrogram analysis
   - Extracted features

4. **Model Analysis**: View model performance metrics and confusion matrix
5. **About**: Learn more about the project

## Project Modules

### Module 1 — Audio Input
- Audio file upload
- Microphone recording support
- WAV/MP3 file reading

### Module 2 — Audio Preprocessing
- Noise reduction
- Normalization
- Audio trimming

### Module 3 — Feature Extraction
- MFCC (Mel-Frequency Cepstral Coefficients)
- Chroma features
- Mel Spectrogram
- Zero Crossing Rate

### Module 4 — Emotion Classification
- CNN (Convolutional Neural Networks)
- LSTM (Long Short-Term Memory)
- Random Forest
- SVM (Support Vector Machines)

### Module 5 — Dashboard & Visualization
- Predicted emotion display
- Confidence score visualization
- Waveform graph
- Spectrogram visualization
- Model performance metrics

## Industry Applications

- **Customer Service Analytics**: Analyze customer satisfaction and frustration
- **Mental Wellness Monitoring**: Detect stress and emotional distress
- **Virtual Assistants**: Improve AI assistant responses based on user emotion
- **Call Center Intelligence**: Quality analysis and agent performance
- **Healthcare Applications**: Mental health monitoring and therapy support
- **HR Interview Systems**: Candidate emotion analysis
- **EdTech Platforms**: Student engagement and emotional state monitoring

## Companies Using Similar Technology

- Google
- Amazon
- Microsoft
- NVIDIA

## Resume-Worthy Value

This project demonstrates:
- AI/ML implementation and understanding
- Deep Learning concepts and architectures
- Audio processing and signal analysis
- NLP/Speech AI understanding
- Real-world dataset handling
- Feature engineering techniques
- Model training and evaluation
- AI application deployment
- Full-stack application development

## Datasets

The project can be trained on:
- **RAVDESS Dataset**: Ryerson Audio-Visual Emotion Speech and Song Database
- **TESS Dataset**: Toronto Emotional Speech Set
- **CREMA-D Dataset**: Crowd-sourced Emotional Multimodal Actors Dataset

## File Structure

```
emotivoice_ai/
├── app.py                 # Main Streamlit application
├── processor.py           # Audio processing and feature extraction
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Future Enhancements

- Integration with pre-trained deep learning models (CNN, LSTM)
- Real-time microphone input processing
- Multi-language support
- Advanced noise reduction algorithms
- Emotion intensity analysis
- Speaker identification
- Real-time dashboard updates

## License

This project is open-source and available for educational purposes.

## Support

For issues or questions, please refer to the project documentation or contact the development team.
