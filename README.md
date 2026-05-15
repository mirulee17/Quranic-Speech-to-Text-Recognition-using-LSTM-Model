# Quranic Speech-to-Text Recognition using LSTM Model

A Final Year Project (FYP) focused on Quranic speech recognition using Deep Learning techniques with an LSTM (Long Short-Term Memory) neural network model.

This project classifies Quranic recitations into specific Surahs using audio preprocessing, MFCC feature extraction, and sequence-based deep learning models.

---

# Project Overview

The system is designed to recognize and classify Quranic verse recitations through audio input.

The backend processes recorded recitations and predicts the corresponding Surah using trained Bidirectional LSTM models.

This repository contains:
- Python backend
- Jupyter Notebook training files
- Trained deep learning models
- Audio preprocessing pipeline
- Feature extraction datasets

---

# Features

- Quranic speech classification
- LSTM-based deep learning model
- MFCC audio feature extraction
- Bidirectional LSTM architecture
- Audio preprocessing pipeline
- Model training visualization
- Flask backend integration
- Multiple Surah classification models

---

# Technologies Used

## Machine Learning & Backend
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Flask

## Development Tools
- PyCharm
- Jupyter Notebook
- GitHub

---

# Project Structure

```bash
QuranicBackend/
│
├── Images/
│
├── debug_audio/
├── exportToHTML/
├── received_audio/
│
├── LSTM alfatihah.ipynb
├── LSTM alikhlas.ipynb
├── LSTM annas.ipynb
│
├── flas_server.py
│
├── bilstm_annas_classifier.keras*
├── bilstm_fatihah_classifier.keras*
│
├── lstm_annas_classifier.keras*
├── lstm_fatihah_classifier.keras*
├── lstm_verse_classifier.h5*
├── lstm_verse_classifier.keras*
│
├── X_annas.npy*
├── X_annas_mel.npy*
├── X_fatihah.npy*
├── X_fatihah_mel.npy*
├── X_mfcc.npy*
│
├── y_annas_labels.npy*
├── y_fatihah_labels.npy*
├── y_labels.npy*
│
└── .gitignore (label with *)
