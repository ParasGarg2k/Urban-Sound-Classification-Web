# 🎧 Urban Sound Classification

This project aims to classify urban sound recordings into predefined categories using four deep learning models: **CNN**, **ResNet**, **Wave2Vec2**, and **AST (Audio Spectrogram Transformer)**. We compare their performances using multiple metrics and deploy a Streamlit web application for real-time predictions.

---

## 📌 Table of Contents

- [🎯 Introduction](#-introduction)
- [📊 Performance Comparison](#-performance-comparison)
- [📈 Confusion Matrices](#-confusion-matrices)
- [🧪 Training & Testing Accuracy](#-training--testing-accuracy)
- [🖥️ Web App Screenshots](#-web-app-screenshots)
- [📄 Project Report](#-project-report)
- [🚀 Run the Project](#-run-the-project)
- [📁 Dataset](#-dataset)
- [🛠️ Technologies Used](#-technologies-used)

---

## 🎯 Introduction

Urban environments are full of distinctive sounds—car horns, sirens, dog barks, drilling, etc. This project leverages deep learning techniques to classify such environmental audio samples from the **UrbanSound8K dataset**. The system can aid in **smart city solutions**, **noise pollution analysis**, and **automatic tagging systems**.

The models evaluated:
- 🔷 CNN (1D)
- 🟩 ResNet (on Mel-spectrograms)
- 🟪 Wave2Vec2 (Facebook’s pretrained speech model)
- 🟨 AST (Audio Spectrogram Transformer)

---

## 📊 Performance Comparison
![Performance Comparison](images/Screenshot 2025-06-10 094837.png)


---

## 📈 Confusion Matrices
![Confusion Matrices](images/Screenshot 2025-06-10 094837.png)


---

## 🧪 Training & Testing Accuracy Graphs
![Training & Testing Accuracy Graphs](images/Screenshot 2025-06-10 094837.png)

---

## 🧪 Training & Testing Loss Graphs
![Training & Testing Loss Graphs](images/Screenshot 2025-06-10 094837)

---

## 🖥️ Web App Screenshots
![Web App Screenshots](images/Screenshot 2025-06-10 094837)


---

## 📄 Project Report

The complete report detailing:
- Data preprocessing techniques (e.g., MFCC, Mel-Spectrograms)
- Model architecture for each approach
- Training parameters (batch size, optimizer, loss)
- Evaluation metrics (confusion matrix, precision, recall, F1)
- Challenges and future improvements

📄 **[Download Report (PDF)](report/Urban_Sound_Report.pdf)**

---


