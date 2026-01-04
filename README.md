# Face Mask Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

This repository contains a complete, end-to-end **computer vision system** for detecting face mask usage in images. The system is designed to classify faces into three categories:

- **With Mask**
- **Without Mask**
- **Incorrect Mask**

The project demonstrates the full pipeline of **dataset preparation, deep learning model training, ethical evaluation, and deployment** as an interactive web application.

---

## Project Objectives

- Develop a robust computer vision system using **Transfer Learning (MobileNetV2)**.
- Apply rigorous **stratified splitting** to ensure statistically significant testing.
- Evaluate performance using **Macro-F1, Precision, and Recall** beyond simple accuracy.
- **Implement Responsible AI practices** by addressing privacy and bias.
- Deploy the model as a user-friendly **Streamlit** web application.

---

## GUI
**Application URL:** [Face Mask Detection ](https://cv-facemaskdetection.streamlit.app/)

---

## Ethical AI & Privacy Protection (Crucial)

This project strictly adheres to **Responsible AI** guidelines. Since the system processes human faces, the following measures were implemented:

### 1. Privacy Preservation (Identity Anonymization)
To protect individual privacy, the deployment application includes an **automatic anonymization layer**. 
- **Mechanism:** The system applies pixelation/blurring to the upper region of the detected face (eyes and forehead).
- **Result:** The system displays the mask status (public health interest) while obscuring the individual's identity (personal privacy).

### 2. Bias & Safety Assessment
- **Dataset Balance:** We used stratified sampling to ensure the model is not biased towards the majority class.
- **Safety Analysis:** The model shows a very low False Negative rate (predicting "With Mask" when the person is unmasked), which is critical for public health safety.

---

## Model Overview

- **Architecture:** MobileNetV2 (pretrained on ImageNet)
- **Framework:** PyTorch
- **Training Strategy:** Fine-tuning the classifier layer
- **Input Resolution:** 224 × 224 RGB images
- **Output:** 3-class softmax probability distribution

MobileNetV2 was selected for its balance between **high accuracy** and **computational efficiency**, making it suitable for both academic experimentation and real-time inference.

---

## Dataset

- **Source:** [Kaggle Face Mask Dataset](https://www.kaggle.com/datasets/shiekhburhan/face-mask-dataset)
- **Total Available:** ~14,500 images
- **Classes:** `with_mask`, `without_mask`, `incorrect_mask`

### Dataset Experimentation
Two experimental settings were evaluated to determine the optimal data size:

| Setting | Total Images | Train | Validation | Test |
|------|-------------|-------|-----------|------|
| Baseline | 1,000 | 699 | 150 | 151 |
| **Final (Selected)** | **5,000** | **~3,500** | **~750** | **751** |

The final model and reported results are based on the **5,000-image stratified subset**, providing stronger evidence of generalization and robustness.

---

## Performance Evaluation

The final model was evaluated on a **held-out test set of 751 images**.

### Test Set Metrics

| Class | Precision | Recall | F1-score |
|------|----------|--------|---------|
| **Incorrect Mask** | 1.00 | 0.97 | 0.98 |
| **With Mask** | 0.96 | 0.97 | 0.96 |
| **Without Mask** | 0.96 | 0.98 | 0.97 |

**Overall Accuracy:** 97%  
**Macro F1-score:** 0.97  

### Key Observations
- **High Generalization:** The model generalizes well with no severe overfitting.
- **Ambiguity Handling:** Slight confusion occurs between *with mask* and *incorrect mask*, which is expected due to visual ambiguity (e.g., masks slipping slightly).
- **Data Scaling:** Increasing the dataset size significantly reduced optimistic bias observed in smaller subsets.

---

## Deployment (Streamlit Web App)

The trained model has been deployed using **Streamlit** to allow interactive image-based inference.

### Features
- **Upload Interface:** Support for JPG/PNG images.
- **Real-time Inference:** Instant prediction with confidence scores.
- **Visual Analytics:** Probability distribution charts.
- **Privacy Mode:** Automatic blurring of faces in the output.
