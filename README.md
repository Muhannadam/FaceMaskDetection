# Face Mask Detection System (Computer Vision Project)

This repository contains a complete, end-to-end **computer vision system** for detecting face mask usage in images.  
The system classifies faces into three categories:

- **With Mask**
- **Without Mask**
- **Incorrect Mask**

The project was developed as part of an academic computer vision coursework and demonstrates the full pipeline of **dataset preparation, model training, evaluation, and deployment** using modern deep learning techniques.

---

## Project Objectives

- Develop a robust computer vision system using deep learning.
- Apply proper **train/validation/test splits** with sufficient test size.
- Evaluate model performance using **multiple metrics** beyond accuracy.
- Demonstrate **responsible AI practices** and transparency.
- Deploy the trained model as an **interactive Streamlit web application**.

---

## Model Overview

- **Architecture:** MobileNetV2 (pretrained on ImageNet)
- **Framework:** PyTorch
- **Training Strategy:** Fine-tuning the classifier layer
- **Input Resolution:** 224 × 224 RGB images
- **Output:** 3-class softmax probability distribution

MobileNetV2 was selected due to its strong balance between **accuracy and computational efficiency**, making it suitable for both academic experimentation and real-time inference.

---

## Dataset

- **Source:** Kaggle – Face Mask Dataset  
  https://www.kaggle.com/datasets/shiekhburhan/face-mask-dataset
- **Total Images Available:** ~14,500
- **Classes:**  
  - `with_mask`  
  - `without_mask`  
  - `incorrect_mask`

### Dataset Usage in This Project

Two experimental settings were evaluated:

| Setting | Total Images | Train | Validation | Test |
|------|-------------|-------|-----------|------|
| Baseline | 1,000 | 699 | 150 | 151 |
| Final (Selected) | **5,000** | ~3,500 | ~750 | **751** |

The final model and reported results are based on the **5,000-image subset**, providing stronger evidence of generalization and robustness.

All dataset splits were **stratified** to preserve class balance.

---

## Performance Evaluation

The final model was evaluated on a **held-out test set of 751 images**.

### Test Set Results

| Class | Precision | Recall | F1-score |
|------|----------|--------|---------|
| Incorrect Mask | 1.00 | 0.97 | 0.98 |
| With Mask | 0.96 | 0.97 | 0.96 |
| Without Mask | 0.96 | 0.98 | 0.97 |

**Overall Accuracy:** 97%  
**Macro F1-score:** 0.97  

### Key Observations
- The model generalizes well with no severe overfitting.
- Slight confusion occurs between *with mask* and *incorrect mask*, which is expected due to visual ambiguity.
- Increasing the dataset size reduced optimistic bias observed in smaller subsets, resulting in more realistic and reliable performance.

---

## Evaluation Metrics

The following metrics were implemented and reported:
- Accuracy
- Precision (macro-average)
- Recall (macro-average)
- F1-score (macro-average)
- Confusion Matrix
- Detailed Classification Report

---

## Deployment (Streamlit Web App)

The trained model has been deployed using **Streamlit** to allow interactive image-based inference.

### Features
- Upload an image via the web interface
- Real-time prediction with confidence score
- Probability distribution visualization across classes

### Tech Stack (Deployment)
- Streamlit
- PyTorch
- Torchvision
- Pillow

