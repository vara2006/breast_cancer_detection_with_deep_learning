# Breast Cancer Detection using Deep Learning

## Project Overview

Breast Cancer Detection is a Machine Learning and Deep Learning based classification project that predicts whether a breast tumor is **Malignant (Cancerous)** or **Benign (Non-Cancerous)** using medical diagnostic features.

In this project, the Breast Cancer Wisconsin dataset is used. A Deep Neural Network (DNN) model is developed using TensorFlow and Keras to learn patterns from the input features and classify tumors accurately.

The model achieved an accuracy of **98.2%** on the test dataset.

---

# Features of the Project

- Data preprocessing and cleaning
- Handling missing values
- Feature scaling using StandardScaler
- Binary classification (Malignant / Benign)
- Deep Learning model implementation
- Model evaluation using:
  - Accuracy
  - Confusion Matrix
  - Accuracy Graph
  - Loss Graph
- Prediction on new patient data
- Saving trained model for future use

---

# Dataset

Dataset used:

**Breast Cancer Wisconsin (Diagnostic) Dataset**

Source:
Kaggle - UCI Machine Learning Repository

Dataset contains information about breast cell nuclei extracted from images.

### Dataset Details:

- Total records: 569
- Total features: 30
- Target variable: Diagnosis

### Classes:

| Label | Meaning |
|---|---|
| M | Malignant (Cancer) |
| B | Benign (Non-Cancer) |

---

# Input Features

The model uses 30 medical features:

- Radius Mean
- Texture Mean
- Perimeter Mean
- Area Mean
- Smoothness Mean
- Compactness Mean
- Concavity Mean
- Concave Points Mean
- Symmetry Mean
- Fractal Dimension Mean

- Radius SE
- Texture SE
- Perimeter SE
- Area SE
- Smoothness SE
- Compactness SE
- Concavity SE
- Concave Points SE
- Symmetry SE
- Fractal Dimension SE

- Radius Worst
- Texture Worst
- Perimeter Worst
- Area Worst
- Smoothness Worst
- Compactness Worst
- Concavity Worst
- Concave Points Worst
- Symmetry Worst
- Fractal Dimension Worst

---

# Technologies Used

## Programming Language

- Python

## Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn

## Machine Learning

- Scikit-learn

Used for:
- Data splitting
- Feature scaling
- Model evaluation

## Deep Learning

- TensorFlow
- Keras

Used for:
- Building Neural Network model
- Training and prediction

## Dataset Handling

- KaggleHub

---

# Model Architecture

The Deep Learning model is built using Artificial Neural Network.

### Layers:

Input Layer:
- 30 input features

Hidden Layer 1:
- Dense layer
- 64 neurons
- ReLU activation

Dropout Layer:
- 30% dropout

Hidden Layer 2:
- Dense layer
- 32 neurons
- ReLU activation

Dropout Layer:
- 30% dropout

Output Layer:
- 1 neuron
- Sigmoid activation

---

# Model Training

Training Parameters:

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Epochs: 50
- Batch Size: 32
- Validation Split: 20%

---

# Model Performance

## Accuracy

Test Accuracy:

**98.2%**

---

# Output Visualizations

The project generates the following outputs:

## 1. Confusion Matrix

Shows the comparison between actual and predicted classes.

## 2.  Accuracy
## 3.  Accuracy graph
## 4.  Loss graph
