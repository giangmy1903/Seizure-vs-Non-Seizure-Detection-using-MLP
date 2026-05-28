# Seizure vs Non-Seizure Detection Using MLP

## Project Overview

This project explores the application of machine learning for automated seizure detection using electroencephalogram (EEG) signal data. The objective was to build a binary classification model capable of identifying seizure activity from non-seizure brain activity using neural networks and structured EEG features.

The project focuses on the complete machine learning pipeline including data preprocessing, feature scaling, model architecture design, training, evaluation, and performance analysis while addressing challenges such as overfitting and model generalization.

## Machine Learning Workflow

**Data Preprocessing**
  * Cleaned and standardized EEG signal data using StandardScaler
  * Segmented raw EEG signals into structured numerical feature vectors
  * Prepared training and validation datasets using train_test_split
  * Processed 178 EEG features per observation for model input
    
**Model Development**
Built a supervised binary classification model using PyTorch with the following architecture:
 * Input Layer: 178 EEG features
 * Hidden Layers: 128 → 64 → 32 neurons
 * Activation Function: ReLU
 * Output Layer: Sigmoid classifier
 * Loss Function: Binary Cross-Entropy Loss
 * Optimizer: Adam Optimizer
 * Training Duration: 200 epochs

The Multi-Layer Perceptron (MLP) architecture was selected for its ability to capture non-linear relationships efficiently while maintaining computational simplicity for structured EEG data.

## Model Performance
**Training Results**
 * Achieved near-100% training accuracy
 * Training loss consistently decreased throughout epochs
 * Successfully learned seizure-related EEG signal patterns
**Validation Results**
 * Achieved approximately 71% validation accuracy
 * Precision, recall, and F1-score averaged approximately 0.71
 * Validation metrics revealed overfitting and limited model generalization

## Key Findings
 * The model effectively identified meaningful EEG classification patterns
 * Precision exceeded recall, indicating fewer false positives, but some missed seizure events
 * Performance differences between training and validation sets highlighted challenges in generalizing neural network models on biomedical datasets

## Limitations
* Training accuracy near 100%, but validation accuracy stagnating at 68% indicates **overfitting** and poor generalization.
* The MLP may not capture the sequential nature of EEG signals. More sophisticated models (e.g., CNNs or RNNs) could better handle temporal dependencies.
* Limited knowledge of biology and neuroscience hindered effective feature engineering and model design.
* MLPs provide little insight into the decision‑making process, making it hard to interpret which features contribute most to the prediction.

## Tools
Python

PyTorch

Pandas

NumPy

Scikit-learn

Jupyter Notebook
