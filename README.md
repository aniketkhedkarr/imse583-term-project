# imse583-term-project

Project Name: IMSE 583: Deepfake Detection System

Members: Aniket Khedkar, Bhavika Solao, Tega, Reshma

Description: This repository contains the term project for IMSE 583, focusing on the detection of AI-generated fake faces. The system leverages Transfer Learning using a pre-trained ResNet-18 architecture to classify images as "Real" or "Fake." The pipeline includes robust data preprocessing, hyperparameter optimization using Optuna, and a hybrid classification approach utilizing XGBoost on extracted deep features.

Original Dataset: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

Preprocessed Dataset: https://drive.google.com/file/d/186o5rl8GU6RfRYDHOCGLVKDrbgXKqY7b/view?usp=share_link

Key Features:

State-of-the-Art Architecture: Fine-tuned ResNet-18 on the "Fake vs Real" dataset.

Advanced Tuning: Automated hyperparameter search (Learning Rate, Batch Size, Optimizer) via Optuna.

Hybrid Modeling: Implementation of a CNN-XGBoost hybrid pipeline for enhanced decision boundaries.

Evaluation Suite: Comprehensive metrics including Confusion Matrix, ROC-AUC curves, and F1-score analysis.

Repo Map

Data_Cleaning_and_Preprocessing.ipynb

Purpose: The entry point for the pipeline.

Function: Downloads the raw dataset from Kaggle (manjilkarki/deepfake-and-real-images), scans for image integrity, and performs a stratified 70/15/15 split.

Output: Generates a structured cleaned_dataset/ directory organized into Train, Validation, and Test folders, ready for PyTorch ImageFolder ingestion.

FAKE_FACE_DETECTION.ipynb

Purpose: The core training and evaluation notebook.

Function: Loads the processed data, initializes a pre-trained ResNet-18 model, and fine-tunes the final layer for binary classification.

Visualization: Generates training history plots (Loss/Accuracy), Confusion Matrices, and ROC-AUC curves to validate model performance.

hyperparameters_deepfake_detection.ipynb

Purpose: Advanced post-processing and hybrid modeling.

Function: Strips the classification head from the trained ResNet model to use it purely as a Feature Extractor. It passes the 512-dimensional feature vectors into an XGBoost Classifier to improve decision boundaries beyond standard Softmax layers.

