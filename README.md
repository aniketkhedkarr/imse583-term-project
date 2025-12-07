# imse583-term-project

Project Name: IMSE 583: Deepfake Detection System
Members: Aniket Khedkar, Bhavika Solao, Tega, Reshma
Description: This repository contains the term project for IMSE 583, focusing on the detection of AI-generated fake faces. The system leverages Transfer Learning using a pre-trained ResNet-18 architecture to classify images as "Real" or "Fake." The pipeline includes robust data preprocessing, hyperparameter optimization using Optuna, and a hybrid classification approach utilizing XGBoost on extracted deep features.

Key Features:

State-of-the-Art Architecture: Fine-tuned ResNet-18 on the "Fake vs Real" dataset.

Advanced Tuning: Automated hyperparameter search (Learning Rate, Batch Size, Optimizer) via Optuna.

Hybrid Modeling: Implementation of a CNN-XGBoost hybrid pipeline for enhanced decision boundaries.

Evaluation Suite: Comprehensive metrics including Confusion Matrix, ROC-AUC curves, and F1-score analysis.

Repo Map

config.py: Central configuration file storing dataset paths, image dimensions (224×224), normalization stats (ImageNet means/stds), and default hyperparameters.

data_loader.py: Handles dataset ingestion using torchvision.datasets.ImageFolder. Defines the transformation pipeline (Resize, ToTensor, Normalize) and creates Train/Val/Test DataLoaders.

model_resnet.py: Defines the DeepfakeResNet class. Loads the pre-trained ResNet-18, freezes early layers (optional), and replaces the final fully connected layer with a binary classification head.

train.py: The main training engine. Runs the training loop, computes CrossEntropyLoss, performs backpropagation, and saves model checkpoints (best_model.pth) based on validation accuracy.

tune_hyperparameters.py: An Optuna-driven script that runs multiple trials to automatically find the optimal learning rate, batch size, and weight decay. It minimizes the validation loss.

evaluate.py: Loads the best saved model and runs it against the Test set. Generates the Confusion Matrix, prints the Classification Report (Precision/Recall/F1), and calculates Test Accuracy.

visualize_metrics.py: Contains helper functions to plot training history (Loss/Accuracy curves over epochs) and generate the ROC (Receiver Operating Characteristic) Curve.

train_hybrid_xgboost.py: A specialized script that strips the classification layer from the trained ResNet, uses the model as a Feature Extractor (outputting 512-dim vectors), and trains an XGBoost classifier on these features for improved performance.

predict.py: Inference script. Takes a single image path as input, applies transformations, and outputs the probability of the face being "Fake" or "Real."
