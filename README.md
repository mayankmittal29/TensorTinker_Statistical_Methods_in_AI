# 🧠 TensorTinker Statistical Methods in AI 🧠

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-red)

This repository contains implementations of various machine learning algorithms from scratch, including Multi-Layer Perceptron (MLP), Gaussian Mixture Models (GMM), Principal Component Analysis (PCA), Autoencoders, and Variational Autoencoders.

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [1. Multi-Layer Perceptron](#-1-multi-layer-perceptron)
  - [1.1 MLP Multi-Class Classifier](#-11-mlp-multi-class-classifier)
  - [1.2 MLP Regressor for Price Prediction in Bangalore](#-12-mlp-regressor-for-price-prediction-in-bangalore)
  - [1.3 Multi-Label News Article Classification](#-13-multi-label-news-article-classification)
- [2. Gaussian Mixture Model](#-2-gaussian-mixture-model)
- [3. Principal Component Analysis](#-3-principal-component-analysis)
  - [3.1 Explained Variance and Lossy Reconstruction](#-31-explained-variance-and-lossy-reconstruction)
  - [3.2 Classification Performance with vs without dimensionality reduction](#-32-classification-performance-with-vs-without-dimensionality-reduction)
- [4. Autoencoder](#-4-autoencoder)
- [5. Variational Autoencoder](#-5-variational-autoencoder)
- [Installation Instructions](#-installation-instructions)
- [Usage](#-usage)
- [Results](#-results)

## 🔍 Project Overview

This project implements different statistical machine learning methods from scratch to solve various real-world problems. The main focus is on understanding the underlying mathematics and algorithms of these methods and implementing them without using existing libraries (except for PyTorch for autoencoders).

## 🧠 1. Multi-Layer Perceptron

### 🔢 1.1 MLP Multi-Class Classifier

#### Problem Statement
Implementing a Multi-Layer Perceptron for classifying handwritten symbols from historical manuscripts in the SYMBOL dataset.

#### Dataset
- Images folder containing all handwritten symbol images
- 10-fold cross-validation setup with train.csv and test.csv in each fold
- Each row contains: path to image, symbol ID, and LaTeX representation

#### Implementation Details
- Custom MLP class with configurable hyperparameters:
  - Learning rate - [0.01, 0.001]
  - Activation functions (Sigmoid, Tanh, ReLU implemented from scratch)
  - Optimizers (SGD, Batch GD, Mini-Batch GD implemented from scratch)
  - Number and size of hidden layers - [[128], [64, 32], [128, 64, 32]]
  - batch sizes - [32, 64, 128]
- Methods for forward and backward propagation
- Training process with various configurations

#### Hyperparameter Tuning with 10-Fold Validation
- Learning rate and epochs optimization
- Different hidden layer configurations
- Comparison of activation functions and optimizers
- Performance metrics: accuracy, precision, recall

#### Results and Visualizations

![Plot](best_summary_2.1.png)

##### Best configuration: Sigmoid, sgd, layers=[64, 32], lr=0.01
##### Mean accuracy: 0.6238±0.0064

### 🏠 1.2 MLP Regressor for Price Prediction in Bangalore

#### Problem Statement
Building an MLP regressor to predict housing prices in Bangalore based on features like location, size, and amenities.

#### Dataset
- Bangalore housing price dataset with various features
- Requires extensive preprocessing due to missing values and outliers

#### Data Preprocessing Steps
1. Handling missing values and outliers
2. Feature selection and engineering
3. Normalization and standardization
4. Visualisations like Pie-charts, scatter-plot, Box-Plot, Correlation Heatmap etc
5. Train-validation-test split

#### Model Implementation
- Same MLP architecture as the classifier but adapted for regression
- Mean Squared Error (MSE) as the loss function
- Configurable hyperparameters

#### Evaluation Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared

#### Results

![Plot](2.2_best.png)
#### All other results of tuning are in Q2.2_files/q2.2_report.csv and cleaned csv also.

### 📰 1.3 Multi-Label News Article Classification

#### Problem Statement
Developing an MLP model to tag news articles with multiple topics simultaneously.

#### Data Preprocessing
- Parsing CSV files and handling multi-label data
- Computing TF-IDF features from scratch (limited to ~5000 features)
- Multi-label binarization
- Train-validation split

#### Model Implementation
- MLP with output neurons for each possible label
- Binary cross-entropy loss for multi-label classification
- Forward and backward propagation with support for multiple outputs

#### Hyperparameter Tuning
- Learning rate, epochs, and architecture variations
- Different activation functions and optimizers

#### Evaluation Metrics
- Accuracy
- Hamming Loss
- Precision, Recall, F1-score (for multi-label)

#### Results
![Plot](2.3_best.png)
##### Results of all hyperparameters tuned are in Q2.3_files/results_2.3.csv

## 🔔 2. Gaussian Mixture Model

#### Problem Statement
Implementing GMM from scratch and using it to segment gray matter, white matter, and cerebrospinal fluid (CSF) from brain MRI images.

#### GMM Implementation
- Expectation-Maximization (EM) algorithm
- Component initialization
- Convergence criteria
- Posterior probability calculation

#### Brain Tissue Segmentation
- Using GMM to segment the MRI image sald_031764_img.nii
- Visualization of segmentation results
- Comparison with original segmentation

#### Segmentation Results After applying GMM on brain MRI scan image:- 
#### Original Axial and its Segmented View
![Original](Q3_files/original_axial.png)
![Segmented](Q3_files/axial.png)
#### Original Coronal and its Segmented View
![Original](Q3_files/original_coronal.png)
![Segmented](Q3_files/coronal.png)
#### Original Sagittal and its Segmented View
![Original](Q3_files/original_sagittal.png)
![Segmented](Q3_files/sagittal.png)
#### Analysis

![Frequency vs Intensity](Q3_files/Frequency_intensity.png)

![GMM Distribution](Q3_files/GMM_distributions.png)

#### Misclassification Analysis
Analysis of regions with highest misclassification based on intensity distributions and GMM model characteristics.

## 🔍 3. Principal Component Analysis

### 📉 3.1 Explained Variance and Lossy Reconstruction

#### Implementation Details
- PCA implementation from scratch using NumPy
- Covariance matrix computation
- Eigenvector and eigenvalue calculation
- Projection and reconstruction

#### Dataset
- MNIST dataset with 1000 randomly sampled images (uniform class distribution)

#### Dimensionality Reduction
- Projecting data to 500, 300, 150, 30, 25, 20, 15, 10, 5 and 2 dimensions

#### Visualization and Analysis

![Plot](4_plots/Cumulative_Variance.png)
![Plot](4_plots/Variance_by_component.png)
![Plot](4_plots/2-pricipal_components.png)

#### Image Reconstruction

![Plot](4_plots/PCA.png)

### 📊 3.2 Classification Performance with vs without dimensionality reduction

#### Experimental Setup
- 40K random samples from MNIST train set and full test set (10K samples)
- MLP classifier with 2-3 fully connected layers
- Dimensionality reduction with PCA to 500, 300, 150, and 30 dimensions

#### Performance Metrics
- Accuracy
- Precision
- Recall

#### Results
![Plot](4_plots/PCA_accuracy.png)

Baseline Classification (No PCA)
Accuracy: 0.9746
Precision: 0.9746
Recall: 0.9746

Classification with 500 PCA Components
Accuracy: 0.9332
Precision: 0.9335
Recall: 0.9332

Classification with 300 PCA Components
Accuracy: 0.9550
Precision: 0.9550
Recall: 0.9550

Classification with 150 PCA Components
Accuracy: 0.9681
Precision: 0.9683
Recall: 0.9681

Classification with 30 PCA Components
Accuracy: 0.9804
Precision: 0.9804
Recall: 0.9804

Classification with 25 PCA Components
Accuracy: 0.9736
Precision: 0.9736
Recall: 0.9736

Classification with 20 PCA Components
Accuracy: 0.9728
Precision: 0.9729
Recall: 0.9728

Classification with 15 PCA Components
Accuracy: 0.9678
Precision: 0.9679
Recall: 0.9678

Classification with 10 PCA Components
Accuracy: 0.9358
Precision: 0.9360
Recall: 0.9358

Classification with 5 PCA Components
Accuracy: 0.7698
Precision: 0.7741
Recall: 0.7698

Classification with 2 PCA Components
Accuracy: 0.4727
Precision: 0.4615
Recall: 0.4727

#### Analysis
- Discussion on how PCA helps mitigate the curse of dimensionality
- Cases where PCA might not be effective
- Limitations of PCA's variance maximization assumption

## 🔄 4. Autoencoder

#### Problem Statement
Implementing an autoencoder for anomaly detection in MNIST digits.

#### Implementation Details
- PyTorch implementation of encoder and decoder networks
- Training on normal data (digits matching last digit of roll number)
- Testing on mixed normal and anomalous digits

#### Reconstruction Error Analysis

### For dimension 8
![Plot](5_plots/training_loss_dim_8.png)
![Plot](5_plots/precision_recall_curve_dim_8.png)
![Plot](5_plots/error_histogram_dim_8.png)
![Plot](5_plots/error_histogram_dim_8.png)
![Plot](5_plots/error_by_digit_dim_8.png)
![Plot](5_plots/reconstructions_dim_8.png)

#### Performance Metrics for Bottleneck Dimension = 8
##### Optimal Threshold: -24.726648
##### Precision: 0.6364
##### Recall: 0.5631
##### F1-Score: 0.5975
##### AUC-ROC: 0.9185
##### Accuracy: 0.9018

### For dimension 16
![Plot](5_plots/training_loss_dim_16.png)
![Plot](5_plots/precision_recall_curve_dim_16.png)
![Plot](5_plots/error_histogram_dim_16.png)
![Plot](5_plots/error_histogram_dim_16.png)
![Plot](5_plots/error_by_digit_dim_16.png)
![Plot](5_plots/reconstructions_dim_16.png)

#### Performance Metrics for Bottleneck Dimension = 16
##### Optimal Threshold: -9.943974
##### Precision: 0.7848
##### Recall: 0.6833
##### F1-Score: 0.7305
##### AUC-ROC: 0.9662
##### Accuracy: 0.9018

### For dimension 32
![Plot](5_plots/training_loss_dim_32.png)
![Plot](5_plots/precision_recall_curve_dim_32.png)
![Plot](5_plots/error_histogram_dim_32.png)
![Plot](5_plots/error_histogram_dim_32.png)
![Plot](5_plots/error_by_digit_dim_32.png)
![Plot](5_plots/reconstructions_dim_32.png)

#### Performance Metrics for Bottleneck Dimension = 32
##### Optimal Threshold: -9.622127
##### Precision: 0.6948
##### Recall: 0.7301
##### F1-Score: 0.7120
##### AUC-ROC: 0.9640
##### Accuracy: 0.9018

#### Anomaly Detection
- Threshold selection based on reconstruction error distribution
- Performance evaluation with precision, recall, and F1-score

#### Hyperparameter Tuning
- Testing 3 different bottleneck dimensions
- Comparison using AUC-ROC score

#### Results

![Plot](5_plots/roc_curves.png)

![Plot](5_plots/metrics_comparison.png)
#### The optimal bottleneck dimension appears to be 16, with the highest AUC (0.966) and a good balance of precision (0.785) and recall (0.683)
## 🧬 5. Variational Autoencoder

#### Problem Statement
Implementing and analyzing a Variational Autoencoder (VAE) on MNIST dataset.

#### Implementation Details
- PyTorch implementation of VAE with encoder, reparameterization, and decoder
- Binary cross-entropy loss for reconstruction
- KL divergence for latent space regularization

#### Latent Space Visualization

![Plot](6_plots/latent_space_latent_space_with_full_vae_(bce).png)

#### Ablation Studies
1. Training without reconstruction loss
![Plot](6_plots/latent_space_latent_space_without_reconstruction_loss.png)

2. Training without KL divergence loss
![Plot](6_plots/latent_space_latent_space_without_kl_loss.png)

#### Latent Space Sampling

![Plot](6_plots/grid_reconstructions_grid_reconstructions_with_full_vae_(bce).png)
![Plot](6_plots/grid_reconstructions_grid_reconstructions_without_reconstruction_loss.png)
![Plot](6_plots/grid_reconstructions_grid_reconstructions_without_kl_loss.png)
![Plot](6_plots/grid_reconstructions_grid_reconstructions_with_mse_loss.png)

#### Loss Function Comparison
- Binary cross-entropy vs. MSE reconstruction loss
- Visual comparison of generated samples

#### Visuals of generated smaples when full VAE with BCE loss and KL loss
![Plot](6_plots/with_bce.png)
#### Visual of generated smaples when VAE without reconstruction loss and but KL loss

![Plot](6_plots/without_rec.png)
#### Visual of generated smaples when full VAE with BCE loss and without KL loss

![Plot](6_plots/without_kl.png)
#### Visual of generated smaples when full VAE with MSE loss and KL loss

![Plot](6_plots/with_mse.png)

![Plot](6_plots/latent_space_latent_space_with_mse_loss.png)

![Plot](6_plots/training_loss_comparison.png)
![Plot](6_plots/test_loss_comparison.png)

## 📦 Installation Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/TensorTinker_Statistical_Methods_in_AI.git
cd TensorTinker_Statistical_Methods_in_AI
```

⭐ Feel free to star this repository if you find it useful! ⭐

📝 For any questions or issues, please open an issue on GitHub.
