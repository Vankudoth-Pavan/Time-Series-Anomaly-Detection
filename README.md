# Time-Series-Anomaly-Detection

## Overview

This project implements an advanced anomaly detection system for S&P 500 (SPX) stock price data using LSTM Autoencoders. The primary objective is to identify unusual patterns or outliers in the stock's closing prices, which could indicate significant market events or anomalies.
Key aspects of the project include:
Data Source: Historical S&P 500 closing price data.

**Methodology:** Utilizes an LSTM Autoencoder architecture for time series analysis.

**Preprocessing:** Includes data normalization and creation of time-step sequences.

**Model Architecture:** Features an encoder-decoder structure with LSTM layers and dropout for regularization.

**Anomaly Detection:** Uses reconstruction error with a set threshold to identify anomalies.

**Visualization:** Provides graphical representations of detected anomalies in the context of stock price movements.

This project demonstrates the application of deep learning techniques in financial data analysis, offering potential insights for market analysis and risk management. It showcases the power of autoencoder models in capturing complex patterns in time series data and identifying deviations from these patterns.

The implementation is flexible, allowing for adjustments in parameters such as the time step window and anomaly threshold, making it adaptable to various financial instruments and market conditions.

## Table of Contents 

1. Introduction

2. Problem Statement

3. Dataset

4. Data Pre-processing

5. Feature Engineering

6. Algorithms

7. Evaluation Metrics

8. Results and Discussion

9. How to Use

10. Conclusion

## Introduction 
This project implements an LSTM Autoencoder for anomaly detection in S&P 500 (SPX) stock price time series data. The model is designed to identify unusual patterns or outliers in the stock's closing prices, which could indicate significant market events or anomalies.

## Problem Statement 
The goal is to develop a robust anomaly detection system for financial time series data, specifically for S&P 500 stock prices. The system should be capable of identifying unusual price movements that deviate from the normal patterns, potentially signaling important market events or anomalies.

## DataSet 
The project uses historical S&P 500 closing price data. The dataset is stored in a CSV file named 'spx.csv', containing daily closing prices of the S&P 500 index.

## Data Pre-processing 
1. The dataset is loaded and parsed, with dates set as the index.
   
2. The data is split into training (95%) and testing (5%) sets.

3. StandardScaler is applied to normalize the closing prices.

## Feature Engineering 
**Time Series Windowing:** The raw closing price data is transformed into a sequence of overlapping windows, each containing 30 time steps (configurable via the TIME_STEPS variable).

**Sequence Creation:** For each window, the subsequent time step's closing price is used as the target value, creating input-output pairs for training.

**Numpy Array Conversion:** The windowed data is converted into numpy arrays for efficient processing by the neural network.

**Training-Validation Split:** The feature engineered data is split into training and validation sets, with a 90-10 split ratio.

**Input Shape Transformation:** The input data is reshaped to match the LSTM layer's expected input shape: (samples, time steps, features).

**Normalization Preservation:** The time step structure preserves the normalized nature of the data, maintaining the temporal relationships between consecutive prices.

**Autoencoder Input Preparation:** The same input sequence is used as both the input and target for the autoencoder, enabling it to learn the underlying patterns in the time series.

## Algorithms

The project utilizes an LSTM Autoencoder architecture:

Encoder: LSTM layer with 64 units

Decoder: RepeatVector layer, LSTM layer with 64 units, and TimeDistributed Dense layer

Dropout layers (20%) are used for regularization

## Evaluation Metrics

**Mean Absolute Error (MAE):** Used as both the loss function during training and the primary evaluation metric for model performance.

**Reconstruction Error:** Calculated as the MAE between the input sequence and the autoencoder's output, serving as the basis for anomaly detection.

**Anomaly Threshold:** Set at 0.65 (configurable) based on the distribution of reconstruction errors in the training set.

**True Positive Rate:** Although not explicitly calculated, it's represented by the correctly identified anomalies in the visualization.

**False Positive Rate:** While not numerically computed, it can be inferred from any incorrectly flagged anomalies in normal price movements.

**Detection Accuracy:** Qualitatively assessed through visual inspection of the anomaly scatter plot against the actual price movements.

**Model Convergence:** Evaluated by plotting the training and validation loss over epochs to ensure proper learning and avoid overfitting.

## Results and Discussion

**Model Training:** The LSTM Autoencoder was trained for 10 epochs, showing a general decrease in both training and validation loss, indicating successful learning of the underlying patterns in the SPX price data.

**Anomaly Detection:** The model successfully identified several anomalies in the test set, particularly during periods of high volatility or sudden price changes.

**Visualization of Results:** A scatter plot overlaid on the stock price chart clearly shows the detected anomalies, allowing for easy interpretation of the results in the context of price movements.

**Threshold Sensitivity:** The chosen threshold of 0.65 appears to effectively balance between detecting significant anomalies and avoiding false positives, though this could be further optimized.

**Detected Anomalies:** Notable anomalies were detected around February 2018, corresponding to a period of increased market volatility known as the "Volmageddon" event.

**Model Limitations:** While effective, the model may sometimes flag the start or end of trends as anomalies, suggesting room for improvement in distinguishing between anomalies and trend changes.

**Potential Applications:** The results demonstrate the potential of this approach for real-time market monitoring, risk management, and as a tool for identifying periods of unusual market behavior for further analysis.

# How to Use

Ensure you have the required libraries installed (TensorFlow, Pandas, NumPy, Matplotlib, Seaborn).

Place your 'spx.csv' file in the project directory.

Run the Jupyter notebook or Python script to train the model and detect anomalies.

Adjust the TIME_STEPS and THRESHOLD variables as needed for your specific use case.

## Conclusion

This project demonstrates the application of LSTM Autoencoders for anomaly detection in financial time series data. The model successfully identifies unusual price movements in the S&P 500 index, which could be valuable for market analysis and risk management. Future work could involve fine-tuning the model, experimenting with different architectures, or applying the technique to other financial instruments.

