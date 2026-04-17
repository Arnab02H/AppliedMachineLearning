<div align="center">

# Course: Applied Machine Learning  
# Student: Arnab Bera (MDS202409)  
# SMS Spam Classification  

</div>
--

## Overview

This repository contains two assignments focused on building a reproducible and production-oriented machine learning workflow for SMS Spam Classification.  

The project covers:

- End-to-end ML prototype development  
- Data Version Control (DVC)  
- Experiment Tracking and Model Versioning using MLflow  

Dataset used for Assignment-01:  
UCI SMS Spam Collection Dataset  
https://archive.ics.uci.edu/ml/datasets/sms+spam+collection  


---

# Assignment-01: ML Prototype Development

## Problem Statement

Build a complete prototype for SMS Spam Classification that:

1. Loads the SMS spam dataset from a given file path  
2. Preprocesses the text data (if necessary)  
3. Splits the dataset into train, validation, and test sets  
4. Saves the splits as:
   - train.csv  
   - validation.csv  
   - test.csv  

Then implement a training pipeline that:

1. Fits a model on the training data  
2. Scores the model on given data  
3. Evaluates predictions using appropriate metrics  
4. Validates the model  
5. Scores on both training and validation sets  
6. Fine-tunes hyperparameters (if required)  
7. Trains three benchmark models  
8. Evaluates all three models on the test set  
9. Selects the best model based on a suitable metric (e.g., AUCPR)  

The goal is to design a structured, modular, and reproducible ML workflow following good experiment design practices.


---

# Assignment-02: Experiment Tracking and Version Control

## Problem Statement

Extend Assignment-01 by introducing proper data and model version control.

## Data Version Control (DVC)

In prepare.ipynb:

1. Track raw_data.csv using DVC  
2. Track train.csv, validation.csv, and test.csv  
3. Change the random seed and regenerate new splits  
4. Track the updated dataset versions  
5. Checkout the first version using DVC and print the distribution of the target variable (number of 0s and 1s) in:
   - train.csv  
   - validation.csv  
   - test.csv  
6. Checkout the updated version and again print the target distribution  
7. (Bonus) Configure Google Drive as remote storage to decouple compute and storage  

The goal is to demonstrate reproducible data pipelines with proper versioning.


## Model Versioning and Experiment Tracking (MLflow)

In train.ipynb:

1. Track experiments using MLflow  
2. Log:
   - Parameters  
   - Metrics  
   - Models  
3. Train and register three benchmark models  
4. Log evaluation metric AUCPR for each model  
5. Compare models using MLflow  
6. Select the best model based on AUCPR  

The goal is to demonstrate structured experiment tracking, model comparison, and model version management following industry best practices.

# Assignment 3: Testing & Model Serving

This assignment focuses on implementing **model scoring, unit testing, integration testing, and Flask-based model serving** for a trained machine learning model.

---

# 1. Score Function Implementation

In `score.py`, implement a function that evaluates a trained model on a given text input.

### Function Signature

```python
def score(text: str, model, threshold: float):
```

### Requirements

The function should:

- Accept a **text input**
- Use a **trained sklearn model**
- Apply a **classification threshold**
- Return:
  - **prediction** (boolean or binary output)
  - **propensity score** (probability value)

---

# 2. Unit Testing

In `test.py`, implement a **unit test function `test_score()`** to verify the correctness of the `score()` function.

You may reload the **best trained model saved during experiments in `train.ipynb`** (saved in `.pkl` or `.joblib` format).

### Test Cases to Consider

The unit test should verify:

- **Smoke Test**
  - The function runs successfully without crashing.

- **Format Test**
  - The input and output formats/types are correct.

- **Sanity Checks**
  - Prediction value should be either **0 or 1**.
  - Propensity score should be between **0 and 1**.

- **Edge Cases**
  - If threshold = **0**, prediction should always be **1**.
  - If threshold = **1**, prediction should always be **0**.

- **Typical Inputs**
  - For an **obvious spam text**, prediction should be **1**.
  - For an **obvious non-spam text**, prediction should be **0**.

---

# 3. Flask Model Serving

In `app.py`, implement a **Flask API endpoint** to serve the model.

### Endpoint

```
/score
```

### Requirements

- Accept **POST requests**
- Receive **text input**
- Call the `score()` function
- Return a **JSON response** containing:
  - prediction
  - propensity score

---

# 4. Integration Testing

In `test.py`, implement an **integration test function `test_flask()`**.

### Integration Test Requirements

The test should:

1. **Launch the Flask application** using a command line call (e.g., `os.system()`).
2. **Send a request to the local endpoint** (`localhost`).
3. **Validate the response** returned by the API.
4. **Terminate the Flask application** after testing.

---

# 5. Coverage Report

Generate a **coverage report** for both **unit tests and integration tests**.

### Requirements

- Use **pytest** to execute tests.
- Generate a **coverage report**.
- Save the output in a file named:

```
coverage.txt
```

---
# Assignment-05: Transfer Learning

## Problem Statement

This assignment focuses on applying **Transfer Learning** techniques on both **image data (CNNs)** and **text data (Transformers)** to build robust classification models.

The objective is to understand how pre-trained models can be fine-tuned for domain-specific tasks and evaluated using proper metrics.

---

## Part 1: Transfer Learning for Image Classification (CNN)

### Task

- Collect a custom dataset:
  - ~100 images of **chickens**
  - ~100 images of **ducks**

- Build an image classification model using:
  - A **pre-trained Convolutional Neural Network (CNN)** (e.g., ResNet)

- Perform:
  - Data preprocessing and augmentation  
  - Train-validation-test split  
  - Transfer learning (freeze base layers)  
  - Fine-tuning (unfreeze deeper layers)

- Evaluate the model using:
  - **Classification Report** (Precision, Recall, F1-score)  
  - **Confusion Matrix** (optional)

---

### Objective

- Learn feature reuse from ImageNet-trained models  
- Improve performance using fine-tuning  
- Understand overfitting and generalization in CNNs  

---

### References

- https://www.analyticsvidhya.com/blog/2021/07/step-by-step-guide-for-image-classification-on-custom-datasets/  
- https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html  
- https://www.learnpytorch.io/  

---

## Part 2: Transfer Learning for Text Classification (Transformers)

### Task

- Download dataset:
  - Sentiment Analysis Dataset  
  - https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset  

- Build a sentiment classifier to classify text into:
  - **Positive**
  - **Neutral**
  - **Negative**

- Use a pre-trained Transformer model (e.g., **BERT**)

- Perform:
  - Tokenization using HuggingFace Transformers  
  - Fine-tuning on custom dataset  
  - Model training and validation  

- Evaluate using:
  - **Classification Report**  
  - Accuracy, Precision, Recall, F1-score  

---
