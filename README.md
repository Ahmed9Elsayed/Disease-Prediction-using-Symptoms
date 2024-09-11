# **Disease Prediction Using Symptoms**

  

## Overview

This project aims to predict diseases based on the symptoms provided by users. The model is trained on a dataset that contains symptoms associated with various diseases, allowing the system to make predictions about the most likely disease based on the input symptoms.

## Table of Contents

- [Overview](#overview)
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#Conclusion)
- [Future Work](#future-work)
- [Contact Information](#Contact Information)


## Project Description

In this notebook, we explore a supervised learning approach for disease prediction using symptoms. The goal is to create a machine learning model that, given a set of symptoms, predicts the most likely disease. This system can assist in early disease detection, offering suggestions for further medical consultation.

The dataset consists of a list of symptoms associated with various human diseases. The model predicts the disease that matches the given symptoms.

## Dataset

The dataset contains rows with symptoms as input features and a corresponding disease as the target label. Each disease is associated with a specific set of symptoms.

### Data Source:

- The dataset is  publicly available on Kaggle:
- First dataset for Disease Prediction [Disease Prediction Using Machine Learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)
- second dataset for Disease description and Precautions to be taken[Disease Symptom Prediction](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset/code)
### Features:

- **Symptoms**: The features include multiple symptoms (e.g., fever, cough, fatigue) that are mapped to diseases.
- **Target**: The target label is the disease (e.g., hepatitis D, diabetes).

## **Installation**

  

To run this notebook, you need a Python environment with the following libraries installed:

  

- `pandas`

- `numpy`

- `scikit-learn`

- `matplotlib`

- `seaborn`

  

You can install the dependencies using:

  

`pip install pandas numpy scikit-learn matplotlib seaborn`

  

## **Usage**

  

1. **Load the Notebook:**

    - Open the notebook in Kaggle or your local Jupyter environment.

2. **Data Preprocessing:**

    - The notebook includes code to preprocess all issues in the two datasets

3. **Feature Engineering:**
    - used Mutual Information (MI) to capture any type of relationship between features and target

4. **Model Training:**

    Several machine learning models are used in this project, including:

	- **Logistic Regression**
	- **Decision Tree Classifier**
	- **Bernoulli's Naive Bayes**

	The models are trained using the symptom features, and the disease prediction is based on the likelihood derived from the symptoms.

  

5. **Evaluation:**

    - Performance metrics such as accuracy, precision, recall, and F1- score are used to evaluate the models.

    - Feature importance is analyzed to identify the key factors contributing to the predictions.

  

## **Results**
The model is evaluated using several metrics, including:

- **Accuracy**: Measures the percentage of correct predictions.
- **Precision**: Measures the number of true positives over the sum of true positives and false positives.
- **Recall**: Measures the number of true positives over the sum of true positives and false negatives.
- **F1-Score**: The harmonic mean of precision and recall.

These metrics are calculated using a test dataset that was not part of the training data.

- Test Data file was so small (41 row: a row for each disease), so Models Should be evaluated on a larger Test dataset  

- **Best Performing Models:**  **Bernoulli's Naive Bayes** with an accuracy of **99%** and **Multinomial Logistic Regression** with an accuracy of **100%**


## **Conclusion**

  

The notebook demonstrates that traditional machine learning models can effectively predict whether a celestial body is potentially hazardous to Earth. The Random Forest Classifier outperformed other models in accuracy and provides a good balance between interpretability and performance.

  

## **Future Work**

  

- Experiment with other advanced models, such as Gradient Boosting or Neural Networks.

- Improve feature engineering further more to enhance the model's accuracy.

  

## **Contact Information**

  

If you have any questions or suggestions to improve, please contact me via Kaggle or at ahmedelsayedtaha467@gmail.com.