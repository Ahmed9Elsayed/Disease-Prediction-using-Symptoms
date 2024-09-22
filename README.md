# **Disease Prediction Using Symptoms**

  

## Overview

This project aims to predict diseases based on the symptoms provided by users. The model is trained on a dataset that contains symptoms associated with various diseases, allowing the system to make predictions about the most likely disease based on the input symptoms.

## Table of Contents

- [Overview](#overview)
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Inference](#inference)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Contact Information](#contact-information)


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

2. **Data Preprocessing and EDA:**

    - The notebook includes code to preprocess all issues in the two datasets
    - also some EDA code to understand the data further

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

    
## **Inference:**
## Inference Steps:

- Patient enters the pain he is experiencing in a text box
  
- Patient's text will be sent to `Symptom_Extractor.py` to be analyzed and return a 0/1 vector of symptoms model was trained on, where 1 is the presence of symptoms and 0 is the absence
  
- We will use a sentence transformer model in `Symptom_Extracor.py` to capture symptoms patient included in his text and convert it into a vector using cosine similarity
  
- Symptoms vector will be send as a parameter to disease-predictor function from `Disease_Predictor.py` along with the pipelined model to predict the disease, Disease Description dataframe to provide description for the predicted disease and Disease Precautions dataframe to provide precautions for the predicted disease

- The final output will be the most likely disease that patient migh be experiencing with possibilty of getting precautions or describtion for it
  
## Setup instructions before Inference

## kaggle-related environment setup: you should run this code before any thing when using kaggle

```
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

## first: install sentence transformers for symptom extractor file

`%pip install sentence_transformers`

  

## second: importing files
#### Load model pipeline file: upload you model pipeline file in the same directory 
```
import joblib

# Load the pipeline from the file
loaded_modelName_pipeline = joblib.load('path/to/model-pipeline-pkl-file')
```
 
### if you work in kaggle you will need to run this code, after uploading the 'symptom extractor' and 'disease predictor' files as a dataset in kaggle's input
##### kaggle-code to import necessary functions
```
import sys


sys.path.append('/kaggle/input/name of the dataset containing Symptoms_Extractor file')
from Symptoms_Extractor import analyze_patient_text

sys.path.append('/kaggle/input/name of the dataset containing Disease_Predictor file')
from Disease_Predictor import disease_predictor

```

##### if you work locally or in colab just upload the files in the same directory of the inference file
##### colab or locally code to import necessary functions
```
from Symptoms_Extractor import analyze_patient_text
from Disease_Predictor import disease_predictor
```
## kaggle related issue

In the time this notebook is created, there is currently a problem with using nltk resources on kaggle so if you encountered this error when running inference code:
```
LookupError: 
**********************************************************************
  Resource 'corpora/wordnet' not found.  Please use the NLTK
  Downloader to obtain the resource:  >>> nltk.download()
  Searched in:
    - '/root/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/local/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/local/lib/nltk_data'
**********************************************************************
```
you need to run this line of code first then rerun the inference code
```
! cp -rf /usr/share/nltk_data/corpora/wordnet2022 /usr/share/nltk_data/corpora/wordnet # temp fix for lookup error.
```

  


## Inference Code
#### setting data up
```
import numpy as np 
import pandas as pd
df_prec = pd.read_csv('path/to/disease-describtion-dataset')
df_desc = pd.read_csv('path/to/disease-precautions-dataset')
# preprocessing related to the datasets

# describtions processing: we need to set this disease's name as in the training data is 'Dimorphic hemmorhoids(piles)' and in description data is 'Dimorphic hemorhoids(piles)' only on m in hemmorhoids so it needs to be handled to avoid issues in printing the description 

df_desc.iloc[16,0] = 'Dimorphic hemmorhoids(piles)'

df_desc.set_index('Disease',inplace = True)

# precations processing
df_prec.replace(float('nan'),'No Precaution',inplace = True)

# Because diabetes in Disease column is saved as 'Diabetes ' , and this will make a lot of trouble after if not handled here
df_prec['Disease'] = df_prec['Disease'].astype(str).str.strip()

df_prec.set_index('Disease',inplace = True)

```
#### Symptoms vector extraction
```
# list of symptoms that exists in the dataset which the model was trained on
symptoms = symptoms = [
    "Itching", "Skin rash", "Nodal skin eruptions", "Continuous sneezing", "Shivering", 
    "Chills", "Joint pain", "Stomach pain", "Acidity", "Ulcers on tongue", 
    "Muscle wasting", "Vomiting", "Burning sensation during urination", "Spotting in urination", 
    "Fatigue", "Weight gain", "Anxiety", "Cold hands and feet", "Mood swings", 
    "Weight loss", "Restlessness", "Lethargy", "Patches in throat", "Irregular sugar levels", 
    "Cough", "High fever", "Sunken eyes", "Breathlessness", "Sweating", 
    "Dehydration", "Indigestion", "Headache", "Yellowish skin", "Dark urine", 
    "Nausea", "Loss of appetite", "Pain behind the eyes", "Back pain", 
    "Constipation", "Abdominal pain", "Diarrhea", "Mild fever", "Yellow urine", 
    "Yellowing of eyes", "Acute liver failure", "Fluid overload", "Swelling of stomach", 
    "Swollen lymph nodes", "Malaise", "Blurred and distorted vision", "Phlegm", 
    "Throat irritation", "Redness of eyes", "Sinus pressure", "Runny nose", 
    "Congestion", "Chest pain", "Weakness in limbs", "Fast heart rate", 
    "Pain during bowel movements", "Pain in anal region", "Bloody stool", "Irritation in anus", 
    "Neck pain", "Dizziness", "Cramps", "Bruising", "Obesity", "Swollen legs", 
    "Swollen blood vessels", "Puffy face and eyes", "Enlarged thyroid", "Brittle nails", 
    "Swollen extremities", "Excessive hunger", "Extra-marital contacts", "Drying and tingling lips", 
    "Slurred speech", "Knee pain", "Hip joint pain", "Muscle weakness", "Stiff neck", 
    "Swelling joints", "Movement stiffness", "Spinning movements", "Loss of balance", 
    "Unsteadiness", "Weakness of one side of the body", "Loss of smell", "Bladder discomfort", 
    "Foul smell of urine", "Continuous feeling of needing to urinate", "Passage of gases", 
    "Internal itching", "Toxic look (typhoid-like symptoms)", "Depression", "Irritability", 
    "Muscle pain", "Altered sensorium", "Red spots over the body", "Belly pain", 
    "Abnormal menstruation", "Discolored patches on the skin", "Watering from the eyes", 
    "Increased appetite", "Polyuria (frequent urination)", "Family history (of disease)", 
    "Mucoid sputum", "Rusty sputum", "Lack of concentration", "Visual disturbances", 
    "Receiving blood transfusion", "Receiving unsterile injections", "Coma", 
    "Stomach bleeding", "Distention of abdomen", "History of alcohol consumption", 
    "Fluid overload", "Blood in sputum", "Prominent veins on calf", "Palpitations", 
    "Painful walking", "Pus-filled pimples", "Blackheads", "Scarring", "Skin peeling", 
    "Silver-like dusting (skin condition)", "Small dents in nails", "Inflammatory nails", 
    "Blisters", "Red sores around nose", "Yellow crust ooze"
]

patient_text = input('Type what you are experiencing of pain:\n')

# Call the function
result = analyze_patient_text(patient_text, symptoms)
symptom_vector = result['presence_vector']
symptom_arr = np.array(symptom_vector).reshape(1,-1)
```

  
#### Disease Prediction along with provided Describtion or precautions
```
disease_predictor(symptom_arr,loaded_model_pipeline,df_desc,df_prec)
```

## Inference Example on kaggle
`inference.ipynb` contain an example on how to do inference on kaggle

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

  

If you have any questions or suggestions to improve, please contact me via Kaggle: https://www.kaggle.com/ahmed321abozeid or at ahmedelsayedtaha467@gmail.com.
