

import nltk
from nltk.corpus import wordnet as wn

import warnings

nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('wordnet2022')


import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util

# Load required models (loaded once globally to avoid reloading every time)
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# Function to get synonyms for each symptom using WordNet
def get_synonyms(symptom):
    synonyms = set()
    for syn in wordnet.synsets(symptom):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return synonyms

# Main function to analyze patient text and symptoms
def analyze_patient_text(patient_text, symptoms):
    """
    Analyze the patient text to find if any symptom is present.
    
    :param patient_text: String, the text spoken by the patient
    :param symptoms: List of symptoms to analyze
    :return: Dictionary with symptoms and the presence vector (1 if present, 0 if absent)
    """
    # Preprocess symptoms to expand them with synonyms
    symptom_synonyms = {symptom: get_synonyms(symptom) | {symptom} for symptom in symptoms}

    # Convert patient text to lowercase for easier matching
    patient_text_lower = patient_text.lower()

    # Tokenize the patient's text into words for comparison
    patient_words = patient_text_lower.split()

    # Encode all symptoms and patient words using sentence embeddings
    symptom_embeddings = similarity_model.encode(symptoms)
    patient_embeddings = similarity_model.encode(patient_words)

    # Set a similarity threshold
    similarity_threshold = 0.6

    # Create the vector based on the presence of symptoms or their synonyms
    vector = []
    for symptom, synonyms in symptom_synonyms.items():
        symptom_present = False
        
        # Check if any synonym matches the patient's text exactly
        for synonym in synonyms:
            if synonym in patient_text_lower:
                symptom_present = True
                break
        
        # If no exact match found, use semantic similarity
        if not symptom_present:
            symptom_embedding = similarity_model.encode([symptom])[0]
            # Calculate similarity between symptom and each patient word
            for patient_embedding in patient_embeddings:
                similarity = util.pytorch_cos_sim(symptom_embedding, patient_embedding).item()
                if similarity > similarity_threshold:
                    symptom_present = True
                    break
        
        vector.append(1 if symptom_present else 0)

    # Return the results as a dictionary
    result = {
        "symptoms": symptoms,
        "presence_vector": vector
    }
    
    return result
