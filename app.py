import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load your machine learning model and data_dict
model, data_dict = joblib.load('disease_prediction_model.joblib')

# Load the dataset to extract symptoms
DATA_PATH = "dataset/Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)

# Extract symptom list from dataset columns
symptoms_list = list(data.columns[:-1])  # Exclude the 'prognosis' column

# Map dataset columns to user-friendly symptom names
symptom_index = { " ".join([i.capitalize() for i in symptom.split("_")]): symptom for symptom in symptoms_list }

# Function to make predictions
def predict_disease(symptoms):
    predictions_classes = data_dict["predictions_classes"]
    
    # Initialize input data with zeros
    input_data = [0] * len(symptom_index)
    
    # Set input data based on symptoms
    for symptom in symptoms:
        symptom = symptom.strip()
        if symptom in symptom_index:
            index = symptoms_list.index(symptom_index[symptom])
            input_data[index] = 1

    # Reshape the input data and make prediction
    input_data = np.array(input_data).reshape(1, -1)
    
    # Generate individual outputs
    rf_prediction = predictions_classes[model.predict(input_data)[0]]
    
    return rf_prediction

def main():
    st.title("Disease Prediction System")

    st.header("Enter Patient Data")

    # Gender input
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    # Age input
    age = st.number_input("Age", min_value=0, max_value=120)

    # Symptoms input
    symptoms = st.multiselect("Symptoms", list(symptom_index.keys()))

    # Create a button for making predictions
    if st.button("Predict"):
        prediction = predict_disease(symptoms)
        st.success(f"The predicted disease is: {prediction}")

if __name__ == "__main__":
    main()

