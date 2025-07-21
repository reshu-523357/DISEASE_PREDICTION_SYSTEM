# Importing libraries
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Reading the train.csv by removing the last column since it's an empty column
DATA_PATH = "dataset/Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)

# Checking whether the dataset is balanced or not
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})

plt.figure(figsize=(18, 8))
sns.barplot(x="Disease", y="Counts", data=temp_df)
plt.xticks(rotation=90)
plt.show()

# Encoding the target value into numerical value using LabelEncoder
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24)

print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")

# Defining scoring metric for k-fold cross validation
def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))

# Initializing Models
models = {
    "SVC": SVC(),
    "Gaussian NB": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=18)
}

# Training and testing the models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    print(f"Model: {model_name}")
    print(f"Accuracy on train data: {accuracy_score(y_train, model.predict(X_train)) * 100:.2f}%")
    print(f"Accuracy on test data: {accuracy_score(y_test, preds) * 100:.2f}%")
    
    cf_matrix = confusion_matrix(y_test, preds)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cf_matrix, annot=True)
    plt.title(f"Confusion Matrix for {model_name} on Test Data")
    plt.show()

# Training the models on whole data
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)

final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

# Save the final models and data_dict
symptoms = X.columns.values
symptom_index = { " ".join([i.capitalize() for i in symptom.split("_")]): index for index, symptom in enumerate(symptoms) }

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

joblib_file = "disease_prediction_model.joblib"
joblib.dump((final_rf_model, data_dict), joblib_file)

# Print the symptom_index and predictions_classes for reference
print("Symptom Index Dictionary:")
print(symptom_index)
print("Predictions Classes:")
print(encoder.classes_)

# Define a function to predict disease based on symptoms
def predictDisease(symptoms_list):
    # Creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    print("Initial input data:", input_data)
    for symptom in symptoms_list:
        symptom = symptom.strip()
        if symptom in data_dict["symptom_index"]:
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
    print("Input data after setting symptoms:", input_data)

    # Convert input data to DataFrame with appropriate feature names
    input_data_df = pd.DataFrame([input_data], columns=symptoms)
    print("Input data DataFrame:\n", input_data_df)

    # Generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data_df)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data_df)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data_df)[0]]

    # Making final prediction by taking mode of all predictions
    final_prediction = pd.DataFrame([rf_prediction, nb_prediction, svm_prediction]).mode().iloc[0, 0]
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }
    return predictions


