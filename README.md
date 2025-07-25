//Disease Prediction System using Machine Learning


Overview

The Disease Prediction System is a machine learning-based web application that predicts possible diseases based on user-input symptoms. The project uses Support Vector Machine (SVM), Naive Bayes, and Random Forest algorithms to train the model, and the Random Forest model is selected as the final model due to its higher accuracy. The frontend is built using Streamlit, and a user authentication system is implemented with SQLite and Passlib for secure login and signup.

Features

Predicts disease based on user-selected symptoms.

Trains and compares three machine learning models: SVM, Naive Bayes, and Random Forest.

Uses Random Forest as the final model due to its high accuracy.

Provides a user-friendly web interface built with Streamlit.

Secure login and signup functionality using SQLite database and Passlib for password hashing.

Visualizes model performance with confusion matrices.

Easy-to-use dropdowns and multiselect for symptoms.

Project Flow

Data Preparation:

Load the dataset (Training and Testing CSV files).

Clean data and remove empty columns.

Encode disease names using LabelEncoder.

Convert symptoms to binary features (0 or 1).

Model Training:

Train three models: SVM, Naive Bayes, and Random Forest.

Evaluate models using accuracy score and confusion matrix.

Select Random Forest as the final model.

Save the trained model and data dictionary using Joblib.

Backend Logic:

Load the saved Random Forest model using Joblib.

Process user inputs (symptoms) into binary format.

Predict the disease and map label back to disease name.

Frontend Interface:

Built using Streamlit.

Login and signup pages handled by main.py.

Disease prediction page handled by app.py.

Displays predictions using interactive UI components.

User Authentication:

SQLite database (users.db) stores user credentials.

Passlib used for secure password hashing and verification.

Technologies and Libraries

Languages: Python

Frontend Framework: Streamlit

Database: SQLite

Machine Learning: scikit-learn (SVM, GaussianNB, RandomForestClassifier)

Other Libraries: pandas, numpy, seaborn, matplotlib, joblib, passlib

Setup Instructions

1. Create and Activate a Python Environment

# Create a virtual environment
python -m venv env

# Activate the environment (Windows)
env\\Scripts\\activate

# Activate the environment (Linux/Mac)
source env/bin/activate

2. Install Required Libraries

pip install -r requirements.txt

3. Run the Application

streamlit run main.py

4. Using the App

Sign up or log in to access the disease prediction page.

Select gender, age, and symptoms, then click Predict to see the result.

Future Enhancements

Add advanced models like Deep Learning.

Include a larger medical dataset with severity levels.

Integrate with FastAPI or Flask for REST API support.

Deploy on cloud platforms like Heroku or AWS.

Add voice-based symptom input and multilingual support.

Author

Developed by Shaik Reshma as part of a machine learning project to predict diseases and create a user-friendly healthcare application.
