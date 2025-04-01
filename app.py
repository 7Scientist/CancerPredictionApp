import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("Mybest_model.pkl")

# Title of the web app
st.title("Cancer Prediction System")

# Input fields for user
st.write("Enter the patient details below:")

# Example input fields (adjust based on dataset features)
feature1 = st.number_input("Index", min_value=0.0, format="%.2f")
feature2 = st.number_input("Patient Id", min_value=0.0, format="%.2f")
feature3 = st.number_input("Gender: 1- Male, 2 - Female", min_value=0.0, format="%.2f")
feature4 = st.number_input("Age", min_value=0.0, format="%.2f")
feature5 = st.number_input("Dust Allergy", min_value=0.0, format="%.2f")
feature6 = st.number_input("Occupation", min_value=0.0, format="%.2f")
feature7 = st.number_input("Genetic Risk", min_value=0.0, format="%.2f")
feature8 = st.number_input("Chronic Lung Disease", min_value=0.0, format="%.2f")
feature9 = st.number_input("Balanced Diet", min_value=0.0, format="%.2f")
feature10 = st.number_input("Obesity", min_value=0.0, format="%.2f")
feature11 = st.number_input("Smoking", min_value=0.0, format="%.2f")
feature12 = st.number_input("Passive Smoker", min_value=0.0, format="%.2f")
feature13 = st.number_input("Chest Pain", min_value=0.0, format="%.2f")
feature14 = st.number_input("Coughing", min_value=0.0, format="%.2f")
feature15 = st.number_input("Fatigue", min_value=0.0, format="%.2f")
feature16 = st.number_input("Weight Loss", min_value=0.0, format="%.2f")
feature17 = st.number_input("Shortness", min_value=0.0, format="%.2f")
feature18 = st.number_input("Wheezing", min_value=0.0, format="%.2f")
feature19 = st.number_input("Swallowing Difficulty", min_value=0.0, format="%.2f")
feature20 = st.number_input("Frequent Cold", min_value=0.0, format="%.2f")
feature21 = st.number_input("Dry Cough", min_value=0.0, format="%.2f")
feature22 = st.number_input("Snoring", min_value=0.0, format="%.2f")
feature23 = st.number_input("Clubbing of Finger Nails", min_value=0.0, format="%.2f")
feature24 = st.number_input("Air Pollution", min_value=0.0, format="%.2f")
feature25 = st.number_input("Alcohol use", min_value=0.0, format="%.2f")

# Button to predict
if st.button("Predict"):
    # Prepare input data
    data = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8,
                feature9, feature10, feature11, feature12, feature13, feature14, feature15, feature16,
                feature17, feature18, feature19, feature20, feature21, feature22, feature23, feature24, feature25]])

    # Make prediction
    prediction = model.predict(data)
    
    # Display result
    result = "🛑 Cancer Detected" if prediction[0] == 1 else "✅ No Cancer"
    st.subheader(f"Prediction: {result}")

