import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Load models
def load_model(file_path):
    try:
        return pickle.load(open(file_path, 'rb'))
    except Exception as e:
        st.error(f"Error loading model {file_path}: {e}")
        return None

heart_disease_model = load_model('heart_disease_model.pkl')
diabetes_model = load_model('diabetes_model.pkl')
parkinsons_model = load_model('parkinsons_model.pkl')

# Prediction functions (unchanged)
def predict_heart_disease(features):
    try:
        features_array = np.array(features).reshape(1, -1)
        prediction = heart_disease_model.predict(features_array)
        prob = heart_disease_model.predict_proba(features_array)[0][1] if hasattr(heart_disease_model, "predict_proba") else None
        return prediction[0], prob
    except Exception as e:
        st.error(f"Heart Disease Prediction Error: {e}")
        return None, None

def predict_diabetes(features):
    try:
        features_array = np.array(features).reshape(1, -1)
        prediction = diabetes_model.predict(features_array)
        prob = diabetes_model.predict_proba(features_array)[0][1] if hasattr(diabetes_model, "predict_proba") else None
        return prediction[0], prob
    except Exception as e:
        st.error(f"Diabetes Prediction Error: {e}")
        return None, None

def predict_parkinsons(features):
    try:
        features_array = np.array(features).reshape(1, -1)
        prediction = parkinsons_model.predict(features_array)
        prob = parkinsons_model.predict_proba(features_array)[0][1] if hasattr(parkinsons_model, "predict_proba") else None
        return prediction[0], prob
    except Exception as e:
        st.error(f"Parkinson's Prediction Error: {e}")
        return None, None

# Symptom-based prediction (updated with additional symptoms)
def symptom_based_prediction(symptoms):
    # This is a simplified example. In a real application, you'd have a more sophisticated mapping.
    symptom_disease_map = {
        "Chest Pain": "Heart Disease",
        "Shortness of Breath": "Heart Disease",
        "Fatigue": "Heart Disease",
        "Irregular Heartbeat": "Heart Disease",
        "Dizziness": "Heart Disease",
        "Swollen Ankles": "Heart Disease",
        "Increased Thirst": "Diabetes",
        "Frequent Urination": "Diabetes",
        "Blurred Vision": "Diabetes",
        "Unexplained Weight Loss": "Diabetes",
        "Slow Wound Healing": "Diabetes",
        "Numbness in Hands or Feet": "Diabetes",
        "Tremor": "Parkinson's Disease",
        "Stiffness": "Parkinson's Disease",
        "Balance Problems": "Parkinson's Disease",
        "Slow Movement": "Parkinson's Disease",
        "Changes in Handwriting": "Parkinson's Disease",
        "Loss of Smell": "Parkinson's Disease"
    }
    
    disease_counts = {"Heart Disease": 0, "Diabetes": 0, "Parkinson's Disease": 0}
    for symptom in symptoms:
        if symptom in symptom_disease_map:
            disease_counts[symptom_disease_map[symptom]] += 1
    
    return max(disease_counts, key=disease_counts.get)

# Streamlit App
st.title("Health Disease Prediction App")

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode",
    ["Prediction Lab", "Manual Expertise", "Disease Graph", "More Info"])

if app_mode == "Prediction Lab":
    st.header("Prediction Lab")
    
    prediction_method = st.radio("Choose prediction method:", ["Symptom-based", "Detail-based"])
    
    if prediction_method == "Symptom-based":
        symptoms = st.multiselect("Select 5 symptoms you are experiencing:", 
            ["Chest Pain", "Shortness of Breath", "Fatigue", "Irregular Heartbeat", "Dizziness", "Swollen Ankles",
             "Increased Thirst", "Frequent Urination", "Blurred Vision", "Unexplained Weight Loss", "Slow Wound Healing", "Numbness in Hands or Feet",
             "Tremor", "Stiffness", "Balance Problems", "Slow Movement", "Changes in Handwriting", "Loss of Smell"])
        
        if len(symptoms) == 5:
            if st.button("Predict Disease"):
                predicted_disease = symptom_based_prediction(symptoms)
                st.success(f"Based on your symptoms, you might have: {predicted_disease}")
                st.info("Please proceed to fill in more details for a more accurate prediction.")
        else:
            st.warning("Please select exactly 5 symptoms.")
    
    if prediction_method == "Detail-based" or (prediction_method == "Symptom-based" and 'predicted_disease' in locals()):
        disease_to_predict = predicted_disease if 'predicted_disease' in locals() else st.selectbox("Select Disease to Predict:", ["Heart Disease", "Diabetes", "Parkinson's Disease"])
        
        if disease_to_predict == "Heart Disease":
            st.subheader("Heart Disease Prediction")
            thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=50, max_value=250)
            exang = st.selectbox("Exercise-Induced Angina (1 = Yes, 0 = No)", [1, 0])
            oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0)
            ca = st.selectbox("Number of Major Vessels (ca) (0-3)", [0, 1, 2, 3])
            cp_3 = st.selectbox("Chest Pain Type: CP_3 (0 = No, 1 = Yes)", [0, 1])
            age = st.number_input("Age", min_value=0, max_value=120)

            if st.button("Predict Heart Disease"):
                features = [thalach, exang, oldpeak, ca, cp_3, age]
                prediction, prob = predict_heart_disease(features)
                if prediction is not None:
                    result = "Positive for Heart Disease" if prediction == 1 else "Negative for Heart Disease"
                    st.success(f"Prediction: {result}")
                    if prob is not None:
                        st.info(f"Prediction Probability: {prob * 100:.2f}%")

        elif disease_to_predict == "Diabetes":
            st.subheader("Diabetes Prediction")
            glucose = st.number_input("Glucose Level (mg/dl)", min_value=50, max_value=300)
            insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=400)
            bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)
            age = st.number_input("Age", min_value=0, max_value=120)

            if st.button("Predict Diabetes"):
                features = [glucose, insulin, bmi, dpf, age]
                prediction, prob = predict_diabetes(features)
                if prediction is not None:
                    result = "Positive for Diabetes" if prediction == 1 else "Negative for Diabetes"
                    st.success(f"Prediction: {result}")
                    if prob is not None:
                        st.info(f"Prediction Probability: {prob * 100:.2f}%")

        elif disease_to_predict == "Parkinson's Disease":
            st.subheader("Parkinson's Disease Prediction")
            mdvp_fo = st.number_input("MDVP: Fo (Hz)", min_value=0.0, max_value=300.0)
            mdvp_fhi = st.number_input("MDVP: Fhi (Hz)", min_value=0.0, max_value=300.0)
            mdvp_flo = st.number_input("MDVP: Flo (Hz)", min_value=0.0, max_value=300.0)
            spread1 = st.number_input("Spread1", min_value=-10.0, max_value=10.0)
            spread2 = st.number_input("Spread2", min_value=-10.0, max_value=10.0)
            ppe = st.number_input("PPE", min_value=0.0, max_value=1.0)

            if st.button("Predict Parkinson's Disease"):
                features = [mdvp_fo, mdvp_fhi, mdvp_flo, spread1, spread2, ppe]
                prediction, prob = predict_parkinsons(features)
                if prediction is not None:
                    result = "Positive for Parkinson's Disease" if prediction == 1 else "Negative for Parkinson's Disease"
                    st.success(f"Prediction: {result}")
                    if prob is not None:
                        st.info(f"Prediction Probability: {prob * 100:.2f}%")

elif app_mode == "Manual Expertise":
    st.header("Manual Expertise")
    disease_to_predict = st.selectbox("Select Disease to Predict:", ["Heart Disease", "Diabetes", "Parkinson's Disease"])
    
    if disease_to_predict == "Heart Disease":
        st.subheader("Heart Disease Prediction")
        thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=50, max_value=250)
        exang = st.selectbox("Exercise-Induced Angina (1 = Yes, 0 = No)", [1, 0])
        oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0)
        ca = st.selectbox("Number of Major Vessels (ca) (0-3)", [0, 1, 2, 3])
        cp_3 = st.selectbox("Chest Pain Type: CP_3 (0 = No, 1 = Yes)", [0, 1])
        age = st.number_input("Age", min_value=0, max_value=120)

        if st.button("Predict Heart Disease"):
            features = [thalach, exang, oldpeak, ca, cp_3, age]
            prediction, prob = predict_heart_disease(features)
            if prediction is not None:
                result = "Positive for Heart Disease" if prediction == 1 else "Negative for Heart Disease"
                st.success(f"Prediction: {result}")
                if prob is not None:
                    st.info(f"Prediction Probability: {prob * 100:.2f}%")

    elif disease_to_predict == "Diabetes":
        st.subheader("Diabetes Prediction")
        glucose = st.number_input("Glucose Level (mg/dl)", min_value=50, max_value=300)
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=400)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)
        age = st.number_input("Age", min_value=0, max_value=120)

        if st.button("Predict Diabetes"):
            features = [glucose, insulin, bmi, dpf, age]
            prediction, prob = predict_diabetes(features)
            if prediction is not None:
                result = "Positive for Diabetes" if prediction == 1 else "Negative for Diabetes"
                st.success(f"Prediction: {result}")
                if prob is not None:
                    st.info(f"Prediction Probability: {prob * 100:.2f}%")

    elif disease_to_predict == "Parkinson's Disease":
        st.subheader("Parkinson's Disease Prediction")
        mdvp_fo = st.number_input("MDVP: Fo (Hz)", min_value=0.0, max_value=300.0)
        mdvp_fhi = st.number_input("MDVP: Fhi (Hz)", min_value=0.0, max_value=300.0)
        mdvp_flo = st.number_input("MDVP: Flo (Hz)", min_value=0.0, max_value=300.0)
        spread1 = st.number_input("Spread1", min_value=-10.0, max_value=10.0)
        spread2 = st.number_input("Spread2", min_value=-10.0, max_value=10.0)
        ppe = st.number_input("PPE", min_value=0.0, max_value=1.0)

        if st.button("Predict Parkinson's Disease"):
            features = [mdvp_fo, mdvp_fhi, mdvp_flo, spread1, spread2, ppe]
            prediction, prob = predict_parkinsons(features)
            if prediction is not None:
                result = "Positive for Parkinson's Disease" if prediction == 1 else "Negative for Parkinson's Disease"
                st.success(f"Prediction: {result}")
                if prob is not None:
                    st.info(f"Prediction Probability: {prob * 100:.2f}%")

elif app_mode == "Disease Graph":
    st.header("Disease Graph")
    
    disease = st.selectbox("Select Disease:", ["Heart Disease", "Diabetes", "Parkinson's Disease"])
    
    if disease == "Heart Disease":
        # Example data for heart disease
        normal_values = [150, 0, 1, 1, 0, 50]
        user_values = [thalach, exang, oldpeak, ca, cp_3, age] if 'thalach' in locals() else [0, 0, 0, 0, 0, 0]
        labels = ['Max Heart Rate', 'Exercise Angina', 'ST Depression', 'Major Vessels', 'Chest Pain Type', 'Age']
    elif disease == "Diabetes":
        # Example data for diabetes
        normal_values = [100, 30, 25, 0.5, 40]
        user_values = [glucose, insulin, bmi, dpf, age] if 'glucose' in locals() else [0, 0, 0, 0, 0]
        labels = ['Glucose', 'Insulin', 'BMI', 'DPF', 'Age']
    else:  # Parkinson's Disease
        # Example data for Parkinson's
        normal_values = [120, 150, 100, 0, 0, 0.2]
        user_values = [mdvp_fo, mdvp_fhi, mdvp_flo, spread1, spread2, ppe] if 'mdvp_fo' in locals() else [0, 0, 0, 0,0, 0]
        labels = ['MDVP:Fo', 'MDVP:Fhi', 'MDVP:Flo', 'Spread1', 'Spread2', 'PPE']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Normal human graph
    ax1.bar(labels, normal_values)
    ax1.set_title('Normal Human Values')
    ax1.set_ylabel('Value')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # User values graph
    ax2.bar(labels, user_values)
    ax2.set_title('Your Values')
    ax2.set_ylabel('Value')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    st.pyplot(fig)

elif app_mode == "More Info":
    st.header("More Information")
    
    st.subheader("About the System")
    st.write("""
    This health disease prediction system uses machine learning models to predict the likelihood of various diseases based on user-provided symptoms and health data. The system covers three major diseases: Heart Disease, Diabetes, and Parkinson's Disease.
    """)
    
    st.subheader("Diseases Covered")
    st.write("""
    1. Heart Disease: Cardiovascular diseases are the leading cause of death globally. Early detection can significantly improve outcomes.
    
    2. Diabetes: A chronic disease that affects how your body turns food into energy. Early diagnosis and treatment can prevent many complications.
    
    3. Parkinson's Disease: A progressive nervous system disorder that affects movement. Early diagnosis can help in managing symptoms effectively.
    """)
    
    st.subheader("Significance")
    st.write("""
    Early detection and prediction of these diseases can significantly improve treatment outcomes and quality of life for patients. This system aims to provide a preliminary assessment, but it's important to consult with a healthcare professional for accurate diagnosis and treatment.
    """)
    
    st.subheader("How to Use")
    st.write("""
    1. Prediction Lab: Choose between symptom-based or detail-based prediction. Follow the prompts to input your data.
    2. Manual Expertise: Directly input your health data for a specific disease prediction.
    3. Disease Graph: Visualize how your health data compares to normal values for each disease.
    4. More Info: Learn about the system, diseases covered, and their significance.
    """)

st.sidebar.info('This app is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.')
