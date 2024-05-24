import streamlit as st
import numpy as np
import pickle

# Loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def diabetes_prediction(input_data):
    # Ensure input data is numeric
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    
    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    print(input_data_reshaped)
    
    # Make prediction
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    st.title('Diabetes Prediction App')

    # User input fields
    Gender = st.selectbox('Gender', ['Male', 'Female','Other'])
    Age = st.number_input('Age', min_value=0)
    Hypertension = st.selectbox('Do you have Hypertension?', ['No', 'Yes'])
    Heart_Disease = st.selectbox('Do you have Heart Disease?', ['No', 'Yes'])
    Smoking_History = st.selectbox('Select Smoking History', ['No Info', 'Current', 'Ever', 'Former', 'Never', 'Not Current'])
    BMI = st.number_input('BMI', min_value=0.0)
    HbA1c_Level = st.number_input('HbA1c Level', min_value=0.0)
    Blood_Glucose_Level = st.number_input('Blood Glucose Level', min_value=0.0)

    # Mapping categorical data to numeric values
    gender_mapping = {'Female': 0,'Male': 1, 'Other':2}
    hypertension_mapping = {'No': 0, 'Yes': 1}
    heart_disease_mapping = {'No': 0, 'Yes': 1}
    smoking_history_mapping = {'No Info': 0, 'Current': 1, 'Ever': 2, 'Former': 3, 'Never': 4, 'Not Current': 5}

    # Convert categorical inputs to numeric
    Gender = gender_mapping[Gender]
    Hypertension = hypertension_mapping[Hypertension]
    Heart_Disease = heart_disease_mapping[Heart_Disease]
    Smoking_History = smoking_history_mapping[Smoking_History]

    # Code for prediction
    diagnosis = ''

    # Creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Gender, Age, Hypertension, Heart_Disease, Smoking_History, BMI, HbA1c_Level, Blood_Glucose_Level])
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()
