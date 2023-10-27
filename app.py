 
import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load the saved PyCaret model
model = load_model('linear_discriminant_model')

# Define the Streamlit app
def main():
    # Front-end elements of the web page
    html_temp = """ 
    <div style ="background-color:cyan;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Triage Severity Prediction</h1> 
    </div> 
    """
      
    # Display the front-end aspect
    st.markdown(html_temp, unsafe_allow_html=True) 
      
    # Following lines create input fields for user data
    gender = st.selectbox('Gender', ("Female", "Male", "Transgender")) 
    age = st.number_input("Age")
    rr = st.number_input("Respiratory Rate")
    sat = st.number_input("Oxygen Saturation")
    pulse = st.number_input("Pulse Rate")
    bps = st.number_input("Systolic Blood Pressure")
    bpd = st.number_input("Diastolic Blood Pressure")
    gcs = st.number_input("Glasgow Coma Scale")
    tempt = st.number_input("Temperature")

    result = ""
      
    # When 'Predict' is clicked, make the prediction and display it
    if st.button("Predict"): 
        # Pre-processing user input
        if gender == "Female":
            gender = 0
        elif gender == 'Male':
            gender = 1
        else:
            gender = 2

        # Create a data frame with the user input
        user_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'rr': [rr],
            'sat': [sat],
            'pulse': [pulse],
            'bps': [bps],
            'bpd': [bpd],
            'gcs': [gcs],
            'tempt': [tempt]
        }, index=None)

        # Use the PyCaret model to generate predictions based on the user's input values
        pred = predict_model(model, data=user_data)
        
        # Create an empty DataFrame with the desired column names
        result_df = pd.DataFrame(columns=['prediction_label'])

        # Append the predicted label to the DataFrame
        result_df.loc[0] = pred['prediction_label'].iloc[0]

        # Assign the appropriate result based on the label value
        if result_df['prediction_label'][0] == 0:
               result = 'Do not require urgent attention'
        elif result_df['prediction_label'][0] == 1:
               result = 'Immediate care needed'
        else:
              result = 'Urgent need required but with high chance of recovery'
        st.success('Report Results: {}'.format(result))

if __name__ == '__main__':
    main()
