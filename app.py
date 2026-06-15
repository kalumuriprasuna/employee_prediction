import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Function to set a background image and custom styles
def set_bg_and_styles():
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("https://www.transparenttextures.com/patterns/cubes.png");
        background-attachment: fixed;
        background-size: cover;
    }}
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    html, body, [class*="st-"], [class*="css-"] {{
        font-family: 'Roboto', sans-serif;
    }}
    .title-anim {{
        animation: fadeInDown 1s;
    }}
    @keyframes fadeInDown {{
        0% {{
            opacity: 0;
            transform: translateY(-20px);
        }}
        100% {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

# Load the trained model pipeline
pipeline = joblib.load("bestmodel_pipeline.pkl")

# For converting user-friendly input to the encoded values the model expects
# These mappings are based on the LabelEncoder transformation in the notebook
workclass_map = {
    'Private': 3, 'Self-emp-not-inc': 5, 'Local-gov': 1, 'Notlisted': 2,
    'State-gov': 6, 'Self-emp-inc': 4, 'Federal-gov': 0
}
marital_status_map = {
    'Never-married': 4, 'Married-civ-spouse': 2, 'Widowed': 6, 'Divorced': 0,
    'Separated': 5, 'Married-spouse-absent': 3, 'Married-AF-spouse': 1
}
occupation_map = {
    'Machine-op-inspct': 6, 'Farming-fishing': 4, 'Protective-serv': 10,
    'others': 14, 'Adm-clerical': 0, 'Exec-managerial': 3, 'Tech-support': 12,
    'Prof-specialty': 9, 'Other-service': 7, 'Craft-repair': 2, 'Transport-moving': 13,
    'Handlers-cleaners': 5, 'Sales': 11, 'Priv-house-serv': 8, 'Armed-Forces': 1
}
relationship_map = {
    'Own-child': 3, 'Husband': 0, 'Not-in-family': 1, 'Unmarried': 4, 'Wife': 5, 'Other-relative': 2
}
race_map = {'Black': 2, 'White': 4, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 0, 'Other': 3}
gender_map = {'Male': 1, 'Female': 0}
native_country_map = {
    'United-States': 39, '?': 0, 'Mexico': 26, 'Philippines': 30, 'Germany': 11, 'Puerto-Rico': 33,
    'Canada': 2, 'El-Salvador': 8, 'India': 19, 'Cuba': 5, 'England': 9, 'China': 3,
    'South': 35, 'Jamaica': 23, 'Italy': 22, 'Dominican-Republic': 6, 'Japan': 24,
    'Guatemala': 13, 'Poland': 31, 'Vietnam': 40, 'Columbia': 4, 'Haiti': 14,
    'Portugal': 32, 'Taiwan': 36, 'Iran': 20, 'Nicaragua': 27, 'Greece': 12, 'Peru': 29,
    'Ecuador': 7, 'France': 10, 'Ireland': 21, 'Hong': 17, 'Thailand': 37, 'Cambodia': 1,
    'Trinadad&Tobago': 38, 'Yugoslavia': 41, 'Outlying-US(Guam-USVI-etc)': 28,
    'Laos': 25, 'Scotland': 34, 'Honduras': 16, 'Hungary': 18, 'Holand-Netherlands': 15
}

set_bg_and_styles()

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("Employee Salary Classification")
st.markdown("This web application predicts whether an employee's income is greater than $50K or not.")

st.sidebar.header("Input Employee Details")

# Create UI elements for user input
age = st.sidebar.slider("Age", 17, 90, 30)
workclass_str = st.sidebar.selectbox("Workclass", list(workclass_map.keys()))
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", value=189778)
educational_num = st.sidebar.slider("Education Level (Num)", 1, 16, 9)
marital_status_str = st.sidebar.selectbox("Marital Status", list(marital_status_map.keys()))
occupation_str = st.sidebar.selectbox("Occupation", list(occupation_map.keys()))
relationship_str = st.sidebar.selectbox("Relationship", list(relationship_map.keys()))
race_str = st.sidebar.selectbox("Race", list(race_map.keys()))
gender_str = st.sidebar.selectbox("Gender", list(gender_map.keys()))
capital_gain = st.sidebar.number_input("Capital Gain", value=0)
capital_loss = st.sidebar.number_input("Capital Loss", value=0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
native_country_str = st.sidebar.selectbox("Native Country", list(native_country_map.keys()))


# Convert string inputs to their mapped integer values
workclass = workclass_map[workclass_str]
marital_status = marital_status_map[marital_status_str]
occupation = occupation_map[occupation_str]
relationship = relationship_map[relationship_str]
race = race_map[race_str]
gender = gender_map[gender_str]
native_country = native_country_map[native_country_str]


# Create a DataFrame from the inputs in the correct feature order
input_features = pd.DataFrame([[
    age, workclass, fnlwgt, educational_num, marital_status, occupation,
    relationship, race, gender, capital_gain, capital_loss,
    hours_per_week, native_country
]], columns=[
    'age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status',
    'occupation', 'relationship', 'race', 'gender', 'capital-gain',
    'capital-loss', 'hours-per-week', 'native-country'
])


st.write("### 🔎 Input Data")
st.write(input_features)

if st.button("Predict Salary Class"):
    # Use the pipeline to predict
    # The pipeline will handle scaling before prediction
    prediction = pipeline.predict(input_features)
    probability = pipeline.predict_proba(input_features)

    st.success(f"✅ Prediction: The individual's income is likely **{prediction[0]}**")
    st.write(f"Confidence: {np.max(probability)*100:.2f}%")
