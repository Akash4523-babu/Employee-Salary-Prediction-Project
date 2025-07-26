import streamlit as st
import pandas as pd
import joblib

# Set Streamlit page configuration
st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

# --- Load the trained model and preprocessor ---
try:
    model = joblib.load("best_model.pkl")
    preprocessor = joblib.load("preprocessing.pkl")
    st.success("Model and preprocessor loaded successfully!")
except FileNotFoundError:
    st.error("Error: 'best_model.pkl' or 'preprocessing.pkl' not found. Please ensure they are in the same directory as app.py.")
    st.stop() # Stop the app if essential files are missing
except Exception as e:
    st.error(f"An error occurred loading the model/preprocessor: {e}")
    st.stop()


st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns **>50K** or **≤50K** based on input features.")

# --- Sidebar inputs for user to provide details ---
st.sidebar.header("Input Employee Details")

# Define ranges and options based on your EDA and original dataset unique values
# Numerical inputs
age = st.sidebar.slider("Age", 18, 75, 30)
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40)
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, max_value=99999, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, max_value=4356, value=0) # Max based on your EDA
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", min_value=12285, max_value=1490400, value=190000) # Range based on EDA
educational_num = st.sidebar.slider("Educational Number", 5, 16, 9) # Range based on your EDA

# Categorical inputs (using selectbox for string values)
workclass = st.sidebar.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Local-gov", "Others", "State-gov",
    "Self-emp-inc", "Federal-gov"
])
education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "Doctorate", "HS-grad", "Assoc-acdm", "Some-college",
    "11th", "10th", "7th-8th", "Prof-school", "9th", "12th", "Assoc-voc"
])
marital_status = st.sidebar.selectbox("Marital Status", [
    "Married-civ-spouse", "Never-married", "Divorced", "Separated",
    "Widowed", "Married-spouse-absent", "Married-AF-spouse"
])
occupation = st.sidebar.selectbox("Job Role", [
    "Prof-specialty", "Craft-repair", "Exec-managerial", "Adm-clerical", "Sales",
    "Other-service", "Machine-op-inspct", "Others", "Transport-moving",
    "Handlers-cleaners", "Farming-fishing", "Tech-support", "Protective-serv",
    "Priv-house-serv"
])
relationship = st.sidebar.selectbox("Relationship", [
    "Husband", "Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"
])
race = st.sidebar.selectbox("Race", [
    "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "Mexico", "Others", "Philippines", "Germany", "Puerto-Rico",
    "Canada", "El-Salvador", "India", "Cuba", "England", "China", "South",
    "Jamaica", "Italy", "Dominican-Republic", "Japan", "Guatemala", "Poland",
    "Vietnam", "Columbia", "Haiti", "Portugal", "Taiwan", "Iran", "Nicaragua",
    "Greece", "Peru", "Ecuador", "France", "Ireland", "Thailand", "Hong",
    "Cambodia", "Trinadad&Tobago", "Laos", "Outlying-US(Guam-USVI-etc)",
    "Yugoslavia", "Scotland", "Honduras", "Hungary", "Holand-Netherlands"
])


# --- Build input DataFrame for single prediction ---
# IMPORTANT: The order of columns in this DataFrame MUST exactly match the original 'X' DataFrame
# used to train the preprocessor in the notebook.
# This order was: ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status',
#                   'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
#                   'hours-per-week', 'native-country']
input_data = {
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'education': [education],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
}

input_df = pd.DataFrame(input_data)

st.write("### 🔎 Input Data")
st.write(input_df)

# --- Predict button for single prediction ---
if st.button("Predict Salary Class"):
    try:
        # Preprocess the input data
        processed_input = preprocessor.transform(input_df)
        
        # Make prediction
        prediction = model.predict(processed_input)
        
        st.success(f"✅ Prediction: Employee Salary is {prediction[0]}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please check the input values and ensure the model is correctly loaded.")

# --- Batch prediction ---
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    
    # Ensure the batch_data columns match the training data columns exactly
    # You might want to add more robust error handling or data cleaning for batch uploads
    # e.g., checking for missing columns, handling '?' values, etc.
    
    try:
        # Preprocess the batch data
        processed_batch_data = preprocessor.transform(batch_data)
        
        # Make predictions
        batch_preds = model.predict(processed_batch_data)
        batch_data['PredictedClass'] = batch_preds
        
        st.write("✅ Predictions:")
        st.write(batch_data.head())
        
        # Provide download button for predictions
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Predictions CSV",
            csv,
            file_name='predicted_classes.csv',
            mime='text/csv'
        )
    except Exception as e:
        st.error(f"An error occurred during batch prediction: {e}")
        st.info("Please ensure the uploaded CSV has the correct columns and data format as the original dataset.")
