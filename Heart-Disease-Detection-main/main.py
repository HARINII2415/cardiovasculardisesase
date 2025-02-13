import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import plotly.express as px


# Load and preprocess data
def load_data():
    """Load the heart disease dataset and perform initial preprocessing"""
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

    df = pd.read_csv(r'C:\Users\harin\Downloads\Heart-Disease-Detection-main\Heart-Disease-Detection-main\heart.csv')
    return df


def preprocess_data(df):
    """Preprocess the data for model training"""
    # Handle missing values
    df = df.fillna(df.mean())

    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train):
    """Train the Random Forest model"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def create_streamlit_app():
    """Create the Streamlit web interface"""
    st.set_page_config(page_title="Heart Disease Detection System", layout="wide")

    st.title("Heart Disease Detection System")
    st.write("Enter patient information to predict heart disease risk")

    # Create input form
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type",
                          ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
        restecg = st.selectbox("Resting ECG Results",
                               ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.0, value=0.0)

    with col3:
        slope = st.selectbox("Slope of Peak Exercise ST Segment",
                             ["Upsloping", "Flat", "Downsloping"])
        ca = st.number_input("Number of Major Vessels", min_value=0, max_value=4, value=0)
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    # Convert categorical inputs to numerical
    sex = 1 if sex == "Male" else 0
    cp = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
    fbs = 1 if fbs == "Yes" else 0
    restecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
    exang = 1 if exang == "Yes" else 0
    slope = ["Upsloping", "Flat", "Downsloping"].index(slope)
    thal = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1

    # Create feature array
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                          exang, oldpeak, slope, ca, thal]])

    if st.button("Predict"):
        # Load the model and scaler
        try:
            model = train_model(X_train_scaled, y_train)

            # Scale the input features
            features_scaled = scaler.transform(features)

            # Make prediction
            prediction = model.predict(features_scaled)
            probability = model.predict_proba(features_scaled)

            # Display results
            st.header("Results")
            if prediction[0] == 1:
                st.error("⚠ High Risk of Heart Disease")
                st.write(f"Probability: {probability[0][1]:.2%}")
            else:
                st.success("✅ Low Risk of Heart Disease")
                st.write(f"Probability: {probability[0][0]:.2%}")

            # Display feature importance
            feature_importance = pd.DataFrame({
                'Feature': columns[:-1],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            st.subheader("Feature Importance")
            fig = px.bar(feature_importance, x='Feature', y='Importance',
                         title='Feature Importance in Prediction')
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    # Load and preprocess data
    df = load_data()
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df)

    # Create the web interface
    create_streamlit_app()
