
import streamlit as st
import pandas as pd
import joblib

# Load the saved preprocessing + Logistic Regression pipeline
pipeline = joblib.load("titanic_pipeline.pkl")

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Survival Predictor")
st.write(
    "Enter passenger details to predict the probability "
    "of survival using Logistic Regression."
)

# Passenger details
pclass = st.selectbox(
    "Passenger Class",
    options=[1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    options=["female", "male"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=25.0,
    step=1.0
)

sibsp = st.number_input(
    "Number of Siblings/Spouses Aboard",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Number of Parents/Children Aboard",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Ticket Fare",
    min_value=0.0,
    value=30.0,
    step=1.0
)

embarked = st.selectbox(
    "Port of Embarkation",
    options=["S", "C", "Q"],
    format_func=lambda x: {
        "S": "Southampton",
        "C": "Cherbourg",
        "Q": "Queenstown"
    }[x]
)

if st.button("Predict Survival", type="primary"):

    # Use the same original columns used during training
    passenger = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }])

    # Pipeline performs preprocessing automatically
    prediction = pipeline.predict(passenger)[0]
    probability = pipeline.predict_proba(passenger)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Estimated Survival Probability",
        f"{probability:.1%}"
    )

    if prediction == 1:
        st.success("Prediction: Survived")
    else:
        st.error("Prediction: Did not survive")

    st.caption(
        "This is a machine-learning prediction, "
        "not a guarantee of an actual outcome."
    )
