import joblib
import streamlit as st
import numpy as np

#load model
model = joblib.load('model.pkl')

races = {
    "American Indian": 0,
    "Multi-Race": 1,
    "Asian": 2,
    "Pacific Islander": 3,
    "Total": 4,
    "African-American": 5,
    "Unknown": 6,
    "Hispanic": 7,
    "White": 8,
    "White/Other": 9
}

@st.cache()

def predict(data: np.array) -> str:
    return "Preterm" if model.predict(data) == 0 else "Very Preterm"


def main():

    st.title("preterm birth")

    text = """
    <div style="background-color:green;">
    <h1 style="font-style:italic;text-align:center;">Welcome to the Preterm birth prediction app</h1>
    """

    st.markdown(text, unsafe_allow_html=True)

    year = st.number_input("Year")
    birth = st.number_input("Total birth")
    events = st.number_input("Events")
    percent = st.number_input("Percent")
    upper = st.number_input("Upper 95% CI")
    lower = st.number_input("Lower 95% CI")
    race = st.text_input("Race")

    data = np.array([year, races.get(race), birth, events, percent, upper, lower]).reshape(1,-1)

    if st.button("Predict"):
        message = predict(data)
        st.success(f"This patient will have a {message} birth when the time comes")
        print(message)


if __name__ == "__main__":
    main()