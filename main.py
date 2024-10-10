import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load and prepare the data
@st.cache_data
def load_data():
    df = pd.read_csv("Balanced_Training_Essay_Data.csv")
    return df

# Load data and prepare the model
df = load_data()
cv = CountVectorizer()
x = cv.fit_transform(df["text"])
y = df["generated"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)
model = MultinomialNB()
model.fit(x_train, y_train)

# Streamlit application
st.title("AI TEXT CLASSIFICATION")

# Custom CSS for background color and other styling
st.markdown(
    """<style>
    body {
        background-color: #f0f8ff;  /* Light blue background */
        color: #333;  /* Dark text color for contrast */
    }
    .text-area {
        background-color: #ffffff;  /* White background for text area */
        border: 2px solid #008b8b;
        border-radius: 5px;
    }
    .button {
        background-color: #008b8b;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    </style>""",
    unsafe_allow_html=True
)

# Sidebar for additional options
st.sidebar.header("Options")
st.sidebar.write("This application classifies text as either AI generated or human generated.")
st.sidebar.write("Enter your text in the main area to see the prediction.")

# Input form for new messages with custom styling
new_message = st.text_area("Enter the TEXT you want to classify:", height=300, key="input", help="Type your text here...", placeholder="Enter text...")

if st.button("Predict", key="predict_button"):
    if new_message:
        # Transform the input message using the same vectorizer
        x_new = cv.transform([new_message])
        # Predict the category of the new message
        prediction = model.predict(x_new)
        prediction_prob = model.predict_proba(x_new)

        # Display the result with color
        if prediction[0] == 1:
            result = "AI GENERATED"
            color = "red"
            confidence = prediction_prob[0][1] * 100
        else:
            result = "HUMAN GENERATED"
            color = "green"
            confidence = prediction_prob[0][0] * 100
        
        # Display the result
        st.markdown(f'<p style="color:{color}; font-size: 24px;"><strong>The TEXT is classified as: {result}</strong></p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:{color};">Confidence: {confidence:.2f}%</p>', unsafe_allow_html=True)
    else:
        st.write("Please enter a TEXT to classify.")