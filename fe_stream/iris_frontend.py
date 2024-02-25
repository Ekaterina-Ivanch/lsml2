import os
import random

import requests
import streamlit as st
from PIL import Image

st.title("Iris Flower Prediction")
st.write("Welcome to the Iris Flower Classifier!")
st.write("""
The Iris dataset holds a significant place in the history of statistics, pattern recognition, and machine learning. The dataset was introduced by the British biologist and statistician Sir Ronald A. Fisher in 1936. Fisher used the Iris dataset in his paper "The use of multiple measurements in taxonomic problems" to illustrate the application of discriminant analysis.
The dataset consists of measurements from 150 iris flowers, representing three different species: setosa, versicolor, and virginica. Four features were measured for each sample: sepal length, sepal width, petal length, and petal width. Fisher's intention was to showcase how these four features could be used to distinguish between the three iris species based on statistical methods.
The Iris dataset gained prominence in the field of machine learning as a benchmark dataset for testing various algorithms, particularly for classification tasks. Its simplicity, yet effectiveness in demonstrating classification concepts, made it a popular choice for researchers and practitioners.
In the later years, the Iris dataset became a standard example used in various statistics and machine learning courses. Its widespread adoption contributed to the development and evaluation of many classification algorithms, including the famous k-nearest neighbors and decision tree classifiers.
Today, the Iris dataset remains a classic and fundamental resource for teaching and learning in the fields of data science, machine learning, and statistics. It symbolizes the importance of carefully curated datasets in the development and validation of algorithms, providing a historical foundation for the exploration and understanding of various statistical and machine learning concepts.
""")

# Features
sepal_length = st.number_input("Enter Sepal Length:")
sepal_width = st.number_input("Enter Sepal Width:")
petal_length = st.number_input("Enter Petal Length:")
petal_width = st.number_input("Enter Petal Width:")


# Trigger prediction
if st.button("Predict"):
    # Make a POST request to the Flask API
    response = requests.post(
        "http://be_app:5000/predict",
        json={"features": [sepal_length, sepal_width, petal_length, petal_width]},
    )
    # Prediction result
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.success(f"Predicted Iris Species: {prediction}")

        # Display image
        image_files = [
            file
            for file in os.listdir("./flowers_examples")
            if prediction in file and file.endswith((".jpg", ".jpeg", ".png"))
        ]
        random_image = random.choice(image_files)
        image = Image.open(f".//flowers_examples/{random_image}")
        st.image(image, caption=prediction, width=400)
    else:
        st.error("Prediction failed. Please try again.")

# Local terminal: streamlit run iris_frontend.py
