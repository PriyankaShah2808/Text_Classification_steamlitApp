import streamlit as st
from transformers import pipeline
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
model_path = os.getenv("MODEL_PATH") 

classifier = pipeline('text-classification', model=model_path, framework="pt")

label_mapping = {
    'LABEL_0': 'Anxiety',
    'LABEL_1': 'Bipolar',
    'LABEL_2': 'Depression',
    'LABEL_3': 'Normal',
    'LABEL_4': 'Personality Disorder',
    'LABEL_5': 'Stress',
    'LABEL_6': 'Suicidal'
}

st.title("Text Classification")

text_input = st.text_area("Enter statement for classification:", "")

# Classify button
if st.button("Classify"):
    if text_input:
        predictions = classifier(text_input)
        st.write("Predictions:")
        for pred in predictions:
            mapped_label = label_mapping.get(pred['label'], pred['label'])
            st.write(f"Label: {mapped_label}, Score: {pred['score']:.4f}")
    else:
        st.error("Please enter some text for classification.")
