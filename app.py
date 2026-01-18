import gradio as gr
import pandas as pd
import pickle
import numpy as np

# 1. Load the Model
with open("insurance.pkl", "rb") as f:
    model = pickle.load(f)

def predict_cost(age, sex, bmi, children, smoker, region):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    log_cost =model.predict(input_df)[0]
    return round(np.expm1(log_cost), 2)

app = gr.Interface(
    fn=predict_cost,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(["male", "female"], label="Sex"),
        gr.Number(label="BMI"),
        gr.Number(label="Children"),
        gr.Radio(["yes", "no"], label="Smoker"),
        gr.Radio(["southwest", "southeast", "northwest", "northeast"], label="Region")
    ],
    outputs="number",
    title="Medical Insurance Cost Prediction" 
)

app.launch(share=True)
