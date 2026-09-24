import gradio as gr
import pandas as pd
import joblib


model = joblib.load("model/titanic_model.pkl")

PAGE_CSS = """
.gradio-container {
  max-width: 900px !important;
  margin-inline: auto !important;
}
"""

def predict_survival(
    pclass,
    sex,
    age,
    siblings_spouses,
    parents_children,
    fare,
    embarked
):
  passenger = pd.DataFrame([{
    "Pclass": int(pclass),
    "Sex": sex,
    "Age": age,
    "SibSp": siblings_spouses,
    "Parch": parents_children,
    "Fare": fare,
    "Embarked": embarked
  }])

  prediction = model.predict(passenger)[0]

  probabilities = model.predict_proba(passenger)[0]

  survival_probability = probabilities[1]

  if prediction == 1:
    result = "Likely survived"
  else:
    result = "Likely did not survived"

  return (
    result,
    f"{survival_probability:.1%}"
  )


demo = gr.Interface(
  fn=predict_survival,
  inputs=[
    gr.Dropdown(
      choices=[1, 2, 3],
      label="Passenger Class",
      value=3
    ),
    gr.Radio(
      choices=["male", "female"],
      label="Sex",
      value="male"
    ),
    gr.Number(
      label="Age",
      value=30
    ),
    gr.Number(
      label="Siblings / Spouses",
      value=0
    ),
    gr.Number(
      label="Parents / Children",
      value=0
    ),
    gr.Number(
      label="Fare",
      value=10
    ),
    gr.Dropdown(
      choices=["S", "C", "Q"],
      label="Embarked",
      value="S"
    )
  ],
  outputs=[
    gr.Textbox(label="Prediction"),
    gr.Textbox(label="Survival probability")
  ],
  title="Titanic Survival Predictor",
  description="""
  **This is an educational project. Its output describes patterns in a historical
  dataset and should not be treated as a factual conclusion about an individual
  passenger or as a model suitable for real-world decisions.**

  Enter passenger information and
  the machine learning model will estimate whether the passenger survived.

  ⭐ **Enjoying the project? [Star it on GitHub](https://github.com/phhien203/titanic-ml-app-v1)!**
  """,
  flagging_mode="never"
)

demo.launch(
  server_name="0.0.0.0",
  server_port=8760,
  css=PAGE_CSS
)
