# Titanic Survival Predictor

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-Data%20Processing-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Web%20Interface-FF7C00?logo=gradio&logoColor=white)](https://www.gradio.app/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

A small end-to-end machine learning application that estimates whether a Titanic passenger would have survived based on details such as age, sex, ticket class, fare, and family members aboard.

The project is designed to make a typical ML workflow easy to explore: a model is trained from tabular data, saved to disk, loaded by a web application, and used to return an interactive prediction.

> This is an educational project. Its output describes patterns in a historical dataset and should not be treated as a factual conclusion about an individual passenger or as a model suitable for real-world decisions.

## What the app does

The browser interface asks for seven passenger attributes and returns:

- a prediction: `Likely survived` or `Likely did not survive`;
- the model's estimated probability of survival.

For example, changing the passenger class or fare may change the result because the trained model found relationships between those values and survival in the source data. That is correlation learned from historical examples—not proof that any one attribute caused the outcome.

## Why this project is useful

This repository demonstrates how the pieces of a small ML product fit together:

- loading and selecting data with pandas;
- filling in missing values;
- converting categories such as `male` and `female` into values a model can process;
- training and evaluating a random forest classifier;
- packaging preprocessing and prediction into one reusable pipeline;
- saving and loading a trained model with joblib;
- exposing the result through a Gradio web interface;
- containerizing the application with Docker.

The project also shows an end-to-end implementation rather than a standalone notebook: data preparation, reproducible training, model persistence, inference UI, and container setup are separated into clear parts.

## How it works

```text
data/titanic.csv
       |
       v
train.py: clean data -> encode categories -> train random forest
       |
       v
model/titanic_model.pkl
       |
       v
app.py: Gradio form -> model prediction -> result in browser
```

The preprocessing and classifier are stored together in a scikit-learn `Pipeline`. This matters because the application applies the same data preparation rules during prediction that were used during training.

### Dataset source and citation

The training data in [`data/titanic.csv`](data/titanic.csv) comes from the `train.csv` file published for Kaggle's [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data) competition. It contains 891 passenger records, including the survival outcome used as the model's target.

Please cite the dataset as:

> Cukierski, Will. *Titanic: Machine Learning from Disaster*. Kaggle, 2012. https://www.kaggle.com/competitions/titanic

### Inputs

| Input | Meaning |
| --- | --- |
| Passenger class (`Pclass`) | Ticket class: 1st, 2nd, or 3rd |
| Sex | Sex recorded in the dataset |
| Age | Passenger age in years |
| Siblings / spouses (`SibSp`) | Number of siblings or spouses aboard |
| Parents / children (`Parch`) | Number of parents or children aboard |
| Fare | Ticket fare paid |
| Embarked | Port of embarkation: Southampton (`S`), Cherbourg (`C`), or Queenstown (`Q`) |

### Model in plain language

The model is a **random forest classifier**. A decision tree learns a series of questions about the input data, such as whether a value is above or below a threshold. A random forest combines many such trees—in this project, 100—and lets them vote on the final classification. Combining trees usually produces a more stable result than relying on a single tree.

Before training or prediction:

- missing numeric values are replaced with the median value from the training data;
- missing categories are replaced with the most common value;
- categorical fields are one-hot encoded into numeric columns.

The included model achieved approximately **80.4% accuracy** on a held-out test set of 179 records. Accuracy means that roughly 80% of those test records were classified correctly. It does not mean every displayed probability is 80% reliable, and a fuller production evaluation would include additional metrics, error analysis, and bias analysis.

## Run locally

### Prerequisites

- Python (Python 3.12 is used by the Docker image)
- `pip`

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Start the application:

```bash
python app.py
```

Then open [http://localhost:8760](http://localhost:8760) in a browser.

The repository already includes a trained model, so retraining is not required before starting the app.

## Retrain the model

Run the training script when you want to rebuild the model from `data/titanic.csv`:

```bash
python train.py
```

The script uses a stratified 80/20 train-test split with a fixed random seed, prints test accuracy, and replaces `model/titanic_model.pkl` with the newly trained pipeline.

## Run with Docker

Build the image:

```bash
docker build -t titanic-survival-predictor .
```

Run the container and expose the Gradio server on port 8760:

```bash
docker run --rm -p 8760:8760 titanic-survival-predictor
```

Then open [http://localhost:8760](http://localhost:8760).

## Project structure

```text
.
├── app.py                    # Gradio user interface and prediction logic
├── train.py                  # Data preprocessing, training, and evaluation
├── data/
│   └── titanic.csv           # Training dataset (891 passenger records)
├── model/
│   └── titanic_model.pkl     # Serialized trained pipeline
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container definition
├── LICENSE                   # MIT license
└── README.md
```

## Technology choices

- **pandas** for loading and selecting tabular data
- **scikit-learn** for preprocessing, training, evaluation, and the ML pipeline
- **joblib** for model serialization
- **Gradio** for the interactive browser interface
- **Docker** for a portable runtime

## Limitations and possible improvements

- The dataset is small and reflects a specific historical event.
- The app uses a single train-test split rather than cross-validation.
- Accuracy alone does not show which types of errors the model makes.
- The model may reproduce social and historical biases present in the data.
- Input validation is minimal, so unrealistic values can be submitted.
- Dependencies are not pinned to exact versions, which can affect reproducibility over time.

Useful next steps would be to add automated tests, validation ranges, a confusion matrix and precision/recall metrics, cross-validation, model explainability, pinned dependencies, and continuous integration.

## License

This project is available under the [MIT License](LICENSE).
