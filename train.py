import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("data/titanic.csv")

features = [
  "Pclass",
  "Sex",
  "Age",
  "SibSp",
  "Parch",
  "Fare",
  "Embarked"
]

X = df[features]
y = df["Survived"]

numeric_features = [
  "Age",
  "SibSp",
  "Parch",
  "Fare"
]

categorical_features = [
  "Pclass",
  "Sex",
  "Embarked"
]

numeric_pipeline = Pipeline([
  ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
  ("imputer", SimpleImputer(strategy="most_frequent")),
  ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
  ("num", numeric_pipeline, numeric_features),
  ("cat", categorical_pipeline, categorical_features)
])

model = Pipeline([
  ("preprocessor", preprocessor),
  ("classifier", RandomForestClassifier(
    n_estimators=100,
    random_state=42
  ))
])

X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size=0.2,
  random_state=42,
  stratify=y
)

model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

joblib.dump(model, "model/titanic_model.pkl")

print("Model saved.")
