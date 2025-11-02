import logging
import pandas as pd
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MLFLOW_TRACKING_URI = "http://34.44.214.136:8100"
logging.info(f"Setting MLflow tracking URI to: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI()

model_name = "iris-lr-model"
model_uri = f"models:/{model_name}/latest"
model = None

try:
    logging.info(f"Attempting to load model '{model_name}' from URI: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    logging.info("Successfully loaded MLflow model.")
except Exception as e:
    logging.error(f"Failed to load model. Error: {e}")
    exit()


@app.post("/predict")
def predict(data: IrisInput):
    """
    This endpoint takes Iris measurements as input and returns the predicted species.
    """
    input_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    input_df = pd.DataFrame(input_data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

    prediction = model.predict(input_df)

    return {"prediction": prediction[0]}

@app.get("/")
def read_root():
    """
    A simple endpoint to test if the API is running.
    """
    return {"message": "Welcome to the Iris Prediction API"}
