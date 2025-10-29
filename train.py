import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

DATA_PATH = 'data/iris.csv'
OUTPUT_DIR = 'outputs'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'model.joblib')
EXPERIMENT_NAME = "iris-logistic-regression"

mlflow.set_tracking_uri("http://127.0.0.1:8100")
mlflow.set_experiment(EXPERIMENT_NAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

hyperparameters = [0.1, 0.5, 1.0, 1.5]

print("Starting hyperparameter tuning...")

for C_value in hyperparameters:
    with mlflow.start_run():
        print(f"--- Training with C={C_value} ---")
        mlflow.log_param("C", C_value)
        
        model = LogisticRegression(C=C_value, max_iter=200, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        mlflow.log_metric("accuracy", accuracy)
        print(f"Accuracy: {accuracy:.3f}")
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="iris-logistic-regression-model"
        )
        
        print(f"Model with C={C_value} logged and registered.")

print("Hyperparameter tuning and training script finished successfully.")