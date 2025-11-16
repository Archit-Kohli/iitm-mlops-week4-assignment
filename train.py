import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

# --- NEW: Import the poisoning script ---
from data_poisoning import poison_labels

DATA_PATH = 'data/iris.csv'
OUTPUT_DIR = 'outputs'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'model.joblib')
EXPERIMENT_NAME = "iris-lr"

mlflow.set_tracking_uri("http://127.0.0.1:8100")
mlflow.set_experiment(EXPERIMENT_NAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']
# --- IMPORTANT: We split the data ONCE. We only poison y_train. ---
# --- y_test remains clean, acting as our "ground truth" validation set ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- NEW: Define poisoning levels ---
poison_levels = [0.0, 0.05, 0.10, 0.50]
hyperparameters = [0.1, 0.5, 1.0, 1.5]

print("Starting hyperparameter tuning with data poisoning...")

# --- NEW: Outer loop for poisoning levels ---
for poison_level in poison_levels:
    print(f"\n=========================================")
    print(f"  STARTING RUNS FOR POISON LEVEL: {poison_level*100}%")
    print(f"=========================================")
    
    # --- NEW: Poison the training labels for this batch of runs ---
    # We use y_train.copy() to ensure the original y_train is not modified
    y_train_poisoned = poison_labels(y_train.copy(), poison_level)

    for C_value in hyperparameters:
        with mlflow.start_run():
            print(f"--- Training with C={C_value} ---")
            
            # --- NEW: Log the poison_level as a parameter ---
            mlflow.log_param("poison_level", poison_level)
            mlflow.log_param("C", C_value)
            
            model = LogisticRegression(C=C_value, max_iter=200, random_state=42)
            
            # --- MODIFIED: Train on the (potentially) poisoned labels ---
            model.fit(X_train, y_train_poisoned)
            
            # --- Test on the CLEAN validation data ---
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            mlflow.log_metric("accuracy", accuracy)
            print(f"Accuracy: {accuracy:.3f}")
            
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                # We register all models, MLFlow will version them
                registered_model_name="iris-lr-model"
            )
            
            print(f"Model with C={C_value} and PoisonLevel={poison_level} logged.")

print("\nHyperparameter tuning and poisoning script finished successfully.")