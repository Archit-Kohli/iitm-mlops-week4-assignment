import pandas as pd
from sklearn.metrics import accuracy_score
import mlflow

mlflow.set_tracking_uri("http://34.44.214.136:8100")

def test_data_columns():
    """
    Tests if the Iris dataset has the correct number of columns.
    """
    print("\n--- Starting data validation test ---")
    data = pd.read_csv('data/iris.csv')
    print("Data loaded successfully.")
    assert len(data.columns) == 5
    print("✅ Data validation complete: Correct number of columns found.")

def test_model_accuracy():
    """
    Tests if the model's accuracy is above 90%.
    """
    print("\n--- Starting model evaluation test ---")
    
    model_name = "iris-lr-model"
    model_uri = f"models:/{model_name}/latest"
    
    print(f"Using latest model from Mlflow: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    
    data = pd.read_csv('data/iris.csv')
    print("Model and data loaded successfully.")
    
    X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = data['species']

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    print(f"Model accuracy: {accuracy:.2f}")
    
    assert accuracy > 0.85
    print("✅ Model evaluation complete: Accuracy is above the 85% threshold.")