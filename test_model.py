import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def test_data_columns():
    """
    Tests if the Iris dataset has the correct number of columns.
    """
    data = pd.read_csv('data/iris.csv')
    assert len(data.columns) == 5

def test_model_accuracy():
    """
    Tests if the model's accuracy is above 90%.
    """
    model = joblib.load('outputs/model.joblib')
    data = pd.read_csv('data/iris.csv')
    X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = data['species']

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    assert accuracy > 0.9