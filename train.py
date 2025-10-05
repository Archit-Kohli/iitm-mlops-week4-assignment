import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

DATA_PATH = 'data/iris.csv'
OUTPUT_DIR = 'outputs'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'model.joblib')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training the model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print(f"Saving model to {MODEL_PATH}")
joblib.dump(model, MODEL_PATH)

print("✅ Training script finished successfully.")