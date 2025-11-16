# MLOps Week 8 Assignment: Data Poisoning and CI/CD

This project demonstrates a full MLOps pipeline, including:
1.  **Experiment Tracking** with MLFlow.
2.  **Automated Testing** (CI) with GitHub Actions.
3.  **Containerization** with Docker and FastAPI.
4.  **Continuous Deployment** (CD) to Google Kubernetes Engine (GKE).
5.  **Data Quality Simulation** (Week 8) by testing the pipeline's resilience against data poisoning attacks.

## Key Files and Their Roles

### 📄 `train.py`

**Role:** Model Training and Experiment Tracking (with Poisoning)

This is the core script responsible for running experiments. For Week 8, it has been updated to test the impact of data poisoning.
* **Loads** the Iris dataset from `data/iris.csv`.
* **Imports** the `poison_labels` function from `data_poisoning.py`.
* **Performs nested-loop experiments**:
    1.  An outer loop iterates through different **data poisoning levels** (e.g., 0%, 5%, 10%, 50%).
    2.  An inner loop performs **hyperparameter tuning** with multiple `C` values.
* **Trains** the model on *clean* features (`X_train`) but *intentionally poisoned* labels (`y_train_poisoned`).
* **Validates** the model against the *original, clean* test set (`X_test`, `y_test`).
* **Integrates with MLFlow** to log each run. For each run, it logs:
    * The `poison_level` parameter.
    * The hyperparameter value (`C`).
    * The resulting evaluation metric (`accuracy`) on the *clean test data*.
    * The trained model artifact itself.
* **Registers** the trained models with the MLFlow Model Registry.

---

### ☣️ `data_poisoning.py`

**Role:** Data Poisoning (Label-Flipping) Simulation

This is a new script for the Week 8 assignment, designed to test the pipeline's resilience against "garbage-in, garbage-out" scenarios.
* **Defines** a function `poison_labels` that takes a set of labels (`y_train`) and a `poison_level` (e.g., 0.05 for 5%).
* **Simulates** a label-flipping attack by randomly selecting a percentage of the training samples.
* **Flips** the labels for these samples to one of the *other* available classes, introducing intentional "noise".
* This module is imported by `train.py` to generate corrupted training data for experiments.

---

### 🧪 `test_model.py`

**Role:** Automated Testing and Best Model Validation

This file contains automated tests for our pipeline.
* **Data Validation (`test_data_columns`)**: Ensures the `iris.csv` dataset has the correct number of columns.
* **Model Evaluation (`test_model_accuracy`)**:
    * It **connects to the MLFlow tracking server**.
    * It **loads the `latest` model** directly from the MLFlow Model Registry.
    * It verifies that this model's accuracy is above our quality threshold (85%).

---

### 📄 `main.py`

**Role:** Model Serving (API)

This script serves the trained Iris model as a web service using **FastAPI**. Its main functions are:

* **Connects to MLFlow** (using the MLflow Tracking URI) to load the `latest` registered version of the `iris-lr-model`.
* **Defines a `/predict` endpoint** that accepts Iris measurements (sepal length, sepal width, etc.) as a JSON input.
* **Returns the model's prediction** as a JSON response.
* Includes a root endpoint `/` to easily check if the API is running.

---

### 🐳 `Dockerfile`

**Role:** Application Containerization

This file is a blueprint for building a **Docker** image of our FastAPI application.
* Starts from a lightweight `python:3.12.3-slim` base image.
* Installs all dependencies from `requirements.txt`.
* Copies the application code (`main.py` and other files) into the image.
* Specifies the command `uvicorn main:app` to run the FastAPI server when the container starts.

---

### 📦 `deployment.yaml`

**Role:** Kubernetes Deployment Blueprint

This Kubernetes manifest tells GKE **what** to run.
* Defines a `Deployment` object to manage our application's state.
* Instructs Kubernetes to run replicas (Pods) for high availability. (The file in the repo shows 1 replica, while the README says 2).
* Specifies the Docker **image** to use from **Google Artifact Registry**.
* Tells Kubernetes that our container listens on `containerPort` (e.g., 8200 in the file, 80 in the README).
* The Deployment's job is to ensure the desired number of Pods are always running.

---

### 🌐 `service.yaml`

**Role:** Kubernetes Service Exposer

This manifest tells GKE **how to expose** our application to the internet.
* Defines a `Service` of `type: LoadBalancer`, which tells Google Cloud to provision an external, public IP address.
* Uses a `selector` (matching `app: iris-api`) to find the Pods created by our `Deployment` and send traffic to them.
* Forwards external traffic from the LoadBalancer's `port: 80` to the container's `targetPort`.

---

### 📦 `hpa.yaml`

**Role:** Autoscaling pods based on average cpu utilisation

This tells GKE when to scale and spin up new pods.
* Uses average CPU utilisation, checks if it goes above 60% then spins up a new pod.
* Sets the `minReplicas` and `maxReplicas` to tell GKE the minimum and maximum number of pods.

---

### 🚀 `.github/workflows/cd.yml`

**Role:** Continuous Deployment (CD) Workflow

This **GitHub Actions** workflow automates the entire deployment process.
* **Triggered** on every push to the `main` branch.
* **Authenticates** with Google Cloud using a Service Account (`GCP_SA_KEY`).
* **Installs** the `gke-gcloud-auth-plugin` required for `kubectl` authentication.
* **Configures Docker** to authenticate with the Google Artifact Registry.
* **Builds** the Docker image using the `Dockerfile`.
* **Pushes** the newly built image to the Google Artifact Registry.
* **Connects to our GKE cluster** (`iris-api-cluster-v2`) and runs `kubectl apply` to update the application with the new image.

---

### ⚙️ `.github/workflows/ci.yml`

**Role:** Continuous Integration (CI) Workflow

This workflow ensures code and model quality *before* deployment.
* **Triggered** on every push or pull request.
* **Runs automated tests** (`test_model.py`) to validate the model quality against our 85% accuracy threshold.
* This CI step ensures that only high-quality, validated models are merged to the `main` branch, which in turn triggers the CD workflow.

### Screenshot showing logging using OpenTelemetry
<img width="1919" height="1028" alt="image" src="https://github.com/user-attachments/assets/99dadb8f-9d28-40cb-adb6-a48e5ed102d0" />

### Screenshot of MLFlow Graph with different poisoning levels
<img width="954" height="450" alt="newplot" src="https://github.com/user-attachments/assets/35f83f2c-5026-48fa-9690-dab6d95a1814" />
