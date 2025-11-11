# MLOps Week 6 Assignment: Continuous Deployment with GKE

This project expands on the Week 5 CI pipeline by adding a complete **Continuous Deployment (CD)** workflow. It containerizes the Iris prediction model as a **FastAPI** application and automatically deploys it to the **Google Kubernetes Engine (GKE)**.

## Key Files and Their Roles

### 📄 `train.py`

**Role:** Model Training and Experiment Tracking

This is the core script responsible for running experiments. Its main functions are:
* **Loads** the Iris dataset from `data/iris.csv`.
* **Performs hyperparameter tuning** by training a `LogisticRegression` classifier with multiple `C` values.
* **Integrates with MLFlow** to log each training run as a separate experiment. For each run, it logs:
    * The hyperparameter value (`C`).
    * The resulting evaluation metric (`accuracy`).
    * The trained model artifact itself.
* **Registers** the trained models with the MLFlow Model Registry for versioning.

---

### 🧪 `test_model.py`

**Role:** Automated Testing and Best Model Validation

This file contains automated tests for our pipeline. Instead of testing a static model file, it now dynamically validates the results of our experiments.
* **Data Validation (`test_data_columns`)**: Ensures the `iris.csv` dataset has the correct number of columns.
* **Model Evaluation (`test_model_accuracy`)**: This test is significantly smarter:
    * It **connects to the MLFlow tracking server**
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

This file is a blueprint for building a **Docker** image of our FastAPI application. This makes our application portable and ensures it runs the same way everywhere.

* Starts from a lightweight `python:3.12.3-slim` base image.

* Installs all dependencies from `requirements.txt`.

* Copies the application code (`main.py` and other files) into the image.

* Specifies the command `uvicorn main:app` to run the FastAPI server when the container starts.

---

### 📦 `deployment.yaml`

**Role:** Kubernetes Deployment Blueprint

This Kubernetes manifest tells GKE **what** to run. It acts as the blueprint for our application's pods.

* Defines a `Deployment` object to manage our application's state.

* Instructs Kubernetes to run **`2` replicas (Pods)** for high availability.

* Specifies the Docker **image** to use from **Google Artifact Registry**.

* Tells Kubernetes that our container listens on `containerPort: 80`.

* The Deployment's job is to ensure that 2 Pods are always running. If one crashes, it automatically replaces it.

---

### 🌐 `service.yaml`

**Role:** Kubernetes Service Exposer

This manifest tells GKE **how to expose** our application to the internet.

* Defines a `Service` of `type: LoadBalancer`, which tells Google Cloud to provision an external, public IP address.

* Uses a `selector` (matching `app: iris-api`) to find the Pods created by our `Deployment` and send traffic to them.

* Forwards external traffic from the LoadBalancer's `port: 80` to the container's `targetPort: 80`.

---

### 📦 `hpa.yaml`
**Role:** Autoscaling pods based on average cpu utilisation

This tells GKE when to scale and spin up new pods.

* Uses average CPU utilisation, checks if it goes above 60% then spins up a new pod
* 
* Sets the `minReplicas` and `maxReplicas` to tell GKE the minimum and maximum number of pods
---

### 🚀 `.github/workflows/cd.yml`

**Role:** Continuous Deployment (CD) Workflow

This **GitHub Actions** workflow automates the entire deployment process, creating a true CD pipeline.

* **Triggered** on every push to the `main` branch.

* **Authenticates** with Google Cloud using a Service Account (`GCP_SA_KEY`).

* **Installs** the `gke-gcloud-auth-plugin` required for `kubectl` authentication.

* **Configures Docker** to authenticate with the Google Artifact Registry.

* **Builds** the Docker image using the `Dockerfile`.

* **Pushes** the newly built image to the Google Artifact Registry.

* **Connects to our GKE cluster** (`iris-api-cluster`) and runs `kubectl apply` to update the application with the new image.

---

### ⚙️ `.github/workflows/ci.yml`

**Role:** Continuous Integration (CI) Workflow

This workflow (from Week 5) remains the first line of defense. It ensures code and model quality *before* deployment.

* **Triggered** on every push or pull request.

* **Runs automated tests** (`test_model.py`) to validate the model quality against our 85% accuracy threshold.

* This CI step ensures that only high-quality, validated models are merged to the `main` branch, which in turn triggers the CD workflow.

### Screenshot showing logging using OpenTelemetry
<img width="1919" height="1028" alt="image" src="https://github.com/user-attachments/assets/99dadb8f-9d28-40cb-adb6-a48e5ed102d0" />

