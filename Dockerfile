# 1. Use official Python base image
FROM python:3.12.3-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the app code
COPY . .

# 5. Expose port (must match deployment.yaml containerPort)
EXPOSE 8200

# 6. Command to run the server
#    We use the port 8200 to match the Week 7 k8s files
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8200"]