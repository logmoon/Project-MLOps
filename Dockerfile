# Dockerfile for Water Potability ML project
# Class: 4DS8
# Author: Ben Aissa Amen Allah

# Use valid python image
FROM python:3.9-slim

# Set working dir
WORKDIR /app

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install dependencies
# No cache dir to keep image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the API
# Host 0.0.0.0 is required for container access
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
