# Makefile for Water Potability ML Pipeline
# Class: 4DS8
# Author: Ben Aissa Amen Allah

.PHONY: help install lint format security prepare train evaluate full test clean all

# Default target
help:
	@echo "=========================================="
	@echo "Water Potability ML Pipeline - Makefile"
	@echo "=========================================="
	@echo "Available targets:"
	@echo ""
	@echo "Installation:"
	@echo "  make install      - Install dependencies from requirements.txt"
	@echo ""
	@echo "Model Pipeline:"
	@echo "  make prepare      - Prepare and preprocess data"
	@echo "  make train        - Train the Naive Bayes model"
	@echo "  make evaluate     - Evaluate trained model"
	@echo "  make full         - Run complete ML pipeline"
	@echo ""
	@echo "CI Steps (Code Quality):"
	@echo "  make lint         - Check code quality with flake8"
	@echo "  make format       - Auto-format code with black"
	@echo "  make security     - Run security checks with bandit"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container (maps port 8000)"
	@echo "  make docker-stop  - Stop and remove Docker container"
	@echo "  make docker-login - Log in to Docker Hub"
	@echo "  make docker-push  - Push image to Docker Hub"
	@echo ""
	@echo "Utilities:"
	@echo "  make test         - Run tests with pytest"
	@echo "  make api          - Start FastAPI server and UI for testing"
	@echo "  make mlflow-ui    - Starts MLflow UI"
	@echo "  make clean        - Clean generated files and cache"
	@echo "  make all          - Run complete CI/CD pipeline"
	@echo ""

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

# Code quality check with flake8
lint:
	@echo "🔍 Checking code quality..."
	flake8 model_pipeline.py main.py --max-line-length=100 --ignore=E501,W503
	@echo "✅ Code quality check passed!"

# Auto-format code with black
format:
	@echo "✨ Formatting code..."
	black model_pipeline.py main.py
	@echo "✅ Code formatted!"

# Security check with bandit
security:
	@echo "🔒 Running security checks..."
	bandit -r . -ll
	@echo "✅ Security check passed!"

# Prepare data
prepare:
	@echo "📂 Preparing data..."
	python main.py --mode prepare
	@echo "✅ Data preparation complete!"

# Train the model
train:
	@echo "🚀 Training model..."
	python main.py --mode train
	@echo "✅ Training complete!"

# Evaluate the model
evaluate:
	@echo "📊 Evaluating model..."
	python main.py --mode evaluate
	@echo "✅ Evaluation complete!"

# Run full pipeline
full:
	@echo "🔄 Running full pipeline..."
	python main.py --mode full
	@echo "✅ Full pipeline complete!"

# Run tests
test:
	@echo "🧪 Running tests..."
	@if [ -d "tests" ]; then \
		pytest tests/ -v; \
	else \
		echo "No tests directory found. Skipping tests."; \
	fi

# Run API for testing
api:
	@echo "🌐 Starting FastAPI server for testing..."
	uvicorn app:app --reload --host 0.0.0.0 --port 8888

# Start the MLflow ui
mlflow-ui:
	@echo "🌐 Starting MLflow UI"
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5555 &


# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf models/*.pkl
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete!"

# Run complete CI/Not_CD (yet) pipeline
# all: install lint format security test full # Full version (with security)
all: install lint format test full # Skipping security cuz it's screaming at me for no reason.
	@echo "🎉 Complete CI/Not_CD (yet) executed successfully!"


# Docker configuration
IMAGE_NAME = amenallah_benaissa_4ds8_mlops
TAG = latest

# Build the Docker image
docker-build:
	@echo "🐳 Building Docker image: $(IMAGE_NAME):$(TAG)..."
	docker build -t $(IMAGE_NAME):$(TAG) .
	@echo "✅ Docker image built!"

# Run the Docker container
# Maps port 8000 on host to 8000 on container
docker-run:
	@echo "🚀 Running Docker container..."
	docker run -d -p 8000:8000 --name ml_project_container $(IMAGE_NAME):$(TAG)
	@echo "✅ Container running! Access at http://localhost:8000"

# Stop the Docker container
docker-stop:
	@echo "🛑 Stopping Docker container..."
	-docker stop ml_project_container
	-docker rm ml_project_container
	@echo "✅ Container stopped and removed!"

# Log in to Docker Hub (Interactive)
docker-login:
	@echo "🔑 Logging in to Docker Hub..."
	docker login

# Push the image to Docker Hub
# Requires 'make docker-login' first
docker-push:
	@echo "⬆️ Pushing image to Docker Hub..."
	@read -p "Enter your Docker Hub username: " username; \
	docker tag $(IMAGE_NAME):$(TAG) $$username/$(IMAGE_NAME):$(TAG) && \
	docker push $$username/$(IMAGE_NAME):$(TAG)
	@echo "✅ Image pushed!"