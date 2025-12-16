# Makefile for Water Potability ML Pipeline
# Class: 4DS8
# Author: Ben Aissa Amen Allah

.PHONY: help install lint format security prepare train evaluate full test clean all

# Default target
help:
	@echo "=========================================="
	@echo "Water Potability ML Pipeline"
	@echo "=========================================="
	@echo "Installation & Setup:"
	@echo "  make install           - Install Python dependencies"
	@echo "  make monitoring-up     - Start Elasticsearch + Kibana + MLflow"
	@echo "  make monitoring-down   - Stop monitoring stack"
	@echo "  make monitoring-status - Check if services are running"
	@echo ""
	@echo "ML Pipeline:"
	@echo "  make prepare  - Prepare and preprocess data"
	@echo "  make train    - Train model"
	@echo "  make evaluate - Evaluate model"
	@echo "  make full     - Run complete pipeline (logs to MLflow + Elasticsearch)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint     - Check code with flake8"
	@echo "  make format   - Format code with black"
	@echo "  make security - Security checks with bandit"
	@echo "  make test     - Run tests"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make docker-stop  - Stop Docker container"
	@echo "  make docker-push  - Push to Docker Hub"
	@echo ""
	@echo "Utilities:"
	@echo "  make api       - Start FastAPI server (port 8888)"
	@echo "  make clean     - Clean generated files"
	@echo "  make all       - Run complete CI pipeline"
	@echo ""

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

lint:
	@echo "🔍 Linting code..."
	flake8 model_pipeline.py main.py --max-line-length=100 --ignore=E501,W503

format:
	@echo "✨ Formatting code..."
	black model_pipeline.py main.py

security:
	@echo "🔒 Running security checks..."
	bandit -r . -ll

prepare:
	@echo "📂 Preparing data..."
	python main.py --mode prepare

train:
	@echo "🚀 Training model..."
	python main.py --mode train

evaluate:
	@echo "📊 Evaluating model..."
	python main.py --mode evaluate

full:
	@echo "🔄 Running full pipeline..."
	python main.py --mode full
	@echo "✅ Done! Check results:"
	@echo "  MLflow: http://localhost:5000"
	@echo "  Kibana: http://localhost:5601"

test:
	@if [ -d "tests" ]; then pytest tests/ -v; else echo "No tests found"; fi

api:
	@echo "🌐 Starting API server..."
	uvicorn app:app --reload --host 0.0.0.0 --port 8888

clean:
	@echo "🧹 Cleaning..."
	rm -rf models/*.pkl __pycache__/ .pytest_cache/
	find . -type f -name "*.pyc" -delete

all: install lint format test full
	@echo "🎉 CI pipeline complete!"

# ============================================
# MONITORING STACK
# ============================================

monitoring-up:
	@echo "🚀 Starting monitoring stack..."
	docker compose up -d
	@echo "⏳ Waiting for services..."
	@sleep 30
	@echo "🎯 Starting MLflow..."
	@nohup mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 > mlflow.log 2>&1 &
	@sleep 5
	@echo "✅ Monitoring stack ready!"
	@echo "  MLflow:        http://localhost:5000"
	@echo "  Kibana:        http://localhost:5601"
	@echo "  Elasticsearch: http://localhost:9200"

monitoring-down:
	@echo "🛑 Stopping monitoring stack..."
	@pkill -f "mlflow server" || true
	docker compose down
	@echo "✅ Stopped!"

monitoring-status:
	@echo "🔍 Service status:"
	@curl -s http://localhost:9200 > /dev/null && echo "  ✅ Elasticsearch" || echo "  ❌ Elasticsearch"
	@curl -s http://localhost:5601 > /dev/null && echo "  ✅ Kibana" || echo "  ❌ Kibana"
	@curl -s http://localhost:5000 > /dev/null && echo "  ✅ MLflow" || echo "  ❌ MLflow"

# ============================================
# DOCKER
# ============================================

IMAGE_NAME = amenallah_benaissa_4ds8_mlops
TAG = latest

docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t $(IMAGE_NAME):$(TAG) .

docker-run:
	@echo "🚀 Running container..."
	@docker rm -f ml_project_container 2>/dev/null || true
	docker run -d -p 8000:8000 --name ml_project_container $(IMAGE_NAME):$(TAG)
	@echo "✅ Running at http://localhost:8000"

docker-stop:
	@echo "🛑 Stopping container..."
	docker stop ml_project_container 2>/dev/null || true
	docker rm ml_project_container 2>/dev/null || true

docker-login:
	docker login

docker-push: docker-login
	@read -p "Docker Hub username: " username; \
	docker tag $(IMAGE_NAME):$(TAG) $$username/$(IMAGE_NAME):$(TAG) && \
	docker push $$username/$(IMAGE_NAME):$(TAG)
