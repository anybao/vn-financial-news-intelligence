#!/bin/bash

# Ensure PYTHONPATH is set so that imports from 'src' work correctly
export PYTHONPATH=$(pwd)

echo "====================================================="
echo "  Starting Financial News Intelligence System        "
echo "====================================================="

# 1. Check if Docker is running (needed for MLOps stack)
if ! docker info > /dev/null 2>&1; then
  echo "Warning: Docker does not seem to be running or is not accessible."
  echo "Please start Docker if you want to run MLflow, Prometheus, and Grafana."
  echo "Continuing without Docker..."
else
  echo "[1/3] Starting MLOps infrastructure (Docker Compose)..."
  docker-compose up -d
fi

echo ""
echo "Setting up Python virtual environment with Python 3.13..."
if [ ! -d "venv" ]; then
    if command -v python3.13 >/dev/null 2>&1; then
        python3.13 -m venv venv
    elif command -v python3.12 >/dev/null 2>&1; then
        python3.12 -m venv venv
    else
        python3 -m venv venv
    fi
fi
source venv/bin/activate
pip install -r requirements.txt

echo ""
echo "[2/3] Starting Data Ingestion Scheduler in the background..."
# Run the scheduler in the background
python src/ingestion/scheduler.py > scheduler.log 2>&1 &
SCHEDULER_PID=$!
echo "Scheduler running with PID: $SCHEDULER_PID (Logs in scheduler.log)"

# Function to handle cleanup when the user stops the script (e.g., via Ctrl+C)
cleanup() {
    echo ""
    echo "====================================================="
    echo "  Shutting down systems...                           "
    echo "====================================================="
    echo "Stopping Scheduler (PID: $SCHEDULER_PID)..."
    kill $SCHEDULER_PID
    
    if docker info > /dev/null 2>&1; then
        echo "Stopping Docker containers..."
        docker-compose down
    fi
    
    echo "Shutdown complete."
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM to trigger the cleanup function
trap cleanup SIGINT SIGTERM

echo ""
echo "[3/3] Starting FastAPI Serving Application..."
echo "API will be available at http://localhost:8000"
echo "API metrics available at http://localhost:8000/metrics"
echo "Press Ctrl+C to stop all systems."
echo ""

# Run FastAPI with uvicorn in the foreground
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
