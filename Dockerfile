# Use specific Python 3.11 image with explicit version
FROM python:3.11.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Force Python version check
RUN python --version

# Install build tools first with explicit versions
RUN pip install --upgrade pip==23.3.1
RUN pip install setuptools==68.2.2 wheel==0.41.2

# Copy requirements
COPY requirements_docker.txt requirements.txt

# Install dependencies with no build isolation
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE $PORT

# Run the application
CMD uvicorn app:app --host 0.0.0.0 --port $PORT