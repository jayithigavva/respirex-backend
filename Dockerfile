FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools first
RUN python -m pip install --upgrade pip
RUN pip install --upgrade setuptools wheel

# Copy requirements first for better caching
COPY requirements_docker.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Render will set this dynamically)
EXPOSE $PORT

# Run the application
CMD uvicorn app:app --host 0.0.0.0 --port $PORT