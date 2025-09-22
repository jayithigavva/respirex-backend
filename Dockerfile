FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_render.txt .

# Install Python dependencies with specific versions that work
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements_render.txt

# Copy application code
COPY . .

# Expose port (Render will set this dynamically)
EXPOSE $PORT

# Run the application
CMD uvicorn app:app --host 0.0.0.0 --port $PORT