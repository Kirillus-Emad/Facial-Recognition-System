FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Force install numpy==1.23.5 ignoring dependency conflicts
RUN pip install --no-cache-dir --force-reinstall --no-deps numpy==1.23.5

# Copy the full application
COPY . .

# Create necessary directories (they may already exist; this ensures no errors)
RUN mkdir -p /tmp/uploads /tmp/processed_videos /tmp/reports static templates models

# Env settings
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Expose port
EXPOSE 7860

# Start API server
CMD ["uvicorn", "main_optimized:app", "--host", "0.0.0.0", "--port", "7860"]
