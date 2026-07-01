# Use official lightweight Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies (needed for compiling any dynamic dependencies if necessary)
# slim image is minimal, but we need minimal overhead
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files, including pre-generated assets
COPY app.py .
COPY clean_catalog.json .
COPY index.faiss .

# Expose port
EXPOSE 10000

# Start server using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
