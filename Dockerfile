# =============================================================================
# Infrastructure Cost Predictor - Docker Image
# =============================================================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY examples/ ./examples/
COPY docs/ ./docs/

# Create models directory
RUN mkdir -p models

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "src/main.py"]

# Labels for metadata
LABEL maintainer="Philip Kumah Junior <philip@vertaforge.com>"
LABEL description="Infrastructure Cost Predictor - AI-powered cloud cost prediction and optimization"
LABEL version="1.0.0"
LABEL repository="https://github.com/jkudjo/infrastructure-cost-predictor"
