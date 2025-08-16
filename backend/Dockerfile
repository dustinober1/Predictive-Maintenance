FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY models/ ./models/

# Copy data directory
COPY ../data/ ./data/

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "app/simple_server.py"]