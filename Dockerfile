# Use the official Python Alpine/Slim image for a smaller footprint
FROM python:3.10-slim

# Expose FastAPI default port
EXPOSE 8000

# Set the working directory
WORKDIR /app

# Install system dependencies required for data science libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire dashboard application
COPY . .

# Set environment variables for FastAPI
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PORT=8000

# Healthcheck to verify the API is running
HEALTHCHECK CMD curl --fail http://localhost:8000/api/health || exit 1

# Run the FastAPI application
ENTRYPOINT ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
