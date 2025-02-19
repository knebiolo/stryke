# Use Python 3.10 base image with Debian-based OS
FROM python:3.10-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for building some Python packages
RUN apt-get update && apt-get install -y \
    python3-distutils \
    python3-setuptools \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# copy requirements first 
COPY requirements.txt /app/

# then Install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary files and directories
COPY Stryke /app/Stryke
COPY webapp /app/webapp
COPY Data /app/Data
COPY requirements.txt /app/
COPY setup.py /app/

# Debugging: Check if files are correctly copied
RUN ls -l /app && ls -l /app/Stryke && ls -l /app/webapp && ls -l /app/Data

# Ensure Stryke is found in the Python path
ENV PYTHONPATH="/app:/app/Stryke:/app/webapp"

# Expose port 5000 for Flask
EXPOSE 5000

# Run the Flask app
CMD ["python", "/app/webapp/app.py"]

