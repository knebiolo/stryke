# Use Python 3.10 with Debian-based image
FROM python:3.10-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies, including PROJ for pyproj
RUN apt-get update && apt-get install -y \
    python3-distutils python3-setuptools python3-pip \
    build-essential python3-venv \
    proj-bin libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Verify `proj` is installed correctly
RUN proj

# Create and activate virtual environment
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Copy and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY Stryke /app/Stryke
COPY webapp /app/webapp
COPY Data /app/Data
COPY setup.py /app/

# Debugging: Check copied files
RUN ls -l /app && ls -l /app/Stryke && ls -l /app/webapp && ls -l /app/Data

# Ensure Python path is set correctly
ENV PYTHONPATH="/app:/app/Stryke:/app/webapp"

# Expose Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "/app/webapp/app.py"]
 