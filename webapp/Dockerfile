# Use a prebuilt Python image with essential scientific libraries
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PROJ_DIR="/usr"
ENV PROJ_LIB="/usr/share/proj"

# Set working directory
WORKDIR /app

# Install system dependencies, including PROJ for pyproj
RUN apt-get update && apt-get install -y \
    python3-distutils python3-setuptools python3-pip \
    build-essential python3-venv \
    proj-bin libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Verify PROJ is installed
RUN proj

# Use virtual environment for isolation
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools wheel

# Copy only the requirements file to leverage Docker cache
COPY requirements.txt /app/

# Install core dependencies first
RUN pip install --no-cache-dir numpy pandas scipy

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY Stryke /app/Stryke
COPY webapp /app/webapp
COPY Data /app/Data
COPY setup.py /app/

# Debugging: Check copied files
RUN ls -l /app && ls -l /app/Stryke && ls -l /app/webapp && ls -l /app/Data

# Set Python path correctly
ENV PYTHONPATH="/app:/app/Stryke:/app/webapp"

# Expose Flask port
EXPOSE 5000

# Run the Flask app
# CMD ["python", "/app/webapp/app.py"]
CMD gunicorn --bind 0.0.0.0:$PORT --workers=2 --threads=2 app.webapp:app