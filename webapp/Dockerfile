# Use a lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

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

# docker build -t stryke-webapp -f webapp/Dockerfile .
# docker run -p 5000:5000 stryke-webapp
# expensive5rudabega!@1
