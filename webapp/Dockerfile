# Use an official Python runtime as a parent image.
FROM python:3.8-slim

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file into the container.
COPY ../requirements.txt .


# Install any needed packages specified in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container.
COPY . .

# Expose port 5000 for the Flask app.
EXPOSE 5000

# Set environment variable (optional)
ENV FLASK_ENV=production

# Run the Flask app.
CMD ["python", "app.py"]
