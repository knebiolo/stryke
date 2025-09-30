# Use an official lightweight Python image.
# FROM python:3.9-slim
FROM public.ecr.aws/docker/library/python:3.9-slim


# Set the working directory to /app
WORKDIR /app

# Copy the main requirements file and install dependencies.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire repository into the container.
COPY . .

# Ensure Python can find modules in the repository root.
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Expose the port (Railway sets the PORT environment variable)
EXPOSE 5000

# Command to run the application using Gunicorn.
# Note that app.py is located in the webapp folder.
# CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --workers=2 --threads=2 webapp.app:app"]

# Use PORT if the platform sets it; default 5000 for local
CMD sh -c 'gunicorn --workers 1 --threads 4 --timeout 0 --bind 0.0.0.0:${PORT:-5000} webapp.app:app'

