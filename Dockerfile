# Use the official Python image from the Docker Hub
FROM python:latest

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the HelloWorld.py script to the working directory
COPY HelloWorld.py .

# Command to run your application
CMD ["python", "HelloWorld.py"]