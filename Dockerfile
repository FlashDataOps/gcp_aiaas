# Use the official Python image from the Docker Hub
FROM python:latest

# Set the working directory
WORKDIR /app

# Copy the HelloWorld.py script to the working directory
COPY HelloWorld.py .

EXPOSE 8080
# Command to run your application
CMD ["python", "HelloWorld.py"]