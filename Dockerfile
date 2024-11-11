# Use the official Python image from the Docker Hub
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

RUN apt-get install portaudio19-dev python3-all-dev
# Install the dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the HelloWorld.py script to the working directory
# COPY HelloWorld.py .

# Copy the rest of the app to the working directory
COPY . .

# Expose port 8080 for Streamlit
EXPOSE 8080

# Command to run your application
# CMD ["python", "HelloWorld.py"]
# Command to run your application with Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]