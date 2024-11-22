# Use the official Python image with Debian-based distribution
FROM python:3.11-slim-bullseye

# Set the working directory
WORKDIR /app

# Update and install additional system dependencies, including wget for downloading SQLite
RUN apt-get update && \
    apt-get install -y build-essential portaudio19-dev wget && \
    # Download and compile a newer version of SQLite
    wget https://www.sqlite.org/2023/sqlite-autoconf-3410200.tar.gz && \
    tar xzf sqlite-autoconf-3410200.tar.gz && \
    cd sqlite-autoconf-3410200 && \
    ./configure && make && make install && \
    cd .. && rm -rf sqlite-autoconf-3410200 sqlite-autoconf-3410200.tar.gz && \
    # Update dynamic linker run-time bindings to recognize new SQLite library
    ldconfig && \
    # Clean up apt cache
    apt-get clean && rm -rf /var/lib/apt/lists/*
# Copy the requirements.txt file to the working directory
COPY requirements.txt .

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