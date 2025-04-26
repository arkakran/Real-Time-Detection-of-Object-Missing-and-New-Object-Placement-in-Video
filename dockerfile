# Use an official Python image
FROM python:3.10-slim

# Install necessary system packages (for OpenCV and display)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the code and requirements
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Allow command-line override (use CMD ["python", "main.py"] as default)
ENTRYPOINT ["python", "main.py"]
