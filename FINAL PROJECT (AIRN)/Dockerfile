# Base image
FROM python:3.12.5

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY . .

# Install system libraries required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip (optional)
RUN python -m pip install --upgrade pip

# Debugging: Print the contents of requirements.txt
RUN cat requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
