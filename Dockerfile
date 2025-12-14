FROM python:3.10-slim

# Install system dependencies for OpenCV and camera access
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libgl1 \
    libglib2.0-0 \
    libgthread-2.0-0 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY gradio_app.py .
COPY model_utils.py .
COPY config.yaml .

# Copy model file
COPY last.pt /app/model.pt

# Expose ports
# 8501 for Streamlit, 7860 for Gradio (internal)
EXPOSE 8501 7860

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_ADDRESS=0.0.0.0

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

