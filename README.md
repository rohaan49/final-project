# YOLOv8 Real-Time Detection with Streamlit & Gradio

Dockerized YOLOv8 model with live camera detection using Streamlit as the main interface and Gradio embedded for camera handling.

## 🚀 Features

- **Real-time object detection** using YOLOv8
- **Live camera feed** via Gradio interface
- **Streamlit UI** for model configuration and monitoring
- **Dockerized** for easy deployment
- **GPU/CPU support** (CUDA, MPS, CPU)
- **Configurable** confidence thresholds and settings

## 📋 Prerequisites

- Docker and Docker Compose installed
- Camera/webcam access
- YOLOv8 model file (`last.pt`)

## 🛠️ Setup

### 1. Verify Files

Ensure you have the following files in the project directory:
- `last.pt` - Your trained YOLOv8 model
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker configuration
- `docker-compose.yml` - Docker Compose configuration
- `app.py` - Streamlit application
- `gradio_app.py` - Gradio camera interface
- `model_utils.py` - Model utilities
- `config.yaml` - Configuration file

### 2. Build and Run with Docker Compose

```bash
# Build and start the container
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### 3. Build and Run with Docker

```bash
# Build the image
docker build -t yolov8-app .

# Run the container
# For Linux (with camera device):
docker run -d \
  -p 8501:8501 \
  -p 7860:7860 \
  --device=/dev/video0 \
  -v $(pwd)/last.pt:/app/model.pt:ro \
  yolov8-app

# For macOS/Windows (with host networking):
docker run -d \
  --network host \
  -v $(pwd)/last.pt:/app/model.pt:ro \
  yolov8-app
```

## 📱 Usage

1. **Access the application:**
   - Open your browser and navigate to `http://localhost:8501`
   - You should see the Streamlit interface

2. **Start the camera:**
   - Click the **"Start Camera"** button
   - Allow camera permissions when prompted
   - The Gradio interface will appear embedded in the page

3. **Configure settings:**
   - Use the sidebar to adjust:
     - Confidence threshold
     - Device (CPU/CUDA/MPS)
     - View class names

4. **Stop the camera:**
   - Click **"Stop Camera"** when finished

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
model:
  path: "model.pt"
  device: "cpu"  # or "cuda", "mps"
  conf_threshold: 0.25
  iou_threshold: 0.45

camera:
  device_id: 0
  width: 640
  height: 480
```

## 🐛 Troubleshooting

### Camera Not Working

**Linux:**
- Ensure camera device is accessible: `ls -l /dev/video0`
- Add user to video group: `sudo usermod -aG video $USER`
- Use `--device=/dev/video0` flag in Docker

**macOS:**
- Grant Docker Desktop camera permissions in Settings
- Use `network_mode: host` in docker-compose.yml
- Or use `--network host` flag

**Windows:**
- Grant Docker Desktop camera permissions
- Use host networking mode

### Model Not Loading

- Verify `last.pt` file exists and is mounted correctly
- Check file permissions
- Ensure model file is compatible with Ultralytics YOLOv8

### Port Already in Use

- Change ports in `docker-compose.yml` or `config.yaml`
- Stop other services using ports 8501 or 7860

### Performance Issues

- Use GPU if available (set device to "cuda" in config)
- Reduce confidence threshold
- Lower camera resolution in config
- Use CPU optimization flags

## 📁 Project Structure

```
.
├── app.py              # Streamlit main application
├── gradio_app.py       # Gradio camera interface
├── model_utils.py      # Model loading and inference
├── config.yaml         # Configuration file
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker image definition
├── docker-compose.yml  # Docker Compose configuration
├── .dockerignore       # Docker ignore file
├── last.pt            # YOLOv8 model file
└── README.md          # This file
```

## 🔍 Architecture

```
┌─────────────────┐
│  Docker Container │
│                   │
│  ┌─────────────┐  │
│  │ Streamlit   │  │
│  │   (8501)    │  │
│  └──────┬──────┘  │
│         │         │
│  ┌──────▼──────┐  │
│  │   Gradio    │  │
│  │   (7860)    │  │
│  └──────┬──────┘  │
│         │         │
│  ┌──────▼──────┐  │
│  │  YOLOv8     │  │
│  │  Model      │  │
│  └─────────────┘  │
└─────────────────┘
         │
         ▼
    Camera Feed
```

## 🧪 Testing

Test the setup:

```bash
# Check if container is running
docker ps

# View logs
docker-compose logs -f

# Test model loading
docker exec -it yolov8-streamlit-gradio python -c "from model_utils import load_model; load_model()"
```

## 📝 Notes

- The model is cached in memory after first load for faster inference
- Gradio runs on an internal port and is embedded in Streamlit via iframe
- Camera access may require additional permissions depending on your OS
- For production use, consider adding authentication and HTTPS

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Review Docker logs: `docker-compose logs`
3. Verify all files are present and correctly configured

## 📄 License

This project uses YOLOv8 from Ultralytics. Please refer to their license for model usage.

