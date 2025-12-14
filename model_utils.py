"""
Model utilities for YOLOv8 inference
"""
import yaml
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from PIL import Image
import torch

# Global model cache
_model = None
_config = None


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    global _config
    if _config is None:
        with open(config_path, 'r') as f:
            _config = yaml.safe_load(f)
    return _config


def load_model(model_path=None, device=None):
    """
    Load YOLOv8 model with caching
    
    Args:
        model_path: Path to model file (defaults to config)
        device: Device to run on (defaults to config)
    
    Returns:
        YOLO model instance
    """
    global _model
    
    if _model is None:
        config = load_config()
        if model_path is None:
            model_path = config['model']['path']
        if device is None:
            device = config['model']['device']
        
        # Check if CUDA is available and requested
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
        
        # Check if MPS is available (Apple Silicon)
        if device == "mps" and not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU")
            device = "cpu"
        
        print(f"Loading model from {model_path} on device: {device}")
        _model = YOLO(model_path)
        _model.to(device)
        print("Model loaded successfully")
    
    return _model


def predict_image(image, conf_threshold=None, iou_threshold=None):
    """
    Run inference on a single image
    
    Args:
        image: PIL Image or numpy array
        conf_threshold: Confidence threshold (defaults to config)
        iou_threshold: IoU threshold (defaults to config)
    
    Returns:
        Annotated image (PIL Image) and results dict
    """
    model = load_model()
    config = load_config()
    
    # Use provided threshold or fall back to config
    if conf_threshold is None:
        conf_threshold = config['model']['conf_threshold']
    if iou_threshold is None:
        iou_threshold = config['model']['iou_threshold']
    
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Run inference
    results = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    # Get annotated image
    annotated_image = results[0].plot()
    annotated_image = Image.fromarray(annotated_image)
    
    # Extract detection info
    detections = []
    for box in results[0].boxes:
        detections.append({
            'class': int(box.cls[0]),
            'confidence': float(box.conf[0]),
            'bbox': box.xyxy[0].tolist()
        })
    
    return annotated_image, {
        'detections': detections,
        'num_detections': len(detections)
    }


def get_class_names():
    """Get class names from the model"""
    model = load_model()
    return model.names if hasattr(model, 'names') else {}


def clear_model_cache():
    """Clear the model cache (useful for reloading)"""
    global _model
    _model = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

