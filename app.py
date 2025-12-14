"""
Streamlit application with embedded Gradio camera interface
"""
import streamlit as st
import subprocess
import threading
import time
import requests
from pathlib import Path
from model_utils import load_model, get_class_names, load_config

# Page configuration
st.set_page_config(
    page_title="YOLOv8 Live Detection",
    page_icon="🎥",
    layout="wide"
)

# Initialize session state
if 'gradio_started' not in st.session_state:
    st.session_state.gradio_started = False
if 'gradio_process' not in st.session_state:
    st.session_state.gradio_process = None


def start_gradio_server():
    """Start Gradio server in background"""
    if not st.session_state.gradio_started:
        try:
            # Start Gradio server
            process = subprocess.Popen(
                ["python", "gradio_app.py", "7860"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            st.session_state.gradio_process = process
            st.session_state.gradio_started = True
            
            # Wait for server to be ready
            max_retries = 30
            for i in range(max_retries):
                try:
                    response = requests.get("http://localhost:7860", timeout=2)
                    if response.status_code == 200:
                        return True
                except:
                    time.sleep(0.5)
            return False
        except Exception as e:
            st.error(f"Error starting Gradio server: {e}")
            return False
    return True


def stop_gradio_server():
    """Stop Gradio server"""
    if st.session_state.gradio_process:
        st.session_state.gradio_process.terminate()
        st.session_state.gradio_process.wait()
        st.session_state.gradio_process = None
    st.session_state.gradio_started = False


# Main UI
st.title("🎥 YOLOv8 Real-Time Object Detection")
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Load model info
    try:
        config = load_config()
        model = load_model()
        class_names = get_class_names()
        
        st.success("✅ Model loaded successfully")
        st.info(f"**Device:** {config['model']['device']}")
        st.info(f"**Classes:** {len(class_names)}")
        
        # Display class names if available
        if class_names:
            with st.expander("📋 Class Names"):
                for idx, name in class_names.items():
                    st.text(f"{idx}: {name}")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
    
    st.markdown("---")
    st.header("📊 Model Settings")
    
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=config['model']['conf_threshold'],
        step=0.05
    )
    
    device_options = ["cpu", "cuda", "mps"]
    current_device = config['model']['device']
    device_index = device_options.index(current_device) if current_device in device_options else 0
    
    device = st.radio(
        "Device",
        options=device_options,
        index=device_index
    )
    
    if st.button("🔄 Reload Model"):
        from model_utils import clear_model_cache
        clear_model_cache()
        model = load_model(device=device)
        st.rerun()

# Main content area
st.markdown("### 📹 Live Camera Detection")

# Start Gradio server
if st.button("🚀 Start Camera", type="primary"):
    if start_gradio_server():
        st.success("✅ Gradio server started!")
        st.rerun()
    else:
        st.error("❌ Failed to start Gradio server")

if st.session_state.gradio_started:
    # Embed Gradio interface
    gradio_url = "http://localhost:7860"
    
    # Check if server is running
    try:
        response = requests.get(gradio_url, timeout=2)
        if response.status_code == 200:
            # Embed Gradio using iframe
            st.components.v1.iframe(
                src=gradio_url,
                height=800,
                scrolling=True
            )
            
            if st.button("🛑 Stop Camera"):
                stop_gradio_server()
                st.rerun()
        else:
            st.warning("⚠️ Gradio server is starting... Please wait.")
            time.sleep(2)
            st.rerun()
    except requests.exceptions.RequestException:
        st.warning("⚠️ Waiting for Gradio server to start...")
        time.sleep(2)
        st.rerun()

else:
    st.info("👆 Click 'Start Camera' to begin live detection")

# Footer
st.markdown("---")
st.markdown("""
### 📝 Instructions
1. Click **"Start Camera"** to launch the camera interface
2. Allow camera permissions when prompted
3. The Gradio interface will appear below with live detection
4. Adjust settings in the sidebar to customize detection parameters
5. Click **"Stop Camera"** when finished

### 🔧 Troubleshooting
- **Camera not working?** Make sure camera permissions are granted
- **Slow performance?** Try using CPU device or reducing confidence threshold
- **Model not loading?** Check that `last.pt` file is in the correct location
""")

