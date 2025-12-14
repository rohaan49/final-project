"""
Gradio interface for camera input and YOLOv8 inference
"""
import gradio as gr
from model_utils import predict_image, get_class_names, load_config
import threading
import queue
import time

# Configuration
config = load_config()
class_names = get_class_names()


def process_frame(image):
    """
    Process a single frame from camera
    
    Args:
        image: Image from Gradio camera component
    
    Returns:
        Annotated image with detections
    """
    if image is None:
        return None
    
    try:
        annotated_image, results = predict_image(image)
        return annotated_image
    except Exception as e:
        print(f"Error processing frame: {e}")
        return image


def create_gradio_interface():
    """
    Create and return Gradio interface for camera input
    """
    with gr.Blocks(title="YOLOv8 Live Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# YOLOv8 Real-Time Object Detection")
        gr.Markdown("Use your camera to detect objects in real-time")
        
        with gr.Row():
            with gr.Column():
                camera_input = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="pil",
                    label="Camera Input"
                )
                
                with gr.Row():
                    conf_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=config['model']['conf_threshold'],
                        step=0.05,
                        label="Confidence Threshold"
                    )
                    
                    device_radio = gr.Radio(
                        choices=["cpu", "cuda", "mps"],
                        value=config['model']['device'],
                        label="Device"
                    )
            
            with gr.Column():
                output_image = gr.Image(
                    type="pil",
                    label="Detections"
                )
                
                info_text = gr.Textbox(
                    label="Detection Info",
                    lines=5,
                    interactive=False
                )
        
        # Process frame when camera updates
        def process_with_conf(image, conf):
            if image is None:
                return None, ""
            try:
                annotated_image, results = predict_image(image, conf_threshold=conf)
                info = f"Detections: {results['num_detections']}\n"
                if results['detections']:
                    for det in results['detections'][:5]:  # Show first 5
                        class_name = class_names.get(det['class'], f"Class {det['class']}")
                        info += f"{class_name}: {det['confidence']:.2f}\n"
                return annotated_image, info
            except Exception as e:
                return image, f"Error: {str(e)}"
        
        # Process when camera input changes
        camera_input.change(
            process_with_conf,
            inputs=[camera_input, conf_slider],
            outputs=[output_image, info_text]
        )
        
        # Also process when confidence slider changes (to update existing frame)
        conf_slider.change(
            process_with_conf,
            inputs=[camera_input, conf_slider],
            outputs=[output_image, info_text]
        )
        
        gr.Markdown("### Instructions")
        gr.Markdown("""
        - Click on the camera input to start your webcam
        - Detections will appear in real-time on the right
        - Adjust the confidence threshold to filter detections
        - The model will automatically process each frame
        """)
    
    return demo


def run_gradio_server(port=7860, share=False):
    """
    Run Gradio server
    
    Args:
        port: Port to run Gradio on
        share: Whether to create a public link
    """
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True
    )


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    run_gradio_server(port=port)

