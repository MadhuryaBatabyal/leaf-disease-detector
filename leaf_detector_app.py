# Live Leaf Disease, Dryness & Pest Detection App
# Built with Streamlit + YOLOv8 for real-time detection

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Try importing YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    st.warning("YOLOv8 not available. Install with: pip install ultralytics")

# Page configuration
st.set_page_config(
    page_title="🌿 Live Leaf Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌿 Live Leaf Disease & Pest Detection System")
st.markdown("Real-time detection of leaf diseases, dryness, and pest infestations using AI")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    detection_mode = st.selectbox(
        "Select Detection Mode",
        ["Upload Image", "Demo Mode"]
    )

    if YOLO_AVAILABLE:
        model_type = st.selectbox(
            "Model Type",
            ["Custom Leaf Model (best.pt)", "YOLOv8n COCO (default)"]
        )
        confidence_threshold = st.slider(
            "Confidence Threshold", 0.1, 1.0, 0.5, 0.05
        )

    show_statistics = st.checkbox("Show Detection Statistics", value=True)

    st.markdown("---")
    st.markdown("### 📚 Model Information")
    st.markdown("""
    **Supported (custom model):**
    - 🦠 Leaf Diseases (blight, spots, etc.)
    - ✅ Healthy Leaves
    """)

# Load model lazily and cache it
@st.cache_resource
def load_model(model_type: str):
    if not YOLO_AVAILABLE:
        return None

    if model_type == "Custom Leaf Model (best.pt)":
        # YOUR TRAINED WEIGHTS (make sure this file exists)
        model_path = "models/best.pt"
    else:
        # Fallback to generic YOLOv8
        model_path = "yolov8n.pt"

    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model '{model_path}': {e}")
        return None

def process_frame(frame, model=None, conf=0.5):
    """Run YOLO on a single frame."""
    if model and YOLO_AVAILABLE:
        results = model(frame, conf=conf)
        annotated_frame = results[0].plot()
        return annotated_frame, results[0]
    else:
        return frame, None

def extract_detections(results):
    """Extract detection count + labels."""
    if results is None or not hasattr(results, "boxes"):
        return {"total": 0, "labels": []}

    labels = []
    for box in results.boxes:
        if hasattr(box, "cls") and hasattr(box, "conf"):
            cls_id = int(box.cls)
            conf = float(box.conf)
            name = results.names.get(cls_id, f"class_{cls_id}")
            labels.append((name, conf))

    return {"total": len(labels), "labels": labels}

# Main interface
if detection_mode == "Upload Image":
    st.markdown("### 📷 Image Upload & Analysis")

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear image of a plant leaf"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### Original Image")
            st.image(image, use_column_width=True)

            if YOLO_AVAILABLE:
                st.markdown("#### Analysis Results")

                model = load_model(model_type)
                if model is None:
                    st.stop()

                try:
                    annotated_frame, results = process_frame(
                        frame,
                        model=model,
                        conf=confidence_threshold
                    )
                    st.image(
                        cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                        caption=f"Detection Results ({model_type})",
                        use_column_width=True
                    )
                except Exception as e:
                    st.error(f"Error during inference: {e}")
                    results = None
            else:
                st.info("Install YOLOv8: pip install ultralytics")
                results = None

        with col2:
            if show_statistics:
                st.markdown("#### Findings")

                det = extract_detections(results)
                st.metric("Total Detections", det["total"])

                if det["total"] > 0:
                    for name, conf in det["labels"]:
                        st.write(f"- **{name}** ({conf:.1%})")
                else:
                    st.info("No objects detected with current threshold.")

else:  # Demo Mode
    st.markdown("### 🎯 Demo Mode - Sample Output")
    demo_image = np.ones((480, 640, 3), dtype=np.uint8) * 240

    cv2.putText(
        demo_image,
        "Demo Mode",
        (180, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 200),
        3,
    )
    cv2.putText(
        demo_image,
        "Upload an image in 'Upload Image'",
        (40, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 100, 200),
        2,
    )

    st.image(
        cv2.cvtColor(demo_image, cv2.COLOR_BGR2RGB),
        caption="Sample output (no model run)",
        use_column_width=True
    )
    st.info("Switch to 'Upload Image' in the sidebar to use the model.")

# Footer
st.markdown("---")
st.markdown("""
**How this works**

- Uses a YOLOv8 model for object detection  
- For production, the app loads your trained weights from `models/best.pt`  
- Deployed via Streamlit (locally or on Streamlit Cloud)

*Built by Madhurya Batabyal – Leaf Disease Detector*
""")
