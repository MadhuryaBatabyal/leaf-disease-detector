"""
🐛 Pest Detection App - Live Camera + Image Upload
Uses YOUR trained YOLOv8 model (models/best.pt)
By Madhurya Batabyal
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

# YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="🐛 Pest Detection",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🐛 Real-time Pest Detection System")
st.markdown("**Detect pests on plants using live camera or uploaded images**")
st.markdown("*Powered by custom YOLOv8 model (trained on 23K+ images)*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Detection Settings")
    
    # Model selection
    model_path = st.selectbox(
        "Model",
        ["models/best.pt (Custom Pest Model)", "yolov8n.pt (Default)"],
        index=0
    )
    
    conf_threshold = st.slider("Confidence", 0.1, 1.0, 0.4, 0.05)
    
    st.markdown("---")
    st.markdown("### 📊 Stats")
    if 'detections' in st.session_state:
        st.metric("Total Pests", st.session_state.detections)

# Load model (cached)
@st.cache_resource
def load_model(path):
    if not YOLO_AVAILABLE:
        return None
    try:
        model = YOLO(path)
        st.success(f"✅ Loaded {path}")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load {path}: {e}")
        return None

def process_image(image, model, conf):
    """Run YOLO detection on image."""
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    if model:
        results = model(frame, conf=conf, verbose=False)
        annotated = results[0].plot()
        detections = len(results[0].boxes) if results[0].boxes is not None else 0
        
        # Update stats
        if 'detections' not in st.session_state:
            st.session_state.detections = 0
        st.session_state.detections = detections
        
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), detections, results
    else:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 0, None

# Main tabs
tab1, tab2 = st.tabs(["📷 Live Camera", "🖼️ Upload Image"])

with tab1:
    st.header("Live Pest Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        camera_input = st.camera_input("🖥️ Take photo with camera")
        
        if camera_input:
            image = Image.open(camera_input)
            
            model = load_model(model_path)
            result_img, num_detections, results = process_image(image, model, conf_threshold)
            
            st.image(result_img, caption=f"Pest Detection Results ({num_detections} pests)", use_column_width=True)
            
            if num_detections > 0 and results:
                st.markdown("### 🐛 Detected Pests:")
                for i, box in enumerate(results[0].boxes):
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    name = results[0].names[cls_id]
                    st.write(f"**{name}: {conf:.1%}**")
    
    with col2:
        st.info("👈 Take photo → see pests instantly!")
        if 'detections' in st.session_state:
            st.metric("Session Pests", st.session_state.detections)

with tab2:
    st.header("Upload Pest Images")
    
    uploaded_file = st.file_uploader(
        "Upload leaf/plant image",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            model = load_model(model_path)
            result_img, num_detections, results = process_image(image, model, conf_threshold)
            
            st.image(result_img, caption=f"Detection Results ({num_detections} pests)", use_column_width=True)
        
        with col2:
            st.metric("Pests Found", num_detections)
            
            if num_detections > 0 and results:
                st.markdown("### Detection Details:")
                for box in results[0].boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    name = results[0].names[cls_id]
                    st.success(f"{name}: {conf:.1%}")

# Demo section
if not YOLO_AVAILABLE:
    st.error("Install YOLOv8: `pip install ultralytics`")
    st.stop()

st.markdown("---")
st.markdown("""
### 🚀 About This App

**Model:** Custom YOLOv8 trained on 23K+ plant disease images  
**Dataset:** [Roboflow Plant Disease Detection](https://universe.roboflow.com/floragenic-9v9os/plant-disease-detection-3anip/1)  
**Features:** Live camera + image upload + real-time stats  

**By Madhurya Batabyal | Portfolio Project 2026**
""")

# Initialize session state
if 'detections' not in st.session_state:
    st.session_state.detections = 0
