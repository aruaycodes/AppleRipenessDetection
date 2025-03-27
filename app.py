import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# Set page config
st.set_page_config(
    page_title="Apple Ripeness Detection",
    page_icon="🍎",
    layout="wide"
)

# Title and description
st.title("🍎 Apple Ripeness Detection")
st.markdown("""
This application uses YOLO to detect and classify apple ripeness levels.
Upload an image of apples to get started!
""")

# Initialize YOLO model
@st.cache_resource
def load_model():
    model = YOLO('model/best.pt')  # We'll need to train and save the model
    return model

# Function to process image
def process_image(image, model):
    # Run YOLO detection
    results = model(image)
    
    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            # Get confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Draw bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Add label
            ripeness_levels = ['20%', '40%', '60%', '80%', '100%']
            label = f"Ripeness: {ripeness_levels[cls]} ({conf:.2f})"
            cv2.putText(image, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Detected Apples")
        # Convert PIL image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        try:
            # Load model and process image
            model = load_model()
            processed_image = process_image(image_cv, model)
            
            # Convert back to RGB for display
            processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            st.image(processed_image_rgb, use_column_width=True)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please make sure the model is properly trained and saved in the model directory.")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit and YOLO</p>
</div>
""", unsafe_allow_html=True) 