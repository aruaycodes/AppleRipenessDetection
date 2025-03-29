import streamlit as st
from ultralytics import YOLO
import cv2
import os
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(
    page_title="Apple Ripeness Detection",
    page_icon="🍎",
    layout="wide"
)

# Title and description
st.title("🍎 Apple Ripeness Detection")
st.markdown("""
This app uses AI to detect apple ripeness levels and provides recommendations for harvesting.
Upload an image of an apple to get started!
""")

# Load the model
@st.cache_resource
def load_model():
    model = YOLO('runs/detect/apple_ripeness_detection9/weights/best.pt')
    return model

# Function to get recommendations based on ripeness
def get_recommendations(ripeness):
    recommendations = {
        '20%_ripe': {
            'message': '🍃 Still Growing',
            'recommendation': 'Leave the apple on the tree. It needs more time to develop its flavor and nutrients.',
            'color': 'green'
        },
        '40%_ripe': {
            'message': '🌱 Early Development',
            'recommendation': 'The apple is in early development. Continue monitoring its growth.',
            'color': 'light-green'
        },
        '60%_ripe': {
            'message': '🌿 Mid-Ripeness',
            'recommendation': 'The apple is halfway ripe. You can start planning for harvest soon.',
            'color': 'yellow'
        },
        '80%_ripe': {
            'message': '🍎 Almost Ready',
            'recommendation': 'The apple is nearly ripe. Prepare for harvest in the next few days.',
            'color': 'orange'
        },
        '100%_ripe': {
            'message': '🍎 Ready to Harvest!',
            'recommendation': 'Harvest immediately for the best flavor and texture.',
            'color': 'red'
        }
    }
    return recommendations.get(ripeness, {
        'message': '❓ Unknown',
        'recommendation': 'Unable to determine ripeness level.',
        'color': 'gray'
    })

# Function to process image and get predictions
def process_image(model, image):
    # Run inference
    results = model.predict(
        source=image,
        conf=0.25,
        save=False  # Don't save to disk
    )
    
    # Get the first result
    result = results[0]
    
    # Get the image with predictions
    img = result.plot()
    
    # Convert to PIL Image
    img = Image.fromarray(img)
    
    # Get predictions
    predictions = []
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_names = ['20%_ripe', '40%_ripe', '60%_ripe', '80%_ripe', '100%_ripe']
        ripeness = class_names[cls]
        recommendations = get_recommendations(ripeness)
        predictions.append({
            'ripeness': ripeness,
            'confidence': conf,
            'recommendations': recommendations
        })
    
    return img, predictions

# Load the model
model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        image = Image.open(uploaded_file)
        # Resize image to a reasonable size
        max_size = 400
        ratio = max_size / max(image.size)
        new_size = tuple([int(x * ratio) for x in image.size])
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        st.image(resized_image, use_column_width=True)
    
    # Process the image
    with st.spinner("Analyzing apple ripeness..."):
        result_img, predictions = process_image(model, image)
        
        with col2:
            st.subheader("Analysis Result")
            # Resize result image to match original
            resized_result = result_img.resize(new_size, Image.Resampling.LANCZOS)
            st.image(resized_result, use_column_width=True)
            
            # Display predictions with recommendations
            for pred in predictions:
                # Create a container for each prediction
                with st.container():
                    # Add a colored box for the ripeness level
                    st.markdown(f"""
                        <div style='background-color: {pred['recommendations']['color']}; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                            <h3 style='margin: 0;'>{pred['recommendations']['message']}</h3>
                            <p style='margin: 5px 0;'>Ripeness Level: {pred['ripeness']}</p>
                            <p style='margin: 5px 0;'>Confidence: {pred['confidence']:.2%}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Add recommendation
                    st.info(pred['recommendations']['recommendation'])

# Add some information about the model
with st.expander("About the Model"):
    st.markdown("""
    ### Model Details
    - Trained on YOLOv8
    - 5 ripeness levels: 20%, 40%, 60%, 80%, 100%
    - mAP50: 98.9%
    - Average inference time: ~20ms per image
    
    ### How to Use
    1. Upload an image of an apple
    2. Wait for the AI to analyze the ripeness
    3. View the recommendations and confidence scores
    """)

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit and YOLO</p>
</div>
""", unsafe_allow_html=True) 