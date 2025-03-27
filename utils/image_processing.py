import cv2
import numpy as np
from PIL import Image

def preprocess_image(image):
    """
    Preprocess the image for YOLO model
    """
    # Convert PIL Image to OpenCV format if necessary
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Resize image if needed (YOLO can handle different sizes, but we can add preprocessing here)
    return image

def draw_detections(image, boxes, confidences, classes):
    """
    Draw bounding boxes and labels on the image
    """
    ripeness_levels = ['20%', '40%', '60%', '80%', '100%']
    
    for box, conf, cls in zip(boxes, confidences, classes):
        x1, y1, x2, y2 = box
        
        # Draw bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Add label
        label = f"Ripeness: {ripeness_levels[int(cls)]} ({conf:.2f})"
        cv2.putText(image, label, (int(x1), int(y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

def get_color_for_ripeness(ripeness_level):
    """
    Get color based on ripeness level
    """
    colors = {
        '20%': (0, 0, 255),    # Red
        '40%': (0, 128, 255),  # Orange
        '60%': (0, 255, 0),    # Green
        '80%': (255, 128, 0),  # Yellow
        '100%': (255, 0, 0)    # Dark Red
    }
    return colors.get(ripeness_level, (0, 255, 0))  # Default to green if level not found 