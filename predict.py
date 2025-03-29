from ultralytics import YOLO
import cv2
import os

def predict_ripeness(model_path, image_path):
    """
    Predict apple ripeness using the trained model
    Args:
        model_path: Path to the trained model weights
        image_path: Path to the input image
    """
    # Load the trained model
    model = YOLO(model_path)

    # Run inference on the image
    results = model.predict(
        source=image_path,
        conf=0.25,  # Confidence threshold
        save=True,  # Save results
        save_txt=True,  # Save predictions in txt format
        project='predictions',  # Save results to predictions directory
        name='apple_ripeness'  # Name of the results directory
    )

    # Process and display results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Get class name (ripeness level)
            class_names = ['20%_ripe', '40%_ripe', '60%_ripe', '80%_ripe', '100%_ripe']
            ripeness = class_names[cls]
            
            print(f"Detected apple ripeness: {ripeness} (Confidence: {conf:.2f})")

def main():
    # Path to your best trained model
    model_path = 'runs/detect/apple_ripeness_detection9/weights/best.pt'
    
    # Create predictions directory if it doesn't exist
    os.makedirs('predictions/apple_ripeness', exist_ok=True)
    
    # You can either specify a single image or a directory of images
    image_path = input("Enter the path to an image or directory of images: ")
    
    if not os.path.exists(image_path):
        print(f"Error: Path '{image_path}' does not exist")
        return
    
    predict_ripeness(model_path, image_path)
    print("\nPredictions completed! Check the 'predictions/apple_ripeness' directory for results.")

if __name__ == '__main__':
    main() 