from ultralytics import YOLO
import os

def train_model():
    # Initialize a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='apple_ripeness_detection',
        patience=20,
        save=True,
        device='0'  # use GPU if available, else 'cpu'
    )

if __name__ == '__main__':
    train_model() 