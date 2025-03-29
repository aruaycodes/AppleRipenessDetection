from ultralytics import YOLO
import os

def train_model():
    # Initialize a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data='data.yaml',
        epochs=30,
        imgsz=416,
        batch=8,
        name='apple_ripeness_detection',
        patience=5,
        save=True,  # save checkpoints
        save_period=1,  # save every epoch
        device='cpu'  # use CPU for training
    )

if __name__ == '__main__':
    train_model() 