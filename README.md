# Apple Ripeness Detection

An AI-powered system that automatically detects and classifies apple ripeness levels using YOLOv8. This project helps farmers optimize harvest timing and improve crop management.

## Features

- Real-time apple ripeness detection
- Five ripeness levels (20%, 40%, 60%, 80%, 100%)
- User-friendly web interface
- High accuracy (98.9% mAP50)
- Instant processing (~20ms per image)
- Smart recommendations for harvest timing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/AppleRipenessDetection.git
cd AppleRipenessDetection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
AppleRipenessDetection/
├── app.py              # Streamlit web interface
├── predict.py          # Prediction script
├── train.py           # Training script
├── data.yaml          # Dataset configuration
├── datasets/          # Dataset directory
│   └── data/
│       ├── train/    # Training images and labels
│       └── val/      # Validation images and labels
├── predictions/       # Prediction results
├── runs/             # Training results and model weights
└── requirements.txt  # Project dependencies
```

## Running the Project

### 1. Web Interface (Recommended)

Run the Streamlit app:
```bash
streamlit run app.py
```
Then open your browser and navigate to `http://localhost:8501`

### 2. Command Line Prediction

Run predictions on images or directories:
```bash
python predict.py
```
When prompted, enter the path to an image or directory of images.

### 3. Training the Model

To train the model on your own dataset:
```bash
python train.py
```

## Testing the Model

### Using Validation Dataset

The project includes a validation dataset with various apple ripeness levels. To test the model:

1. **Using Web Interface**:
   - Navigate to `datasets/data/val/images/`
   - Try these sample images:
     - `20%_citra10.jpg` (Early growth)
     - `40%_citra13.jpg` (Early development)
     - `60%_citra11.jpg` (Mid-ripeness)
     - `80%_citra12.jpg` (Almost ready)
     - `100%_citra11.jpg` (Ready to harvest)

2. **Using Command Line**:
   ```bash
   # Test single validation image
   python predict.py
   # Enter: datasets/data/val/images/100%_citra11.jpg

   # Test all validation images
   python predict.py
   # Enter: datasets/data/val/images/
   ```

### Expected Results

When testing with validation images, you should see:
- Accurate ripeness level detection
- Confidence scores above 80%
- Clear bounding boxes around apples
- Specific harvest recommendations

## Results Location

- **Training Results**: `runs/detect/apple_ripeness_detection9/`
  - Best model weights: `weights/best.pt`
  - Training logs and metrics

- **Prediction Results**: `predictions/`
  - Each prediction run creates a new directory
  - Contains processed images and detection information

## Model Performance

- mAP50: 98.9%
- mAP50-95: 93.4%
- Precision: 86.7%
- Recall: 93.4%
- Inference Speed: ~20ms per image

## Usage Examples

1. **Using the Web Interface**:
   - Upload an apple image
   - View real-time ripeness analysis
   - Get harvest recommendations

2. **Using Command Line**:
   ```bash
   # Predict single image
   python predict.py
   # Enter: datasets/data/val/images/100%_citra100.jpg

   # Predict directory of images
   python predict.py
   # Enter: datasets/data/val/images/
   ```

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- Streamlit
- OpenCV
- Pillow

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 