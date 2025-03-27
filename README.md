# Apple Ripeness Detection

This project uses YOLO (You Only Look Once) for detecting and classifying apple ripeness levels. The system can identify five different ripeness levels: 20%, 40%, 60%, 80%, and 100%.

## Features

- Real-time apple ripeness detection
- Five ripeness level classifications
- Interactive web interface using Streamlit
- YOLO-based object detection

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main Streamlit application
- `model/`: Directory containing the YOLO model
- `utils/`: Utility functions for image processing
- `data/`: Dataset directory
- `requirements.txt`: Project dependencies

## Usage

1. Launch the application using the command above
2. Upload an image of apples
3. The system will detect apples and classify their ripeness level
4. Results will be displayed with bounding boxes and ripeness percentages 