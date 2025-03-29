# Apple Ripeness Detection

An AI-powered application that detects and analyzes apple ripeness levels using YOLOv8. The application provides real-time recommendations for harvesting based on the detected ripeness level.

## Features

- Real-time apple ripeness detection
- 5 ripeness levels (20%, 40%, 60%, 80%, 100%)
- User-friendly recommendations for harvesting
- Interactive web interface using Streamlit
- High accuracy (98.9% mAP50)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AppleRipenessDetection.git
cd AppleRipenessDetection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload an image of an apple to get started!

## Model Details

- Architecture: YOLOv8
- Training Dataset: Custom apple ripeness dataset
- Performance Metrics:
  - mAP50: 98.9%
  - Average inference time: ~20ms per image

## Ripeness Levels and Recommendations

- 20%: Still Growing - Leave on tree for further development
- 40%: Early Development - Continue monitoring growth
- 60%: Mid-Ripeness - Start planning for harvest
- 80%: Almost Ready - Prepare for harvest in coming days
- 100%: Ready to Harvest - Harvest immediately for best quality

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 