# 🚭 AI-Powered Vape Marketing Detector

## 🔍 Overview

This tool uses AI to **detect and flag vape product images** that may target underage audiences through marketing strategies such as:

- Young brand ambassadors (via face detection and age classification)
- Cartoon imagery
- Bright, colorful visuals
- Vape devices and related content

It integrates computer vision models (YOLO, MTCNN, AgeNet) into a **Flask web dashboard**, allowing users to upload and analyze brand image datasets from local directories or AWS S3. Ideal for researchers, regulators, and developers working on digital health compliance.

---

## 📁 Project Structure

```
├── app.py # Main Flask app
├── dashboard.html # Web dashboard template
├── models/
│ ├── face_detection.py
│ ├── age_classification.py
│ ├── cartoon_detection.py
│ └── vape_type_detection.py
│ └── image_processing.py # Image analysis pipeline
├── static/
│ └── visualization_images/ # Output images for dashboard
├── requirements.txt
└── README.md
```


📦 Output
- Flagged images with:
- Face and age estimation
- Vape detection
- Cartoon elements
- Chart visualizations for readability, brightness, etc.
- Downloadable CSV (optional)

🧠 Models Used
- Face Detection: MTCNN
- Age Estimation: AgeNet (Caffe-based / CNN)
- Cartoon Detection: Custom-trained ResNet classifier
- Vape Product Detection: YOLOv5 custom model
- HistGradientModel: Classifier to flag the image

### Steps to Run This Project

1. Clone the Repository
```
git clone https://github.com/MayankG514/VapeWatchers.git
cd VapeWatchers
```

2. Set Up a Virtual Environment
```
python -m venv venv
source venv/bin/activate   # For Windows: venv\Scripts\activate
```

3. Install the Required Dependencies
```
pip install -r requirements.txt
```

4. Add AWS Credentials (If Using S3)
```
Change config.py file with appropriate keys
```

5. Run the Flask Application
```
python app.py
```


