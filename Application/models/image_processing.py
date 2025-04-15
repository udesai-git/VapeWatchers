import io
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from collections import namedtuple
from mtcnn import MTCNN
from ultralytics import YOLO
import boto3
from io import BytesIO
from models.easy_ocr import main as easy_ocr_main
from models.ocr_csv_eda import main as ocr_analysis_main


# Load models
AGE_LIST = ['(0-2)', '(4-6)', '(8-13)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
prototxt_path = "/Users/mayankgrover/Documents/DAEN_698/Vape_Regulation_Project/models/deploy_age.prototxt"
model_path = "/Users/mayankgrover/Documents/DAEN_698/Vape_Regulation_Project/models/age_net.caffemodel"
age_model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

cartoon_model = YOLO('/Users/mayankgrover/Documents/DAEN_698/Vape_Regulation_Project/models/yolov8_trained.pt')
vape_model = YOLO('/Users/mayankgrover/Documents/DAEN_698/Vape_Regulation_Project/models/best.pt')


print("🔄 Starting image processing...")
data = []


AWS_ACCESS_KEY="ASIARMWYWT5W52CZXXBI"
AWS_SECRET_KEY="c9orzaCeF8NsHy+pITmG3gYk3knbRELOTTtT7+4X"
aws_session_token="IQoJb3JpZ2luX2VjEKX//////////wEaCXVzLXdlc3QtMiJHMEUCIQCj0tLOhKjnDfCY5Ml6tLpDdDf1Es92KFu/toW1dJ1VnQIgY/tru9CTU9edVaqZzUOR9s5o3piJ3MlZCwekL8MvqX4qqgIILhABGgwwOTYwMTc0ODk3NzMiDB+v1RCHfTDeHYzJOyqHAqVuxYR4tNpwlKeEOT8aKNGQCGqPudL0QLfJtYrZftffi7EtYGVhtU2UCUaK1niZFY+ckpkCDVZsXyPtLxGdVZofkbyV12rEHwF/rCRmh8IKGCxS+PNnwj4bPeJbIsHrrSVyYKRV3NhoqDDESLMOdczgA3TcknuAmgxR6tUBt7e/4hEXPl5LatB844kd8g7i9fmMS4gwOE+6J918OhiWvuhzL7IlfQ3vRgcEGCoQL46PkuqGjfM0zI0EPVMtWDB1bD377L4kSoQ5KjrjREF5ZAaQD+Uq2pw3/lb3F5ZnPmYVELqnUd1h1Ytki4JXgodah7mprUFezUi/gte3klabqRcz1xdg6E9oMJ+l+b8GOp0Bugt5jI36GV6EC/zRUR1qOXK/a4dZns++0e81jPhn9kCY6Hk2flAlpdZDyKtQBrWX7Uzc8f0z96PtJF8BoEJjESlM48POZHofuAY+ap5ib4XnqN2dDO3IrTLjNQWLyr0+WLVA9oc6UNA9zRARKTfqQNpefdyx5+G99levH8OFp/GinVwV/HR7A+SWiVS2B+UGkUvbXhT5cFUu4LP0rw=="


s3_session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    aws_session_token=aws_session_token
)

s3_client = s3_session.client('s3')

bucket_name = 'vapewatchers-2025'
prefix = 'MarketingImages/'

from scipy.spatial import distance

# Named tuple for brightness levels
BLevel = namedtuple("BLevel", ['brange', 'bval'])
_blevels = [
    BLevel(brange=range(0, 24), bval=0),
    BLevel(brange=range(23, 47), bval=1),
    BLevel(brange=range(46, 70), bval=2),
    BLevel(brange=range(69, 93), bval=3),
    BLevel(brange=range(92, 116), bval=4),
    BLevel(brange=range(115, 140), bval=5),
    BLevel(brange=range(139, 163), bval=6),
    BLevel(brange=range(162, 186), bval=7),
    BLevel(brange=range(185, 209), bval=8),
    BLevel(brange=range(208, 232), bval=9),
    BLevel(brange=range(231, 256), bval=10),
]

from scipy.spatial import distance


COLORS = {
    # VIBGYOR base
    "Violet": (148, 0, 211),
    "Indigo": (75, 0, 130),
    "Blue": (0, 0, 255),
    "Green": (0, 255, 0),
    "Yellow": (255, 255, 0),
    "Orange": (255, 165, 0),
    "Red": (255, 0, 0),

    # Bright variants
    "Bright Blue": (135, 206, 250),
    "Bright Green": (144, 238, 144),
    "Bright Yellow": (255, 255, 102),
    "Bright Orange": (255, 200, 0),
    "Bright Red": (255, 99, 71),
    "Bright Pink": (255, 105, 180),

    # Dull/Muted variants
    "Dull Blue": (100, 100, 150),
    "Dull Green": (85, 107, 47),
    "Dull Yellow": (204, 204, 102),
    "Dull Orange": (210, 105, 30),
    "Dull Red": (139, 0, 0),
    "Dull Pink": (219, 112, 147),
    
    "Turquoise": (64, 224, 208),
    "Cyan": (0, 255, 255),
    "Aqua": (0, 255, 255),
    "Teal": (0, 128, 128),
    "Magenta": (255, 0, 255),
    "Pink": (255, 192, 203),
    "Hot Pink": (255, 105, 180),
    "Fuchsia": (255, 20, 147),
    "Maroon": (128, 0, 0),
    "Burgundy": (128, 0, 32),
    "Peach": (255, 218, 185),
    "Beige": (245, 245, 220),
    "Coral": (255, 127, 80),
    "Salmon": (250, 128, 114),

    "White": (255, 255, 255),
    "Black": (0, 0, 0),
    "Gray": (128, 128, 128),
    "Light Gray": (211, 211, 211),
    "Dark Gray": (64, 64, 64),
    "Brown": (139, 69, 19),
    "Tan": (210, 180, 140),
    "Olive": (128, 128, 0),
    "Khaki": (195, 176, 145),
    "Neon Green": (57, 255, 20),
    "Neon Pink": (255, 20, 147),
    "Neon Blue": (77, 77, 255),
    "Neon Yellow": (204, 255, 0),
    "Neon Orange": (255, 153, 51),

}

def closest_vibgyor_color(color):
    """Finds the closest VIBGYOR color based on Euclidean distance."""
    return min(COLORS, key=lambda vib_color: distance.euclidean(color, COLORS[vib_color]))


def extract_Dominant_COLOR(image_input, top_n=5):
    """Extracts the most common color and maps it to VIBGYOR.

    Accepts either:
    - A local image path (str)
    - A NumPy image array (for S3 images)
    """
    if isinstance(image_input, str):
        # Read from local file
        image = cv2.imread(image_input)
    elif isinstance(image_input, np.ndarray):
        # Use image array directly (from S3)
        image = image_input
    else:
        raise ValueError("Unsupported image format. Provide a file path or a NumPy array.")

    if image is None:
        raise ValueError("Error: Could not load the image. Check the file path or S3 object.")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)

    # Count most common colors
    color_counts = Counter(map(tuple, pixels))
    most_common_colors = color_counts.most_common(top_n)

    # Map colors to VIBGYOR
    mapped_colors = [closest_vibgyor_color(color[0]) for color in most_common_colors]

    # Return the most frequently occurring VIBGYOR color
    return max(set(mapped_colors), key=mapped_colors.count)



def detect_level(h_val):
    h_val = int(h_val)
    for blevel in _blevels:
        if h_val in blevel.brange:
            return blevel.bval
    raise ValueError("Brightness Level Out of Range")


def get_img_avg_brightness(image_input):
    """
    Computes the average brightness of an image.
    - Accepts either a local image path (str) or an image as a NumPy array.
    """
    if isinstance(image_input, str):
        # Read from local file
        img = cv2.imread(image_input)
    elif isinstance(image_input, np.ndarray):
        # Directly use NumPy image (from S3)
        img = image_input
    else:
        raise ValueError("Unsupported image format. Provide a file path or a NumPy array.")

    if img is None:
        raise ValueError("Error: Could not load the image. Check the file path or S3 object.")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    return int(np.average(v.flatten()))


def get_company_name(image_path):
    path_parts = image_path.split('/')
    if "MarketingImages" in path_parts:
        idx = path_parts.index("MarketingImages")
        if idx + 1 < len(path_parts):
            return path_parts[idx + 1]
    return "Unknown"


def detect_age_with_mtcnn(img):
    detector = MTCNN()
    faces = detector.detect_faces(img)
    results = []
    for face in faces:
        x, y, w, h = face['box']
        face_img = img[y:y + h, x:x + w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), (78.426, 87.769, 114.896), swapRB=False)
        age_model.setInput(blob)
        age_preds = age_model.forward()
        age = AGE_LIST[age_preds[0].argmax()]
        results.append((x, y, w, h, age))
    return results

def detect_cartoon_objects(img):
    results = cartoon_model.predict(img)
    return any(cartoon_model.names[int(box[5])] == 'cartoon' for r in results for box in r.boxes.data)

def detect_vape_type(img):
    results = vape_model.predict(img)
    best_vape = None
    highest_conf = 0
    for r in results:
        for box in r.boxes.data:
            class_id = int(box[5])
            confidence = float(box[4])
            class_name = vape_model.names[class_id]
            if confidence > highest_conf:
                highest_conf = confidence
                best_vape = class_name
    return best_vape if best_vape else "None"

def label_data(row):
    score = 0

    # -------------------------
    # Image-based cues with weights
    # -------------------------
    if row['Face'] == 1 and row['Age'] in ['(0-2)', '(4-6)', '(8-13)', '(15-20)', '(25-32)']:
        score += 2  # young face detected

    if row['Cartoon'] == 1:
        score += 2  # cartoon is a strong youth indicator

    if row['Brightness_Level'] > 5:
        score += 1  # brighter images may appeal to younger viewers

    if row['Dominant_COLOR'] in [
        'Red', 'Orange', 'Yellow',
        'Bright Red', 'Bright Orange', 'Bright Yellow',
        'Neon Red', 'Neon Orange', 'Neon Yellow',
        'Hot Pink', 'Fuchsia', 'Bright Pink'
    ]:
        score += 1.5  # attention-grabbing colors

    # -------------------------
    # Text-based cues with weights
    # -------------------------
    if row.get('youth_appeal_score', 0) > 0.5:
        score += 2  # directly models youth-oriented language

    if row.get('readability_youth_score', 0) > 0.5:
        score += 1  # simple language

    if row.get('special_chars_youth_score', 0) > 0.5:
        score += 1  # emojis, hashtags, etc.

    if row.get('contains_warning', 1) == 0 and row.get('Vape_Type', 'N/A') not in [None, '', 'N/A']:
        score += 2  # high alert: vape mentioned but no warning

    # -------------------------
    # Final decision based on score
    # -------------------------
    if score >= 3:
        return 1
    else:
        return 0

    
def analyze_and_save_results(s3_bucket_name, prefix):
    print(f"📂 Accessing S3 bucket: {s3_bucket_name} with prefix: {prefix}")
    
    data = []
    continuation_token = None

    while True:
        list_kwargs = {'Bucket': s3_bucket_name, 'Prefix': prefix}
        if continuation_token:
            list_kwargs['ContinuationToken'] = continuation_token

        response = s3_client.list_objects_v2(**list_kwargs)
        if "Contents" not in response:
            print(f"❌ No files found in {prefix} on bucket {s3_bucket_name}")
            break

        for obj in response["Contents"]:
            image_path = obj["Key"]
            if not image_path.lower().endswith((".png", ".jpg", ".jpeg")):
                continue  # Skip non-image files

            print(f"📌 Processing: {image_path}")

            try:
                brand_name = image_path.split("/")[1]
                image_name = image_path.split("/")[-1]

                img_data = s3_client.get_object(Bucket=s3_bucket_name, Key=image_path)["Body"].read()
                image_array = np.asarray(bytearray(img_data), dtype=np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"❌ Error loading image: {image_path}")
                    continue

                age_results = detect_age_with_mtcnn(img)
                face_detected = len(age_results) > 0
                detected_age = age_results[0][4] if face_detected else "N/A"
                cartoon_detected = detect_cartoon_objects(img)
                vape_type = detect_vape_type(img)
                brightness_value = get_img_avg_brightness(img)
                brightness_level = detect_level(brightness_value)
                colors = extract_Dominant_COLOR(img, top_n=3)

                data.append({
                    "image_name": image_name,
                    "Brand": brand_name,
                    "Face": "Yes" if face_detected else "No",
                    "Age": detected_age,
                    "Cartoon": "Yes" if cartoon_detected else "No",
                    "Vape_Type": vape_type,
                    "Brightness_Level": brightness_level,
                    "Dominant_COLOR": colors
                })

            except Exception as e:
                print(f"❌ Error processing {image_path}: {e}")
                continue

        # Check if there are more objects to list
        if response.get("IsTruncated"):  # More results available
            continuation_token = response.get("NextContinuationToken")
        else:
            break

    # Save results to S3
    df = pd.DataFrame(data)

    output_csv_key = "marketing_image_processed.csv"
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=s3_bucket_name, Key=output_csv_key, Body=csv_buffer.getvalue(), ContentType="text/csv")

    print(f"✅ CSV successfully saved to s3://{s3_bucket_name}/{output_csv_key}")
    
    dataset_path = f"s3://{s3_bucket_name}/{prefix}"
    output_dir = "output"
    brand_limit = None  # or a value like 5 if you want to limit
    
    print("Calling OCR Model")
    # Call the easy_ocr main method
    easy_ocr_main( dataset_path,output_dir,aws_access_key=AWS_ACCESS_KEY, aws_secret_key=AWS_SECRET_KEY,aws_session_token=aws_session_token )
    print("OCR Model just finished")
   
    print("calling text analysis on ocr extracted text")
    ocr_analysis_main(s3_client)
    print("finished text analysis on ocr extracted text")
    
     
    key_image_models = 'marketing_image_processed.csv'
    key_ocr_model = 'vape_ocr_text_analysis.csv'
    
    response_image_models = s3_client.get_object(Bucket=s3_bucket_name, Key=key_image_models)
    image_model_csv = pd.read_csv(BytesIO(response_image_models['Body'].read()))
    response_ocr_model = s3_client.get_object(Bucket=s3_bucket_name, Key=key_ocr_model)
    ocr_model_csv = pd.read_csv(BytesIO(response_ocr_model['Body'].read()))
    
    combined_df = pd.merge(image_model_csv, ocr_model_csv, on='image_name', how='outer')
    combined_df["Label"] = combined_df.apply(label_data, axis=1)
    csv_buffer = io.StringIO()
    combined_df.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=s3_bucket_name, Key='combined_dataset.csv', Body=csv_buffer.getvalue(), ContentType="text/csv")
