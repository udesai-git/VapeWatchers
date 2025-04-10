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

# Load models
AGE_LIST = ['(0-2)', '(4-6)', '(8-13)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
prototxt_path = "/Users/mayankgrover/Documents/DAEN_698/Vape_Regulation_Project/models/deploy_age.prototxt"
model_path = "/Users/mayankgrover/Documents/DAEN_698/Vape_Regulation_Project/models/age_net.caffemodel"
age_model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

cartoon_model = YOLO('/Users/mayankgrover/Documents/DAEN_698/Vape_Regulation_Project/models/yolov8_trained.pt')
vape_model = YOLO('/Users/mayankgrover/Documents/DAEN_698/Vape_Regulation_Project/models/best.pt')


print("🔄 Starting image processing...")
data = []


AWS_ACCESS_KEY="ASIARMWYWT5WXMIH756U"
AWS_SECRET_KEY="HxmXPjM255nectVrgPIusFgiJgOHCYeC1YFnJrfP"
aws_session_token="IQoJb3JpZ2luX2VjEPH//////////wEaCXVzLXdlc3QtMiJHMEUCIF0oIvMP4z8Tk7CVEEZ95fJVHpo3Ph5xWtERLDDBN13MAiEAy29ggiROBkDp7gs6VAz3HiTcqT3Uq6cHvLGgp+nw/E4qqgIIahABGgwwOTYwMTc0ODk3NzMiDBL4sfBBqi9YNSyHVSqHAg+d/H2bLQJd6XfuIC59iSDbcQIAViVnG8c5PgNqy2iXY/KF2ec8UoJMcb3FIxMVrKWZV1hhf3Cekn5d2lA/simQkRYoBvkQokhILQRSPWVJ7xkUJGsISqusTrd63cwcRVLocpt2dHL7sEYvQ3vFtfsWlqehyI6InimfQ0sSGHStqczJeaU3ZHLcG1uOa70xdLZavWYiHbOCK9KqqA/fRczBPc6ktGIng9j4cckGxfmwLG1baiC5FIJiVpGbGOMVFa5IrCOEZsEXvNcE56OPAlTgR3x5SCJGwknVwErkQSnHPd7yFybBMXAmQMD1dA34ZSTNpIyWG9ZIUpN+qh4C9drRV4OIf5CBMLja0b8GOp0B8qr4Gn9wNJQiBqA1OWjxq2Hz77yit0M5yBkI2HKmM72/JpnIlJR+i9QQ31F13jNxMiAL0AXCBuoTyj8QiC+JEa/1/ucaQTiIcFNsn2npOJQ1q/G5yHMQMAau2oeEzs1CA3/s1J4XzY6r9CgRZBV8X103yvY0vh7SIj1GE7ku57zYsByVBTFGD/qvm+gj8W5ZKEHJSsV0dqoiDds7AA=="


s3_session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    aws_session_token=aws_session_token
)

s3_client = s3_session.client('s3')

bucket_name = 'vapewatchers-2025'
prefix = 'MarketingImages/'


# # Named tuple for brightness levels
# BLevel = namedtuple("BLevel", ['brange', 'bval'])
# _blevels = [
#     BLevel(brange=range(0, 24), bval=0),
#     BLevel(brange=range(23, 47), bval=1),
#     BLevel(brange=range(46, 70), bval=2),
#     BLevel(brange=range(69, 93), bval=3),
#     BLevel(brange=range(92, 116), bval=4),
#     BLevel(brange=range(115, 140), bval=5),
#     BLevel(brange=range(139, 163), bval=6),
#     BLevel(brange=range(162, 186), bval=7),
#     BLevel(brange=range(185, 209), bval=8),
#     BLevel(brange=range(208, 232), bval=9),
#     BLevel(brange=range(231, 256), bval=10),
# ]

from scipy.spatial import distance

# COLORS = {
#     # VIBGYOR base
#     "Violet": (148, 0, 211),
#     "Indigo": (75, 0, 130),
#     "Blue": (0, 0, 255),
#     "Green": (0, 255, 0),
#     "Yellow": (255, 255, 0),
#     "Orange": (255, 165, 0),
#     "Red": (255, 0, 0),

#     # Bright variants
#     "Bright Blue": (135, 206, 250),
#     "Bright Green": (144, 238, 144),
#     "Bright Yellow": (255, 255, 102),
#     "Bright Orange": (255, 200, 0),
#     "Bright Red": (255, 99, 71),
#     "Bright Pink": (255, 105, 180),

#     # Dull/Muted variants
#     "Dull Blue": (100, 100, 150),
#     "Dull Green": (85, 107, 47),
#     "Dull Yellow": (204, 204, 102),
#     "Dull Orange": (210, 105, 30),
#     "Dull Red": (139, 0, 0),
#     "Dull Pink": (219, 112, 147),
    
#     "Turquoise": (64, 224, 208),
#     "Cyan": (0, 255, 255),
#     "Aqua": (0, 255, 255),
#     "Teal": (0, 128, 128),
#     "Magenta": (255, 0, 255),
#     "Pink": (255, 192, 203),
#     "Hot Pink": (255, 105, 180),
#     "Fuchsia": (255, 20, 147),
#     "Maroon": (128, 0, 0),
#     "Burgundy": (128, 0, 32),
#     "Peach": (255, 218, 185),
#     "Beige": (245, 245, 220),
#     "Coral": (255, 127, 80),
#     "Salmon": (250, 128, 114),

#     "White": (255, 255, 255),
#     "Black": (0, 0, 0),
#     "Gray": (128, 128, 128),
#     "Light Gray": (211, 211, 211),
#     "Dark Gray": (64, 64, 64),
#     "Brown": (139, 69, 19),
#     "Tan": (210, 180, 140),
#     "Olive": (128, 128, 0),
#     "Khaki": (195, 176, 145),
#     "Neon Green": (57, 255, 20),
#     "Neon Pink": (255, 20, 147),
#     "Neon Blue": (77, 77, 255),
#     "Neon Yellow": (204, 255, 0),
#     "Neon Orange": (255, 153, 51),


# }


# def closest_vibgyor_color(color):
#     """Finds the closest VIBGYOR color based on Euclidean distance."""
#     return min(COLORS, key=lambda vib_color: distance.euclidean(color, COLORS[vib_color]))

# def extract_dominant_vibgyor(image_path, top_n=5):
#     """Extracts the most common color and maps it to VIBGYOR."""
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     pixels = image.reshape(-1, 3)
#     color_counts = Counter(map(tuple, pixels))

#     # Get top N colors
#     most_common_colors = color_counts.most_common(top_n)

#     # Map extracted colors to VIBGYOR
#     mapped_colors = [closest_vibgyor_color(color[0]) for color in most_common_colors]

#     # Return the most frequently occurring VIBGYOR color
#     return max(set(mapped_colors), key=mapped_colors.count)


# def detect_level(h_val):
#     h_val = int(h_val)
#     for blevel in _blevels:
#         if h_val in blevel.brange:
#             return blevel.bval
#     raise ValueError("Brightness Level Out of Range")

# def get_img_avg_brightness(image_path):
#     img = cv2.imread(image_path)
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     _, _, v = cv2.split(hsv)
#     return int(np.average(v.flatten()))

# def get_company_name(image_path):
#     path_parts = image_path.split('/')
#     if "MarketingImages" in path_parts:
#         idx = path_parts.index("MarketingImages")
#         if idx + 1 < len(path_parts):
#             return path_parts[idx + 1]
#     return "Unknown"


# def detect_age_with_mtcnn(img):
#     detector = MTCNN()
#     faces = detector.detect_faces(img)
#     results = []
#     for face in faces:
#         x, y, w, h = face['box']
#         face_img = img[y:y + h, x:x + w].copy()
#         blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), (78.426, 87.769, 114.896), swapRB=False)
#         age_model.setInput(blob)
#         age_preds = age_model.forward()
#         age = AGE_LIST[age_preds[0].argmax()]
#         results.append((x, y, w, h, age))
#     return results

# def detect_cartoon_objects(img):
#     results = cartoon_model.predict(img)
#     return any(cartoon_model.names[int(box[5])] == 'cartoon' for r in results for box in r.boxes.data)

# def detect_vape_type(img):
#     results = vape_model.predict(img)
#     best_vape = None
#     highest_conf = 0
#     for r in results:
#         for box in r.boxes.data:
#             class_id = int(box[5])
#             confidence = float(box[4])
#             class_name = vape_model.names[class_id]
#             if confidence > highest_conf:
#                 highest_conf = confidence
#                 best_vape = class_name
#     return best_vape if best_vape else "None"

# def label_data(row):
#     if row['Face'] == 1 and row['Age'] in ['(0-2)', '(4-6)', '(8-13)', '(15-20)', '(25-32)']:
#         return 1 
#     elif row['Cartoon'] == 1:
#         return 1
#     elif row['Brightness_Level'] > 5:
#         return 1
#     elif row['Dominant_VIBGYOR'] in ['Red', 'Orange', 'Yellow']:
#         return 1
#     return 0

# def analyze_and_save_results(base_folder):
#     data = []
#     for company_folder in os.listdir(base_folder):
#         company_path = os.path.join(base_folder, company_folder)
#         if not os.path.isdir(company_path):
#             continue
#         for image_name in os.listdir(company_path):
#             image_path = os.path.join(company_path, image_name)
#             if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 continue
#             print(f"Processing: {image_path}")
#             img = cv2.imread(image_path)
#             if img is None:
#                 print(f"Error: Could not load {image_path}")
#                 continue
#             company_name = get_company_name(image_path)
#             # drive_link = get_drive_link(image_path)
#             age_results = detect_age_with_mtcnn(img)
#             face_detected = len(age_results) > 0
#             detected_age = age_results[0][4] if face_detected else "N/A"
#             cartoon_detected = detect_cartoon_objects(img)
#             vape_type = detect_vape_type(img)
#             brightness_value = get_img_avg_brightness(image_path)
#             brightness_level = detect_level(brightness_value)
#             colors = extract_dominant_vibgyor(image_path, top_n=3)
#             data.append({
#                 "image_name": image_name,
#                 "Company": company_name,
#                 "Face": "Yes" if face_detected else "No",
#                 "Age": detected_age,
#                 "Cartoon": "Yes" if cartoon_detected else "No",
#                 "Vape_Type": vape_type,
#                 "Brightness_Level": brightness_level,
#                 "Dominant_VIBGYOR": extract_dominant_vibgyor(image_path)
#             })
    
#     df = pd.DataFrame(data)
#     df['Label'] = df.apply(label_data, axis=1)
#     output_csv = "/Users/mayankgrover/Documents/DAEN_698/Vape_Regulation_Project/static/marketing_image_processed.csv"
#     df.to_csv(output_csv, index=False)
#     print(f"✅ CSV successfully generated at: {output_csv}")[]


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


def extract_dominant_vibgyor(image_input, top_n=5):
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
    if row['Face'] == 1 and row['Age'] in ['(0-2)', '(4-6)', '(8-13)', '(15-20)', '(25-32)']:
        return 1
    elif row['Cartoon'] == 1:
        return 1
    elif row['Brightness_Level'] > 5:
        return 1
    elif row['Dominant_VIBGYOR'] in ['Red', 'Orange', 'Yellow']:
        return 1
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
                colors = extract_dominant_vibgyor(img, top_n=3)

                data.append({
                    "image_name": image_name,
                    "Brand": brand_name,
                    "Face": "Yes" if face_detected else "No",
                    "Age": detected_age,
                    "Cartoon": "Yes" if cartoon_detected else "No",
                    "Vape_Type": vape_type,
                    "Brightness_Level": brightness_level,
                    "Dominant_VIBGYOR": colors
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
    df["Label"] = df.apply(label_data, axis=1)

    output_csv_key = "marketing_image_processed.csv"
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=s3_bucket_name, Key=output_csv_key, Body=csv_buffer.getvalue(), ContentType="text/csv")

    print(f"✅ CSV successfully saved to s3://{s3_bucket_name}/{output_csv_key}")
