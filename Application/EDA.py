import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import boto3
from io import StringIO

import boto3
import pandas as pd
from io import BytesIO


AWS_ACCESS_KEY="ASIARMWYWT5WXYNUMKHG"
AWS_SECRET_KEY="tUqiukQ0s+o9YsEojE439lqLjKazPFU5nNpMbytT"
aws_session_token="IQoJb3JpZ2luX2VjEK3//////////wEaCXVzLXdlc3QtMiJHMEUCIQDjpsflsXN4BbzYbCa/3IMXBZGBJKKlMcwdTGPgmSA+NgIgX17Xv6CjafvLNSXmEuCIaQapdsCGYcDCTwosCTbbii4qqgIINhABGgwwOTYwMTc0ODk3NzMiDDpPPv8eXe1kFtJcPSqHAr3DNsIY6tHlZl0DOIbfmxUeVNn/nDWJ/yrZVfT2uIoTDW5a3bTQYAKZsqIxYqFiuk5yPfbzN6BvcG9ycTsGOLSMstqNAPRxRnbPm3Q1Qx4Dau9Vt+lenk2zUDEn76RfRQ2w4LUuKD1Mdkr7gh09D9zJuC5Yj5EAIH5bYiDXSryCfKSfREEsna4ai2OICTXQsRpBBIMZ9YmjoWiPmqbNJFeGvM4z/ICdJPvW/ue9rNOOz6cdiRubMZKuNIYbVf3QC66xNN7SWCOgi02+hCmLkhbB9QTeWj1WSH4R1QKv40XaG396SrdpxmBucwirN/Q1sttWEOd8eFB1SrEuTndzLhYKk7lekKcbMOWL+78GOp0BvUebS/IYdGCA7f6Jo8fsxUILk1ZwYghC87SG8pGYrc72mWUCqCO38b4O+DGt5vh3kidV12Amb5iBRivXLNbmfIdI6ECS4b0RlyL1h1flYDaX/XfbFFIY9DkpN7Ign0IszHRF9hKuYfU3MEZ5tvf5z89hjxcXiF2AjATMVFMw1zsUsKZ8Kr9+WwrsiDM/86fF09m16Co5XOQqdfRRCw=="


s3_session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    aws_session_token=aws_session_token
)

s3 = s3_session.client('s3')
csv_obj = s3.get_object(Bucket='vapewatchers-2025', Key='marketing_image_processed.csv')
body = csv_obj['Body'].read()

df = pd.read_csv(BytesIO(body))


# s3_path = f"s3://vape-watchers-2025/marketing_image_processed.csv"
# df = pd.read_csv(s3_path)


# dataset_path = "static/marketing_image_processed.csv"

# Define the static image folder
visualization_folder = "static/visualization_images"

# Delete old images
if os.path.exists(visualization_folder):
    for file in os.listdir(visualization_folder):
        file_path = os.path.join(visualization_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
else:
    os.makedirs(visualization_folder)

# # Load the dataset
# df = pd.read_csv(dataset_path)

# Quick overview
print(df.info())
df.head()

sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Brightness_Level')
plt.title('Distribution of Brightness Levels')
plt.xlabel('Brightness Level')
plt.ylabel('Count')
image_path = os.path.join(visualization_folder, "Distribution_of_Brightness_Levels.png")
plt.savefig(image_path)
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Dominant_COLOR', order=df['Dominant_COLOR'].value_counts().index)
plt.title('Dominant VIBGYOR Color Distribution')
plt.xlabel('Dominant Color')
plt.ylabel('Count')
image_path = os.path.join(visualization_folder, "Dominant_COLOR_Color_Distribution.png")
plt.savefig(image_path)
plt.close()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Cartoon')
plt.title('Presence of Cartoon Elements in Images')
plt.xlabel('Cartoon')
plt.ylabel('Count')
image_path = os.path.join(visualization_folder, "Presence_of_Cartoon_Elements_in_Images.png")
plt.savefig(image_path)
plt.close()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Face')
plt.title('Presence of Human Faces in Images')
plt.xlabel('Face Detected')
plt.ylabel('Count')
image_path = os.path.join(visualization_folder, "Presence_of_Human_Faces_in_Images.png")
plt.savefig(image_path)
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(data=df[df['Vape_Type'].notnull()], x='Vape_Type', order=df['Vape_Type'].value_counts().index)
plt.title('Vape Type Distribution')
plt.xlabel('Vape Type')
plt.ylabel('Count')
plt.xticks(rotation=15)
image_path = os.path.join(visualization_folder, "Vape_Type_Distribution.png")
plt.savefig(image_path)
plt.close()

plt.figure(figsize=(10, 5))
sns.countplot(data=df[df['Age'].notnull()], x='Age',
              order=df['Age'].value_counts().index)
plt.title('Age Distribution in Images')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=45)
image_path = os.path.join(visualization_folder, "Age_Distribution_in_Images.png")
plt.savefig(image_path)
plt.close()

pivot_brightness = pd.crosstab(df['Vape_Type'], df['Brightness_Level'])
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_brightness, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap of Vape Type vs Brightness Level')
plt.xlabel('Brightness Level')
plt.ylabel('Vape Type')
image_path = os.path.join(visualization_folder, "VapeType_vs_BrightnessLevel.png")
plt.savefig(image_path)
plt.close()

cartoon_color_ct = pd.crosstab(df['Dominant_COLOR'], df['Cartoon'])
cartoon_color_ct.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Stacked Bar Chart: Cartoon Presence by Dominant Color')
plt.xlabel('Dominant VIBGYOR Color')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Cartoon')
plt.tight_layout()
image_path = os.path.join(visualization_folder, "Stacked Bar Chart.png")
plt.savefig(image_path)
plt.close()

plt.figure(figsize=(8, 6))
sns.violinplot(data=df, x='Face', y='Brightness_Level', palette='muted')
plt.title('Brightness Level Distribution by Face Presence')
plt.xlabel('Face Detected')
plt.ylabel('Brightness Level')
image_path = os.path.join(visualization_folder, "BrightnessLevelDistribution.png")
plt.savefig(image_path)
plt.close()


# # Create a cross-tabulation of Cartoon vs Company
# cartoon_company_ct = pd.crosstab(df['Company'], df['Cartoon'])

# # Plot the stacked bar chart
# cartoon_company_ct.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Set2')
# plt.title('Stacked Bar Chart: Cartoon vs Non-Cartoon Images by Company')
# plt.xlabel('Company')
# plt.ylabel('Number of Images')
# plt.xticks(rotation=45)
# plt.legend(title='Cartoon')
# plt.tight_layout()
# image_path = os.path.join(visualization_folder, "Cartoon_vs_NonCartoon.png")
# plt.savefig(image_path)
# plt.close()