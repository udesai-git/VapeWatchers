import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import boto3
from io import StringIO

import boto3
import pandas as pd
from io import BytesIO


AWS_ACCESS_KEY="ASIARMWYWT5W7OHA4IMC"
AWS_SECRET_KEY="SJptkQ/vYrxOVVBFaKjLih6H4pizzFvLtuTYoe8D"
aws_session_token="IQoJb3JpZ2luX2VjEM7//////////wEaCXVzLXdlc3QtMiJGMEQCIGtaxlTQa4NiddB15wrpK9Wq5xEpe1PBM4UeGPVj1tgKAiAYhct6vuPeOBZDKvVtrbISh26GQmMUzRZiwdiKHT3PfyqqAghnEAEaDDA5NjAxNzQ4OTc3MyIMVi5sybM8BJ6I3pDtKocCz63qdxrdP49fnXoBm30+prJmmA8qLJF3nj73ZSNdkkpTnSt+czJ0OUby3kKmLQFc7a1r0VBjSiYFjAIcwuCe7WQL70DE751oUWmxdQHwDt718QkUQC17PRZEmYQJxyFw3JcqIiTCazzitUf0qUTLdaOnVSpc36dLPDpNsotFKjqffCgoxmO7hRT2FF8R5TJu49DH32oMJcFtrH+JoOQm2BTyF8t0O4VnxSOWccMXjKSjuFOP3cXXtV3KsZ02ta6m1w2nbAXhsAHP1YRRUseXetDtiG8pjtIimakss0kM2tmOvSw1awpZLme2A83d5TUWbhKcC7d17/n/THMZOmL1ggGO9zZMVywwk8S6wAY6ngEo6IElf4fgUmj1mQoxJ/EvIqa9tuGTFZoR7g/cB2tnapk4jWf1qiYI5mIYX/xr8IbSEdBbgpibEw4V9sJ36OaQsI1E4TuL68E/HN/LgZlHGpHBMyc8nObaEIYfynGmkEauIYxPGE25o6v6Zbm85R1CDlcB/66xFTUvqsVpPgPqcmvL1emZ03nsz9LGwu7nljvhLWZ+uzp1qHHEZka5og=="


s3_session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    aws_session_token=aws_session_token
)

s3 = s3_session.client('s3')
csv_obj = s3.get_object(Bucket='vapewatchers-2025', Key='marketing_image_processed.csv')
body = csv_obj['Body'].read()

df = pd.read_csv(BytesIO(body))

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
plt.title('Dominant Color Distribution')
plt.xlabel('Dominant Color')
plt.ylabel('Count')
plt.xticks(rotation=45)
image_path = os.path.join(visualization_folder, "Dominant_COLOR_Distribution.png")
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