import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import boto3
from io import StringIO

import boto3
import pandas as pd
from io import BytesIO


AWS_ACCESS_KEY="ASIARMWYWT5W4G62QMRW"
AWS_SECRET_KEY="8w5emt1huSBPI0jZvKkrSeJNw7HAt6IUqpa6Y9SA"
aws_session_token="IQoJb3JpZ2luX2VjEKX//////////wEaCXVzLXdlc3QtMiJGMEQCIH6fYeZ14g7dwA667SRa5BjnQCkV9HcgDKlxKqWhyviWAiATwEUcW8i/BSZq5wtKzRXnjykmMmgtAR8n3YYvt9L4dCqqAgg+EAEaDDA5NjAxNzQ4OTc3MyIM0aYJScB/NK0JTq90KocCsKBrGvhph8iiTtC/0q+NVdcWEjEF+lj9dm/3dy4oAggHLmFq27znUgynfdPn+jsC3ciqwcDJ8r7M0lQMPMurycSJFTGp5tVf2Z64ymopLazKjSEQv4G47pMq2YPWQg5fa5QG259RdYfG6GrFgaQSXyzSo1xz5IdbSvHI0Ru18iub+1tR6VRWVnxk/nHA+iUEi7/XTLCcnknPKqbXKsj3hwQ8vS9/1WJtYlOQuRNEXD6bUYDl+W99IVwJCXIbtPuxC+k6kyTN2bU2S/DeiiHHh/vpH+zWT/vwwd7bPDSdMFL9chmxcbuo/jc0SiSZCn9Zgb9sHB0HfHXGH2F8NTSbxqAT9t5HY24wh86xwAY6ngGATfjHS3KeYHORnf7Soh+bjrIUaCVxRyVznOsq6jzNZqwTxmwP4E9/zvnfzxDdrC0bOll8kHeCqzDpaPIKE0hW9LRmCFiKXyDUDLEjkkson9zN9exp2y6e3jBF4JST3PLijtTVC/NqtBuVvqyqyxsU4LvFcpWOINdgomPBVcZEnyBNTtO/EUKZIPYbZ8XVRMCHxJgjcdpCzh/oDC7m/w=="


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