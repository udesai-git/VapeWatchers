from models.config import AWS_ACCESS_KEY, AWS_SECRET_KEY, aws_session_token
import pandas as pd
import boto3
from io import BytesIO
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

def generate_radar_chart(df, brand_name):
    metrics = {}

    metrics['Youth Appeal Score'] = df['youth_appeal_score'].mean()
    metrics['Readability Youth Score'] = df['readability_youth_score'].mean()
    metrics['Special Characters Youth Score'] = df['special_chars_youth_score'].mean()

    if 'contains_warning' in df.columns:
        metrics['Warning Presence'] = df['contains_warning'].map({'Yes': 1, 'No': 0}).mean() * 100
    else:
        metrics['Warning Presence'] = 0

    # Handle cartoon and face presence 
    if 'Cartoon' in df.columns:
        metrics['Cartoon Presence'] = df['Cartoon'].map({'Yes': 1, 'No': 0}).mean() * 100
    else:
        metrics['Cartoon Presence'] = 0

    if 'Face' in df.columns:
        metrics['Face Presence'] = df['Face'].map({'Yes': 1, 'No': 0}).mean() * 100
    else:
        metrics['Face Presence'] = 0

    labels = list(metrics.keys())
    values = list(metrics.values())

    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(f"Brand Radar Chart: {brand_name}", y=1.1)
    ax.grid(True)

    save_path = f"static/visualization_images/{brand_name}"
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/{brand_name}_radar.png")
    plt.close()

    print(f"Radar chart saved to {save_path}/{brand_name}_radar.png")

 
def create_brand_bar_graph(brand_df, brand_name):
    columns_to_include = [
        'youth_appeal_score', 'readability_youth_score', 'special_chars_youth_score',
        'contains_warning', 'Cartoon', 'Face'
    ]

    highlight_columns = [
        'youth_appeal_score', 'readability_youth_score', 'special_chars_youth_score',
        'contains_warning', 'Cartoon', 'Face'
    ]

    avg_values = {}
    for col in columns_to_include:
        if col in brand_df.columns:
            if col in ['contains_warning', 'Cartoon', 'Face']:
                avg_values[col] = brand_df[col].map({'Yes': 1, 'No': 0}).mean()
            elif pd.api.types.is_numeric_dtype(brand_df[col]):
                avg_values[col] = brand_df[col].mean()

    if not avg_values:
        print(f"No numeric metrics available for brand {brand_name}")
        return

    avg_df = pd.DataFrame({'metric': list(avg_values.keys()), 'value': list(avg_values.values())})

    bar_colors = ['#ff7f0e' if col in highlight_columns else '#1f77b4' for col in avg_df['metric']]

    plt.figure(figsize=(12, 8))
    bars = plt.bar(avg_df['metric'], avg_df['value'], color=bar_colors)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    plt.title(f'Average Youth Appeal Metrics for {brand_name}', fontsize=16)
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Average Value', fontsize=14)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()

    legend_elements = [
        Patch(facecolor='#ff7f0e', label='Youth Appeal Metrics'),
        Patch(facecolor='#1f77b4', label='Other Metrics')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    save_path = f'static/visualization_images/{brand_name}'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/{brand_name}_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Brand bar graph saved to {save_path}/{brand_name}_metrics.png")



def generate_brand_specific_EDA():
    s3_session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        aws_session_token=aws_session_token
    )
    s3_client = s3_session.client('s3')
    s3_bucket_name = 'vapewatchers-2025'
    prefix = 'MarketingImages/'

    # List all folders inside 'MarketingImages/'
    response = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix=prefix, Delimiter='/')
    brand_folders = [content['Prefix'] for content in response.get('CommonPrefixes', [])]

    # Find the latest folder based on object LastModified
    latest_brand = None
    latest_time = None

    for folder in brand_folders:
        folder_objects = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix=folder)
        if 'Contents' in folder_objects:
            first_object = folder_objects['Contents'][0]
            last_modified = first_object['LastModified']
            if latest_time is None or last_modified > latest_time:
                latest_time = last_modified
                latest_brand = folder

    if not latest_brand:
        print("No brands found in the MarketingImages/ folder!")
        return

    brand_name = latest_brand.split('/')[-2]  # Extract brand name
    print(f"📦 Latest brand detected: {brand_name}")

    # Step 4: Load merged CSVs (Image model + OCR analysis)
    combined_data_key = "combined_dataset_new.csv"

    response_image_models = s3_client.get_object(Bucket=s3_bucket_name, Key=combined_data_key)
    combined_data_csv = pd.read_csv(BytesIO(response_image_models['Body'].read()))

    # Step 5: Filter rows belonging to the latest brand
    brand_df = combined_data_csv[combined_data_csv['brand'] == brand_name]

    if brand_df.empty:
        print(f"No data found for brand {brand_name}.")
        return

    # Step 6: Generate Radar Chart
    generate_radar_chart(brand_df, brand_name)

    # Step 7: Generate Brand Metrics Table
    create_brand_bar_graph(brand_df, brand_name)

    print(f"Brand-specific EDA completed for {brand_name}!")