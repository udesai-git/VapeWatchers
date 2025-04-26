from flask import Flask, render_template, jsonify, send_file, request, send_from_directory
import os
import subprocess
import pandas as pd
from models.image_processing import analyze_and_save_results
from models.HistGradientBoostingClassifier import main as classifier_main
from flask import request, render_template
from models.capstone_EDA_csv_script import main as EDA_main
import subprocess

import boto3
import subprocess
import uuid
from io import BytesIO

app = Flask(__name__)


AWS_ACCESS_KEY="ASIARMWYWT5W4G62QMRW"
AWS_SECRET_KEY="8w5emt1huSBPI0jZvKkrSeJNw7HAt6IUqpa6Y9SA"
aws_session_token="IQoJb3JpZ2luX2VjEKX//////////wEaCXVzLXdlc3QtMiJGMEQCIH6fYeZ14g7dwA667SRa5BjnQCkV9HcgDKlxKqWhyviWAiATwEUcW8i/BSZq5wtKzRXnjykmMmgtAR8n3YYvt9L4dCqqAgg+EAEaDDA5NjAxNzQ4OTc3MyIM0aYJScB/NK0JTq90KocCsKBrGvhph8iiTtC/0q+NVdcWEjEF+lj9dm/3dy4oAggHLmFq27znUgynfdPn+jsC3ciqwcDJ8r7M0lQMPMurycSJFTGp5tVf2Z64ymopLazKjSEQv4G47pMq2YPWQg5fa5QG259RdYfG6GrFgaQSXyzSo1xz5IdbSvHI0Ru18iub+1tR6VRWVnxk/nHA+iUEi7/XTLCcnknPKqbXKsj3hwQ8vS9/1WJtYlOQuRNEXD6bUYDl+W99IVwJCXIbtPuxC+k6kyTN2bU2S/DeiiHHh/vpH+zWT/vwwd7bPDSdMFL9chmxcbuo/jc0SiSZCn9Zgb9sHB0HfHXGH2F8NTSbxqAT9t5HY24wh86xwAY6ngGATfjHS3KeYHORnf7Soh+bjrIUaCVxRyVznOsq6jzNZqwTxmwP4E9/zvnfzxDdrC0bOll8kHeCqzDpaPIKE0hW9LRmCFiKXyDUDLEjkkson9zN9exp2y6e3jBF4JST3PLijtTVC/NqtBuVvqyqyxsU4LvFcpWOINdgomPBVcZEnyBNTtO/EUKZIPYbZ8XVRMCHxJgjcdpCzh/oDC7m/w=="

s3_session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    aws_session_token=aws_session_token
)
print(os.getcwd())

@app.route('/')
def home():
    return render_template("index.html")


@app.route("/generate-csv", methods=["GET"])
def generate_csv():
    try:
        print("📝 Starting CSV generation...")
        analyze_and_save_results(s3_bucket_name='vapewatchers-2025', prefix='MarketingImagesTest/')
        run_eda()
        
        print('Calling mthod to train the model again')
        classifier_main()
        message = "CSV successfully generated and saved to S3!"
        message_class = "success"
    except Exception as e:
        print(f"Error generating CSV: {str(e)}")  # Log error to console
        message = f"Error generating CSV: {str(e)}"
        message_class = "error"

    print(f"Message: {message}")  # Debugging message to print the result
    return render_template("index.html", message=message, message_class=message_class)



@app.route('/visualization_images/<path:filename>')
def serve_image(filename):
    return send_from_directory('static/visualization_images', filename)


def run_eda():
    # Run the EDA script
    EDA_main(AWS_ACCESS_KEY,AWS_SECRET_KEY,aws_session_token)

@app.route("/dashboard")
def dashboard():
    image_root = "static/visualization_images"
    selected_brand = request.args.get("brand")

    # List brand folders (exclude 'nan')
    brand_folders = [
        folder for folder in os.listdir(image_root)
        if os.path.isdir(os.path.join(image_root, folder)) and folder.lower() != "nan"
    ]

    images = []

    if selected_brand and selected_brand in brand_folders:
        # Show images from the selected brand's subfolder
        brand_path = os.path.join(image_root, selected_brand)
        images = [
            os.path.join("visualization_images", selected_brand, img)
            for img in os.listdir(brand_path)
            if img.endswith((".png", ".jpg", ".jpeg"))
        ]
    else:
        # Initial load: only show images directly in static/visualization_images/
        images = [
            os.path.join("visualization_images", img)
            for img in os.listdir(image_root)
            if os.path.isfile(os.path.join(image_root, img)) and img.endswith((".png", ".jpg", ".jpeg"))
        ]

    return render_template(
        "dashboard.html",
        images=images,
        brands=brand_folders,
        selected_brand=selected_brand
    )



@app.route('/download')
def download_csv():
    # S3 configuration
    s3_bucket_name = 'vapewatchers-2025'
    csv_key = 'marketing_image_processed.csv'  # Update this with the correct key/path if necessary

    # Create a session and S3 client
    s3 = s3_session.client('s3')

    try:
        # Fetch the CSV file from S3
        csv_obj = s3.get_object(Bucket=s3_bucket_name, Key=csv_key)
        body = csv_obj['Body'].read()

        # Use BytesIO to create a file-like object from the CSV data
        file_like_object = BytesIO(body)

        # Return the CSV file as an attachment for download
        return send_file(file_like_object, as_attachment=True, download_name='marketing_image_processed.csv', mimetype='text/csv')

    except Exception as e:
        return f"Error: {str(e)}", 500


@app.route('/scrape-data', methods=['POST'])
def scrape_data():
    website_url = request.form.get('url_input')

    try:
        subprocess.run([
            "python3", "models/HybridScraperAWS.py",
            website_url,
            AWS_ACCESS_KEY,
            AWS_SECRET_KEY,
            aws_session_token,
            "50"
        ], check=True)

        return render_template("dashboard.html", message="✅ Scraping complete!")
    
    except subprocess.CalledProcessError as e:
        return render_template("dashboard.html", message="❌ Scraping failed.")


# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    run_eda()
    app.run(host='0.0.0.0', port=80)
