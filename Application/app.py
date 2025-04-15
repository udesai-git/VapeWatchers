from flask import Flask, render_template, jsonify, send_file
import os
import subprocess
import pandas as pd
from models.image_processing import analyze_and_save_results
from flask import send_from_directory
import boto3
from io import BytesIO

app = Flask(__name__)


AWS_ACCESS_KEY="ASIARMWYWT5WXYNUMKHG"
AWS_SECRET_KEY="tUqiukQ0s+o9YsEojE439lqLjKazPFU5nNpMbytT"
aws_session_token="IQoJb3JpZ2luX2VjEK3//////////wEaCXVzLXdlc3QtMiJHMEUCIQDjpsflsXN4BbzYbCa/3IMXBZGBJKKlMcwdTGPgmSA+NgIgX17Xv6CjafvLNSXmEuCIaQapdsCGYcDCTwosCTbbii4qqgIINhABGgwwOTYwMTc0ODk3NzMiDDpPPv8eXe1kFtJcPSqHAr3DNsIY6tHlZl0DOIbfmxUeVNn/nDWJ/yrZVfT2uIoTDW5a3bTQYAKZsqIxYqFiuk5yPfbzN6BvcG9ycTsGOLSMstqNAPRxRnbPm3Q1Qx4Dau9Vt+lenk2zUDEn76RfRQ2w4LUuKD1Mdkr7gh09D9zJuC5Yj5EAIH5bYiDXSryCfKSfREEsna4ai2OICTXQsRpBBIMZ9YmjoWiPmqbNJFeGvM4z/ICdJPvW/ue9rNOOz6cdiRubMZKuNIYbVf3QC66xNN7SWCOgi02+hCmLkhbB9QTeWj1WSH4R1QKv40XaG396SrdpxmBucwirN/Q1sttWEOd8eFB1SrEuTndzLhYKk7lekKcbMOWL+78GOp0BvUebS/IYdGCA7f6Jo8fsxUILk1ZwYghC87SG8pGYrc72mWUCqCO38b4O+DGt5vh3kidV12Amb5iBRivXLNbmfIdI6ECS4b0RlyL1h1flYDaX/XfbFFIY9DkpN7Ign0IszHRF9hKuYfU3MEZ5tvf5z89hjxcXiF2AjATMVFMw1zsUsKZ8Kr9+WwrsiDM/86fF09m16Co5XOQqdfRRCw=="

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
        analyze_and_save_results(s3_bucket_name='vapewatchers-2025', prefix='MarketingImages/')
        
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
    subprocess.run(["python", "EDA.py"])
    
@app.before_request
def before_request():
    """Runs EDA.py before serving any page to ensure fresh data."""
    run_eda()

@app.route("/dashboard")
def dashboard():
    image_folder = "static/visualization_images"
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    return render_template("dashboard.html", images=images)



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



if __name__ == "__main__":
    app.run(debug=True)
