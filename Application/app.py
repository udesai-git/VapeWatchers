from flask import Flask, render_template, jsonify, send_file
import os
import subprocess
import pandas as pd
from models.image_processing import analyze_and_save_results
from flask import send_from_directory
import boto3
from io import BytesIO

app = Flask(__name__)


AWS_ACCESS_KEY="ASIARMWYWT5WXMIH756U"
AWS_SECRET_KEY="HxmXPjM255nectVrgPIusFgiJgOHCYeC1YFnJrfP"
aws_session_token="IQoJb3JpZ2luX2VjEPH//////////wEaCXVzLXdlc3QtMiJHMEUCIF0oIvMP4z8Tk7CVEEZ95fJVHpo3Ph5xWtERLDDBN13MAiEAy29ggiROBkDp7gs6VAz3HiTcqT3Uq6cHvLGgp+nw/E4qqgIIahABGgwwOTYwMTc0ODk3NzMiDBL4sfBBqi9YNSyHVSqHAg+d/H2bLQJd6XfuIC59iSDbcQIAViVnG8c5PgNqy2iXY/KF2ec8UoJMcb3FIxMVrKWZV1hhf3Cekn5d2lA/simQkRYoBvkQokhILQRSPWVJ7xkUJGsISqusTrd63cwcRVLocpt2dHL7sEYvQ3vFtfsWlqehyI6InimfQ0sSGHStqczJeaU3ZHLcG1uOa70xdLZavWYiHbOCK9KqqA/fRczBPc6ktGIng9j4cckGxfmwLG1baiC5FIJiVpGbGOMVFa5IrCOEZsEXvNcE56OPAlTgR3x5SCJGwknVwErkQSnHPd7yFybBMXAmQMD1dA34ZSTNpIyWG9ZIUpN+qh4C9drRV4OIf5CBMLja0b8GOp0B8qr4Gn9wNJQiBqA1OWjxq2Hz77yit0M5yBkI2HKmM72/JpnIlJR+i9QQ31F13jNxMiAL0AXCBuoTyj8QiC+JEa/1/ucaQTiIcFNsn2npOJQ1q/G5yHMQMAau2oeEzs1CA3/s1J4XzY6r9CgRZBV8X103yvY0vh7SIj1GE7ku57zYsByVBTFGD/qvm+gj8W5ZKEHJSsV0dqoiDds7AA=="


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
        print("📝 Starting CSV generation...")  # Debugging line to check if the route is being hit
        analyze_and_save_results(s3_bucket_name='vapewatchers-2025', prefix='MarketingImages/')  # or whatever your prefix is
        message = "✅ CSV successfully generated and saved to S3!"
        message_class = "success"
    except Exception as e:
        print(f"❌ Error generating CSV: {str(e)}")  # Log error to console
        message = f"❌ Error generating CSV: {str(e)}"
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
