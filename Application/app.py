from flask import Flask, render_template, jsonify, send_file
import os
import subprocess
import pandas as pd
from models.image_processing import analyze_and_save_results
from flask import send_from_directory
import boto3
from io import BytesIO

app = Flask(__name__)


AWS_ACCESS_KEY="ASIARMWYWT5W52CZXXBI"
AWS_SECRET_KEY="c9orzaCeF8NsHy+pITmG3gYk3knbRELOTTtT7+4X"
aws_session_token="IQoJb3JpZ2luX2VjEKX//////////wEaCXVzLXdlc3QtMiJHMEUCIQCj0tLOhKjnDfCY5Ml6tLpDdDf1Es92KFu/toW1dJ1VnQIgY/tru9CTU9edVaqZzUOR9s5o3piJ3MlZCwekL8MvqX4qqgIILhABGgwwOTYwMTc0ODk3NzMiDB+v1RCHfTDeHYzJOyqHAqVuxYR4tNpwlKeEOT8aKNGQCGqPudL0QLfJtYrZftffi7EtYGVhtU2UCUaK1niZFY+ckpkCDVZsXyPtLxGdVZofkbyV12rEHwF/rCRmh8IKGCxS+PNnwj4bPeJbIsHrrSVyYKRV3NhoqDDESLMOdczgA3TcknuAmgxR6tUBt7e/4hEXPl5LatB844kd8g7i9fmMS4gwOE+6J918OhiWvuhzL7IlfQ3vRgcEGCoQL46PkuqGjfM0zI0EPVMtWDB1bD377L4kSoQ5KjrjREF5ZAaQD+Uq2pw3/lb3F5ZnPmYVELqnUd1h1Ytki4JXgodah7mprUFezUi/gte3klabqRcz1xdg6E9oMJ+l+b8GOp0Bugt5jI36GV6EC/zRUR1qOXK/a4dZns++0e81jPhn9kCY6Hk2flAlpdZDyKtQBrWX7Uzc8f0z96PtJF8BoEJjESlM48POZHofuAY+ap5ib4XnqN2dDO3IrTLjNQWLyr0+WLVA9oc6UNA9zRARKTfqQNpefdyx5+G99levH8OFp/GinVwV/HR7A+SWiVS2B+UGkUvbXhT5cFUu4LP0rw=="

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
