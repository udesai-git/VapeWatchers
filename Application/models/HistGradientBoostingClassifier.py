import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from models.config import AWS_ACCESS_KEY,AWS_SECRET_KEY,aws_session_token
import boto3
import io
import os
import seaborn as sns
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib


def data_pre_processing(data, generate_labels):
    # Step 1: Select columns
    if generate_labels:
        data = data[['youth_appeal_score', 'readability_youth_score', 'special_chars_youth_score',
                     'contains_warning', 'Face', 'Cartoon', 'Vape_Type', 'Dominant_COLOR', 'Label']].copy()
    else:
        data = data[['youth_appeal_score', 'readability_youth_score', 'special_chars_youth_score',
                     'contains_warning', 'Face', 'Cartoon', 'Vape_Type', 'Dominant_COLOR']].copy()

    # Step 2: Safely map contains_warning
    data['contains_warning'] = data['contains_warning'].map({'Yes': 1, 'No': 0})

    # VERY IMPORTANT: Fix Vape_Type BEFORE get_dummies
    data['Vape_Type'] = data['Vape_Type'].fillna('unknown')
    data['Vape_Type'] = data['Vape_Type'].replace('None', 'unknown')

    # Step 3: Now dummy encoding
    data_en = pd.get_dummies(data, columns=['Face', 'Cartoon', 'Vape_Type', 'Dominant_COLOR', 'contains_warning'])
    # Step 4: Expected features
    expected_features = [
    'youth_appeal_score', 'readability_youth_score', 'special_chars_youth_score',
    'Face_No', 'Face_Yes',
    'Cartoon_No', 'Cartoon_Yes',
    'Vape_Type_None', 'Vape_Type_electronic-cigarette', 'Vape_Type_mod', 
    'Vape_Type_pod', 'Vape_Type_unknown', 'Vape_Type_vape-bP5a',
    'Dominant_COLOR_Beige', 'Dominant_COLOR_Black', 'Dominant_COLOR_Blue',  # <-- comma
    'Dominant_COLOR_Bright Blue', 'Dominant_COLOR_Bright Green', 'Dominant_COLOR_Bright Orange', 
    'Dominant_COLOR_Bright Pink', 'Dominant_COLOR_Bright Red', 'Dominant_COLOR_Bright Yellow',
    'Dominant_COLOR_Brown', 'Dominant_COLOR_Burgundy', 'Dominant_COLOR_Coral', 'Dominant_COLOR_Cyan',
    'Dominant_COLOR_Dark Gray', 'Dominant_COLOR_Dull Blue',  # <-- comma
    'Dominant_COLOR_Dull Green', 'Dominant_COLOR_Dull Orange', 'Dominant_COLOR_Dull Pink',
    'Dominant_COLOR_Dull Red', 'Dominant_COLOR_Dull Yellow', 'Dominant_COLOR_Fuchsia',
    'Dominant_COLOR_Gray', 'Dominant_COLOR_Indigo', 'Dominant_COLOR_Khaki',
    'Dominant_COLOR_Light Gray', 'Dominant_COLOR_Maroon',  # <-- comma
    'Dominant_COLOR_Neon Blue', 'Dominant_COLOR_Neon Green', 'Dominant_COLOR_Neon Orange',
    'Dominant_COLOR_Olive', 'Dominant_COLOR_Orange', 'Dominant_COLOR_Peach',
    'Dominant_COLOR_Pink', 'Dominant_COLOR_Red', 'Dominant_COLOR_Salmon',
    'Dominant_COLOR_Tan', 'Dominant_COLOR_Teal', 'Dominant_COLOR_Turquoise',
    'Dominant_COLOR_Violet', 'Dominant_COLOR_White',  # <-- comma
    'contains_warning_0', 'contains_warning_1'
    ]


    if generate_labels:
        expected_features.append('Label')

    # Step 5: Add missing columns with 0
    for col in expected_features:
        if col not in data_en.columns:
            data_en[col] = 0

    # Step 6: Select only expected columns in correct order
    data_en = data_en[expected_features]

    return data_en



def train_model(processed_data):
    print('inside train method....')
    y=processed_data['Label']
    X=processed_data.drop('Label',axis=1)

    correlations = processed_data.corr(numeric_only=True)['Label'].sort_values(ascending=False)
    print(correlations)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, X_train,X_test,y_train, y_test

def test_model(model,X_train,X_test,y_train, y_test):
    
    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Optional: More detailed evaluation
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print("Train Accuracy:", train_acc)
    print("Test Accuracy :", test_acc)
    
    return test_acc

def plot_feature_importance_seaborn(model, X_test, y_test, features, save_path="static/visualization_images/Permutation_Feature_Importance.png"):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    importances = result.importances_mean

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, y="Feature", x="Importance", palette="viridis")
    plt.title("Permutation Feature Importance")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def main():
    
    final_dataset_key = 'final_dataset.csv'
    s3_bucket_name = 'vapewatchers-2025'

    s3_session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        aws_session_token=aws_session_token
    )

    s3_client = s3_session.client('s3')
        
    response_final_dataset = s3_client.get_object(Bucket=s3_bucket_name, Key=final_dataset_key)
    data= pd.read_csv(io.BytesIO(response_final_dataset['Body'].read()))
    
    data_pre_processed = data_pre_processing(data)
    model, X_train,X_test,y_train, y_test = train_model(data_pre_processed)
    
    test_model(model,X_train=X_train, X_test=X_test, y_train=y_train,y_test=y_test)
    plot_feature_importance_seaborn(model, X_test, y_test, X_test.columns)
    
    model_path = "models/trained_model.pkl"
    joblib.dump(model, model_path)
    