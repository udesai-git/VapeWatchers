import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from models.image_processing import AWS_ACCESS_KEY,AWS_SECRET_KEY,aws_session_token
import boto3
import io
import os
import seaborn as sns
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib



def data_pre_processing(data):
    data=data[['youth_appeal_score','readability_youth_score','special_chars_youth_score','contains_warning','Face','Cartoon','Vape_Type','Dominant_COLOR','Label']]
    data['contains_warning']=data['contains_warning'].map({'Yes': 1, 'No': 0})
    data_en=pd.get_dummies(data,columns=['Face','Cartoon','Vape_Type','Dominant_COLOR','contains_warning'])
    print(data_en.dtypes)
    nan_counts = data_en.isna().sum()
    print(nan_counts[nan_counts > 0])
    return data_en


def train_model(processed_data):
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

    # ✅ Accuracy
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
    
    



    # # Get permutation importance on the test set
    # result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

    # # Sort and plot
    # importances = result.importances_mean
    # sorted_idx = importances.argsort()
    # features = X_test.columns[sorted_idx]
    # visualization_folder = "static/visualization_images"

    # plt.figure(figsize=(10, 6))
    # plt.barh(features, importances[sorted_idx])
    # plt.xlabel("Mean Decrease in Accuracy")
    # plt.title("Permutation Feature Importance")
    # plt.tight_layout()
    # image_path = os.path.join(visualization_folder, "Permutation_Feature_Importance.png")
    # plt.savefig(image_path)
    # plt.close()

    
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
    