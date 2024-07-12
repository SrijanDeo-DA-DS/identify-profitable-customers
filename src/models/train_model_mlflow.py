import numpy as np
import os,sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import mlflow
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")

params = yaml.safe_load(open('params.yaml','r'))['train_model']

train_data = pd.read_csv('./data/processed/train_processed.csv')


X_train = train_data.drop(['important_customer'],axis=1)
y_train = train_data['important_customer']

rf = RandomForestClassifier(class_weight={0:1,1:7}, max_depth = params['max_depth'], min_samples_split = params['min_samples_split'],
                             min_samples_leaf = params['min_samples_leaf'])

dt = DecisionTreeClassifier(class_weight={0:1,1:7}, max_depth = params['max_depth'], min_samples_split = params['min_samples_split'],
                             min_samples_leaf = params['min_samples_leaf'])

dt.fit(X_train,y_train)

## read test data
test_data = pd.read_csv('./data/processed/test_processed.csv')

X_test = test_data.drop(['important_customer'],axis=1)
y_test = test_data['important_customer']

mlflow.set_experiment("DecisonTree")

with mlflow.start_run():
    # import model
    y_pred = dt.predict(X_test)

    # Generate classification report
    #report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('recall', recall)

    mlflow.log_param('max_depth', params['max_depth'])
    mlflow.log_param('min_samples_split', params['min_samples_split'])
    mlflow.log_param('min_samples_leaf', params['min_samples_leaf'])

    print('Accuracy', accuracy)
    print('Recall', recall)

    cm =confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(dt, "DecisionTreeClassifier")
    mlflow.set_tag("model","Dt")
    mlflow.set_tag("author","srijan")
