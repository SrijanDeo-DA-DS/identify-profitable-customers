import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import json
import yaml
from dvclive import Live

## read test data
test_data = pd.read_csv('./data/processed/test_processed.csv')

X_test = test_data.drop(['important_customer'],axis=1)
y_test = test_data['important_customer']


# import model
rf = pickle.load(open('./models/model.pkl', 'rb'))

y_pred = rf.predict(X_test)

# Generate classification report
#report = classification_report(y_test, y_pred, output_dict=True)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# load parameters for logging
with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

#params = yaml.safe_load(open('params.yaml','r'))['train_model']

# log metrics and paramteres 
with Live(save_dvc_exp=True) as live:
    live.log_metric('accuracy', accuracy)
    live.log_metric('precision', precision)
    live.log_metric('recall', recall)
    live.log_metric('f1_score', f1)

    for param,value in params.items():
        for key,val in value.items():
            live.log_param(f'{param}_{key}', val)

metrics = {
    'accuracy': accuracy,
    'precision' : precision,
    'recall' : recall,
    'f1_score' : f1
}

report_json = json.dumps(metrics, indent=4)

with open('metrics.json', 'w') as json_file:
    json_file.write(report_json)

#with open('metrics.json', 'w') as file:
#    json.dump(metrics_dict, file, indent=4)