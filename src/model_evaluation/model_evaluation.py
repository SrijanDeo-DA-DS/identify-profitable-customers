import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import json

## read test data
test_data = pd.read_csv('./data/processed/test_processed.csv')

X_test = test_data.drop(['important_customer'],axis=1)
y_test = test_data['important_customer']


# import model
rf = pickle.load(open('./models/model.pkl', 'rb'))

y_pred = rf.predict(X_test)

# Generate classification report
report = classification_report(y_test, y_pred, output_dict=True)

metrics_dict = {
    'classification_report':report
}

report_json = json.dumps(report, indent=4)

with open('metrics.json', 'w') as json_file:
    json_file.write(report_json)

#with open('metrics.json', 'w') as file:
#    json.dump(metrics_dict, file, indent=4)