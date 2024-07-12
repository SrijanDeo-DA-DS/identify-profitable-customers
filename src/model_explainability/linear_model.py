import pandas as pd
import numpy as np
import os,sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


train_data = pd.read_csv('./data/processed/train_processed.csv')


X_train = train_data.drop(['important_customer'],axis=1)
y_train = train_data['important_customer']

lr = LogisticRegression(class_weight={0:1,1:7},n_jobs=-1)

lr.fit(X_train,y_train)

## read test data
test_data = pd.read_csv('./data/processed/test_processed.csv')

X_test = test_data.drop(['important_customer'],axis=1)
y_test = test_data['important_customer']

y_pred = lr.predict(X_test)

print(classification_report(y_test,y_pred))

# Get coefficients
coefficients = lr.coef_

# Intercept (if applicable)
intercept = lr.intercept_

print("Coefficients:", sorted(coefficients))
print("Intercept:", intercept)