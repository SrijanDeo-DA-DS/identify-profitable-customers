import pandas as pd
import numpy as np
import os,sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
#from xgboost import XGBClassifier
import pickle
import yaml

params = yaml.safe_load(open('params.yaml','r'))['train_model']
## fetch the data

train_data = pd.read_csv('./data/processed/train_processed.csv')


X_train = train_data.drop(['important_customer'],axis=1)
y_train = train_data['important_customer']

rf = RandomForestClassifier(class_weight={0:1,1:7}, max_depth = params['max_depth'], min_samples_split = params['min_samples_split'],
                             min_samples_leaf = params['min_samples_leaf'])
rf.fit(X_train,y_train)


## save

pickle.dump(rf, open('./models/model.pkl', 'wb'))