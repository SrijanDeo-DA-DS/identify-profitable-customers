import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import VarianceThreshold

## fetch the data

train_data = pd.read_csv('./data/interim/training_data.csv')
test_data = pd.read_csv('./data/interim/testing_data.csv')

## transformation

transformer_1 = ColumnTransformer(
    transformers = [('tnf1', SimpleImputer(strategy='mean'), [0]),
                    ('tnf2', SimpleImputer(strategy='median'), [6]),
                    ('tnf3', SimpleImputer(strategy='most_frequent'), [7,10,12,15,16,22])
                    ],
     remainder = 'passthrough'
).set_output(transform='pandas')

transformer_2 = ColumnTransformer(
    transformers = [('trf4', OneHotEncoder(sparse_output=False,drop='first'),[2])],
    remainder = 'passthrough'
).set_output(transform='pandas')

transformer_3 = ColumnTransformer(
    transformers = [('trf5', RobustScaler(),[1,8,9,11,18,19,23,24,25]),
                   ('trf6', MinMaxScaler(),[0,2,3,4,5,6,7,10,12,13,14,15,16,17,20,21,22])
                   ],
    remainder = 'passthrough'
).set_output(transform='pandas')

transformer_4 = ColumnTransformer(
    transformers = [('trf6', VarianceThreshold(threshold=0.0005),[i for i in range(train_data.shape[1])])],
    remainder = 'passthrough'
).set_output(transform='pandas')

pipe = Pipeline(
    [
        ('transformer_1', transformer_1),
        ('transformer_2', transformer_2),
        ('transformer_3', transformer_3),
        ('transformer_4', transformer_4)
    ]
)

train_data = pipe.fit_transform(train_data)
test_data = pipe.transform(test_data)

extracted_names_train = [col.split('__')[-1] for col in train_data.columns]
extracted_names_test = [col.split('__')[-1] for col in test_data.columns]

train_data.columns = extracted_names_train
test_data.columns = extracted_names_test

data_path = os.path.join("data","processed")

os.mkdir(data_path)

train_data.to_csv(os.path.join(data_path,"train_processed.csv"),index=False)
test_data.to_csv(os.path.join(data_path,"test_processed.csv"),index=False)