import pandas as pd
import numpy as np
import os,sys
import pickle

df = pd.read_csv('C:/Users/Srijan-DS/Documents/Projects/identify-important-customers/data/raw/train_df.csv')

df.drop(['id'],axis=1,inplace=True)


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler

from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import VarianceThreshold

transformer_1 = ColumnTransformer(
    transformers = [('tnf1', SimpleImputer(strategy='mean'), [0]),
                    ('tnf2', SimpleImputer(strategy='median'), [6]),
                    ('tnf3', SimpleImputer(strategy='most_frequent'), [7,10,12,15,16,22])
                    ],
     remainder = 'passthrough'
).set_output(transform='pandas')

transformer_2 = ColumnTransformer(
    transformers = [('trf4', OneHotEncoder(sparse=False,drop='first'),[2])],
    remainder = 'passthrough'
).set_output(transform='pandas')

transformer_3 = ColumnTransformer(
    transformers = [('trf5', RobustScaler(),[1,8,9,11,18,19,23,24,25]),
                   ('trf6', MinMaxScaler(),[0,2,3,4,5,6,7,10,12,13,14,15,16,17,20,21,22])
                   ],
    remainder = 'passthrough'
).set_output(transform='pandas')

transformer_4 = ColumnTransformer(
    transformers = [('trf6', VarianceThreshold(threshold=0.05),[i for i in range(df.shape[1])])],
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

df = pipe.fit_transform(df)

extracted_names = [col.split('__')[-1] for col in df.columns]

df.columns = extracted_names

pickle.dump(df, open('data.pkl','wb'))