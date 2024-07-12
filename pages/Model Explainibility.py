import pandas as pd
import numpy as np
import os,sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap
import streamlit as st
import pickle
import pandas as pd

st.set_option('deprecation.showPyplotGlobalUse', False)

model = pickle.load(open('C:/Users/Srijan-DS/Documents/Projects/identify-profitable-customers/artifacts/model.pkl','rb'))

training_data_order = pickle.load(open('C:/Users/Srijan-DS/Documents/Projects/identify-profitable-customers/artifacts/training_data_order.pkl','rb'))

## training shap
train_data = pd.read_csv('C:/Users/Srijan-DS/Documents/Projects/identify-profitable-customers/data/processed/train_processed.csv')


X_train = train_data.drop(['important_customer'],axis=1)
y_train = train_data['important_customer']

rf = RandomForestClassifier(class_weight={0:1,1:7},n_jobs=-1)

rf.fit(X_train,y_train)
explainer = shap.Explainer(rf, X_train)

#UI
st.title('Model Explainability')
st.subheader('You can enter any customer ID and the model will explain the output')

customer_id = st.selectbox('Select Customer ID', training_data_order.index.tolist())

if st.button('Explain'):
    
    test_data = pd.DataFrame([training_data_order.iloc[customer_id]])

    test_data = test_data.reindex(columns=training_data_order.columns)

    prediction = model.predict(test_data)[0]

    if prediction == 0:
        result = 'Non-Profitable'
    else:
        result = 'Profitable'

    st.text(f"The customer is: {result}")

    shap_values = explainer(test_data, check_additivity=False)

    customer_id = int(customer_id)

    fig = shap.plots.waterfall(shap_values[0][:,1])


    st.pyplot(fig)
