identify-profitable-customers
==============================

This project aims to help business understand their Customers and increase CLV, aid targeted campaigning and retaining profitable customers by reducing Churn. There are 4 modules : -
1. **Data Analytics** : Understand your customer's demographics like Age/ Spending score/ Total income etc. in a comprehensive way and strategize your business around it
2. **Predictor** : If ever in doubt for a customer, use this tool to predict if a particular Customer will be profitable for your business or not. This is trained on the historic customer database and their behaviour
3. **Explainability** : Get detailed insights and root level reasons as to why a particular customer was tagged as 'Profitable' or 'Non-Profitable' by our model, what all features were important for the decision.
4. **Recommender System** : For running targeted campaigns, idenfity your Top 5 similar customers to the customer of your choice and increase the chance of business

## Code and Resources Used
* Python Version: 3.11
* Packages: pandas, numpy, sklearn, seaborn, streamlit, dvc, mlflow
* For Web Framework Requirements: pip install -r requirements.txt

## Website Features
#### 1. Analytics Dashboard
#### 2. Profitable Customers Predictor
#### 3. Model Explainability
#### 4. Recommender System

## 1. Analytics Dashboard
Users can filter the data by Profitable vs Non-Profitbale customers and look at the analysis. For ex - Age Distribution, Purchase Amount
![analytics_dash](https://github.com/user-attachments/assets/401f0b1e-62ee-4986-94ac-6a2d156cb364)

## 2. Predictor
Users can input various details about a customer and our model will predict if the customer is Profitable or Non-Profitable
![predictor](https://github.com/user-attachments/assets/de25e3c5-c3fc-4b5c-9663-fb3445a5a858)

## 3. Model Explainability
Users can visually see what all factors were responsible in our model's output of Profitablle vs Non-Profitable
![model_explain](https://github.com/user-attachments/assets/f92fb607-a4fd-4e29-bfd8-b761a507be4b)

## 4. Recommnder System
Users can get recommendation of Top 5 similar customers based on their Customer IDs for better marketing
![recommender](https://github.com/user-attachments/assets/28ac6974-e0e9-443b-83b1-c0abf4dad63d)

## MLflow
Used mlflow to track the model performance in different experiments. Also tracked models, code, images and other artifacts
![mlflow](https://github.com/user-attachments/assets/f63a6704-1d67-4c21-8de1-aa2eba22c3df)
Metrics -
![metrics](https://github.com/user-attachments/assets/e4950075-e75e-4987-9df9-fdae5b463ef9)
Arifacts -
![artifacts](https://github.com/user-attachments/assets/9b00aafa-11d3-406e-b48d-0f41253d9f63)



