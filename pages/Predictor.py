import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Viz Demo")

df = pickle.load(open('C:/Users/Srijan-DS/Documents/Projects/identify-profitable-customers/artifacts/data.pkl','rb'))

pipe = pickle.load(open('C:/Users/Srijan-DS/Documents/Projects/identify-profitable-customers/artifacts/fe_pipe.pkl','rb'))

model = pickle.load(open('C:/Users/Srijan-DS/Documents/Projects/identify-profitable-customers/artifacts/model.pkl','rb'))

st.header('Enter your inputs')

    # Define the form components
purchase_amount = float(st.number_input("Purchase Amount"))
asset_amount = float(st.number_input("Asset Amount"))
average_ratio = float(st.number_input("Average Ratio"))
personal_id_1 = float(st.number_input("Personal ID 1"))
personal_id_2 = float(st.number_input("Personal ID 2"))
age = st.selectbox("Age", sorted(df['age'].unique().tolist()))
area = float(st.number_input("Area"))
job_type = st.selectbox("Job Type", df['job_type_Self employed'].unique().tolist())
phone = st.selectbox("Phone", df['phone'].unique().tolist())
personal_card_1 = st.selectbox("Personal Card 1", df['personal_card_1'].unique().tolist())
personal_card_2 = st.selectbox("Personal Card 2", df['personal_card_2'].unique().tolist())
personal_card_3 = st.selectbox("Personal Card 3", df['personal_card_3'].unique().tolist())
personal_card_4 = st.selectbox("Personal Card 4", df['personal_card_4'].unique().tolist())
car = st.selectbox("Car", df['car'].unique().tolist())
purchase_score = float(st.number_input("Purchase Score"))
campaign_use = st.selectbox("Campaign Use", df['campaign_use'].unique().tolist())
card_expired = st.selectbox("Card Expired", df['card_expired'].unique().tolist())
average_favorite_score = float(st.number_input("Average Favorite Score"))
card_history_period = float(st.number_input("Card History Period"))
score_1 = st.selectbox("Score 1", df['score_1'].unique().tolist())
score_2 = st.selectbox("Score 2", df['score_2'].unique().tolist())
score_3 = st.selectbox("Score 3", df['score_3'].unique().tolist())
score_4 = st.selectbox("Score 4", df['score_4'].unique().tolist())
total_amount_1 = float(st.number_input("Total Amount 1"))
total_amount_2 = float(st.number_input("Total Amount 2"))
total_amount_3 = float(st.number_input("Total Amount 3"))

    # Collect user input into a dictionary
user_input = {
    "purchase_amount": purchase_amount,
    "asset_amount": asset_amount,
    "average_ratio": average_ratio,
    "personal_id_1": personal_id_1,
    "personal_id_2": personal_id_2,
    "age": age,
    "area": area,
    "job_type": job_type,
    "phone": phone,
    "personal_card_1": personal_card_1,
    "personal_card_2": personal_card_2,
    "personal_card_3": personal_card_3,
    "personal_card_4": personal_card_4,
    "car": car,
    "purchase_score": purchase_score,
    "campaign_use": campaign_use,
    "card_expired": card_expired,
    "average_favorite_score": average_favorite_score,
    "card_history_period": card_history_period,
    "score_1": score_1,
    "score_2": score_2,
    "score_3": score_3,
    "score_4": score_4,
    "total_amount_1": total_amount_1,
    "total_amount_2": total_amount_2,
    "total_amount_3": total_amount_3
}

if st.button('Predict'):

    one_df = pd.DataFrame([user_input])

    st.dataframe(one_df)

    #df_transformed = pipe.transform(one_df)

    prediction = model.predict(one_df)

    st.text("The customer will {}".format(prediction[0]))