import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Viz Demo")


pipe = pickle.load(open('C:/Users/Srijan-DS/Documents/Projects/identify-profitable-customers/artifacts/fe_pipe.pkl','rb'))

model = pickle.load(open('C:/Users/Srijan-DS/Documents/Projects/identify-profitable-customers/artifacts/model.pkl','rb'))

st.header('Enter your inputs')

    # Define the form components
purchase_amount = st.number_input("Purchase Amount")
asset_amount = st.number_input("Asset Amount", format="%d")
average_ratio = st.number_input("Average Ratio")
personal_id_1 = st.number_input("Personal ID 1", format="%d")
personal_id_2 = st.number_input("Personal ID 2", format="%d")
age = st.number_input("Age", format="%d")
area = st.number_input("Area")
job_type = st.text_input("Job Type")
phone = st.number_input("Phone", format="%d")
personal_card_1 = st.number_input("Personal Card 1", format="%d")
personal_card_2 = st.number_input("Personal Card 2")
personal_card_3 = st.number_input("Personal Card 3", format="%d")
personal_card_4 = st.number_input("Personal Card 4")
car = st.number_input("Car", format="%d")
purchase_score = st.number_input("Purchase Score", format="%d")
campaign_use = st.number_input("Campaign Use")
card_expired = st.number_input("Card Expired")
average_favorite_score = st.number_input("Average Favorite Score")
card_history_period = st.number_input("Card History Period")
score_1 = st.number_input("Score 1", format="%d")
score_2 = st.number_input("Score 2", format="%d")
score_3 = st.number_input("Score 3", format="%d")
score_4 = st.number_input("Score 4")
total_amount_1 = st.number_input("Total Amount 1", format="%d")
total_amount_2 = st.number_input("Total Amount 2", format="%d")
total_amount_3 = st.number_input("Total Amount 3", format="%d")
important_customer = st.number_input("Important Customer", format="%d")

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
    "total_amount_3": total_amount_3,
    "important_customer": important_customer
}

if st.button('Predict'):

    # form a dataframe
    # Iterate over the dictionary
    key_list = []
    values_list = []

    for key, value in user_input.items():
        user_input.append(key)
        values_list.append(value)

    # Convert to DataFrame
    one_df = pd.DataFrame(values_list, columns=user_input)

    st.dataframe(one_df)

    # predict
    #base_price = np.expm1(pipe.predict(one_df))[0]

    # display
    #st.text("The price of the flat is between {} Cr and {} Cr".format(round(low,2),round(high,2)))