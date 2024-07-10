import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Recommender Systems")

st.title("Recommend Similar Customers")

df = pd.read_csv('C:/Users/Srijan-DS/Documents/Projects/identify-profit-customer-profile/data/raw/raw.csv')

df = df.sample(n=5000, random_state=42)

bins = [20, 30, 40, 50]
labels = ['21-30', '31-40', '41-50']

df['age_bracket'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

st.dataframe(df)

st.subheader('Recommend on Profitable Customers or Non-Profitable Customers ?')

customer_type = st.selectbox('1 -> Profit 0 -> Non-Profit', df['important_customer'].unique())

st.subheader('Select Age Group:')

age_bracket = st.selectbox('Age Group', df['age_bracket'].unique())

## recommendation function

cosine_sim1 = pickle.load(open('C:/Users/Srijan-DS/Documents/Projects/identify-profitable-customers/artifacts/cosine_sim1.pkl','rb'))
cosine_sim2 = pickle.load(open('C:/Users/Srijan-DS/Documents/Projects/identify-profitable-customers/artifacts/cosine_sim2.pkl','rb'))

def recommend_customers(id, cosine_sim1, cosine_sim2):
    idx = df.index[df['id']==id].tolist()[0]

    cosine_sim = 1*cosine_sim1 + 2*cosine_sim2

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key = lambda i:i[1], reverse=True)

    sim_scores = sim_scores[1:6]

    customer_index = [i[0] for i in sim_scores]

    recommendations_df = pd.DataFrame(
        {
            'Customer_id' : df['id'].iloc[customer_index].tolist(),
            'SImilarity_score' : sim_scores
        }
    )

    return recommendations_df


if st.button('Filter'):
    new_df = df[df['important_customer']==customer_type]

    new_df_1 = new_df[new_df['age_bracket'] == age_bracket]

    st.dataframe(new_df_1)

selected_customer = int(st.selectbox('Select Customer ID for Recommendation', sorted(df.index.to_list())))

if st.button('Recommend'):
    recommendation_df = recommend_customers(df.index.to_list(),cosine_sim1,cosine_sim2)

    if not recommendation_df.empty:
        st.dataframe(recommendation_df)
    else:
        st.write('No recommendations found.')