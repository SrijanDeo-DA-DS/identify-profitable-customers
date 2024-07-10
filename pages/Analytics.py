import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Visualization")

st.title("Analytics Dashboard")

new_df = pd.read_csv('C:/Users/Srijan-DS/Documents/Projects/identify-profit-customer-profile/data/raw/raw.csv')


## Pie chart to show imbalance

# Title of the application
st.title('Streamlit Pie Chart Example')

# Calculate normalized value counts
value_counts = new_df['important_customer'].value_counts(normalize=True)

# Plotting a pie chart
st.subheader('Pie Chart of Important Customers')
fig, ax = plt.subplots()
ax.pie(value_counts, labels=value_counts.index, autopct='%0.2f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig)  # Show plot using Streamlit

# Additional information or text if needed
st.write('This is a simple example of a pie chart in Streamlit.')

## Plot to show distributions
# Title of the application
# Title of the application
st.title('Streamlit Seaborn Distribution Plots Example')

# Sidebar widget for filtering important customers
important_customer_filter = st.selectbox(
    'Filter by Important Customer',
    [1, 0]
)

# Filter data based on the selected value
new_df_1 = new_df[new_df['important_customer'] == important_customer_filter]

# Plotting distribution plots using Seaborn
st.subheader('Distribution Plots')

# Distribution plot for age
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(new_df_1['age'], kde=True, bins=10, color='skyblue')
plt.title('Age of Population')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Distribution plot for purchase amount
plt.subplot(1, 2, 2)
sns.histplot(new_df_1['purchase_amount'], kde=True, bins=10, color='lightgreen')
plt.title('Distribution of Purchase Amount')
plt.xlabel('Purchase Amount')
plt.ylabel('Frequency')

plt.tight_layout()

# Display plots using Streamlit
st.pyplot(plt)

# Additional information or text if needed
st.write('This is a simple example of distribution plots in Streamlit.')

## Box plot

# Title of the application
st.title('Streamlit Seaborn Box Plots Example')

# Plotting box plots using Seaborn
st.subheader('Box Plots')

# Box plot for Age
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.boxplot(y=new_df['age'], color='skyblue')
plt.title('Distribution of Age')
plt.ylabel('Age')

# Box plot for purchase amount
plt.subplot(1, 2, 2)
sns.boxplot(y=new_df['purchase_amount'], color='lightgreen')
plt.title('Distribution of Purchase Amount')
plt.ylabel('Purchase Amount')

plt.tight_layout()

# Display plots using Streamlit
st.pyplot(plt)

# Additional information or text if needed
st.write('This is a simple example of box plots in Streamlit.')


## Heatmap

# Title of the application
st.title('Streamlit Seaborn Heatmap Example')

# Pivot table and heatmap
st.subheader('Heatmap of Purchase Amount by Customer Importance and Campaign Use')

# Pivot table calculation
pivot_table = pd.pivot_table(new_df, index='important_customer', columns='campaign_use',
                             values='purchase_amount', aggfunc='mean')

# Round to 2 decimal places
pivot_table_rounded = np.round(pivot_table, 2)

# Plot heatmap using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table_rounded, annot=False, fmt='.2f', cmap='YlGnBu', cbar=True)
plt.title('Average Purchase Amount')
plt.xlabel('Campaign Use')
plt.ylabel('Important Customer')

# Display heatmap using Streamlit
st.pyplot(plt)

# Additional information or text if needed
st.write('This is a simple example of a heatmap in Streamlit.')

## Scatterplot

# Title of the application
st.title('Streamlit Seaborn Scatter Plot Example')

# Plotting scatter plot using Seaborn
st.subheader('Scatter Plot of Purchase Amount by Card History Period')

plt.figure(figsize=(12, 8))
sns.scatterplot(x=new_df['card_history_period'], y=new_df['purchase_amount'], hue=new_df['important_customer'])
plt.title('Scatter Plot')
plt.xlabel('Card History Period')
plt.ylabel('Purchase Amount')
plt.legend(title='Important Customer')

# Display scatter plot using Streamlit
st.pyplot(plt)

# Additional information or text if needed
st.write('This is a simple example of a scatter plot in Streamlit.')