import sklearn
import pickle
import pandas as pd
import streamlit as st
import dill

st.title(':blue[Aid Allocation for Countries App]')

st.markdown(''':blue-background[Enter the below fields to check if a country needs financial aid or not]''')

col1, col2, col3 = st.columns(3)
with col1:
    child_mortality = st.number_input("Enter the Child Mortality Rate")
    total_fertility = st.number_input("Enter the Total Fertility Rate")
    life_expectancy = st.number_input("Enter the Life Expectancy")
with col2:
    health = st.number_input("Enter the health spending per capita")
    gdp = st.number_input("Enter the GDP")
    inflation = st.number_input("Enter the Inflation Rate")
with col3:
    income = st.number_input("Enter the Income per person")
    imports = st.number_input("Enter the Imports per capita")
    exports = st.number_input("Enter the Exports per capita")


with open("threshold.pkl", "rb") as file:
        threshold = dill.load(file)

with open("Kmeans_model.pkl", "rb") as file:
        model = dill.load(file)

with open("scaler.pkl", "rb") as file:
        scaler = dill.load(file)

def feature_engineer(df, threshold):
  df['High_Child_Mortality'] = (df['child_mort'] > threshold['child_mort_threshold']).astype(int)

  df['High_Inflation'] = (df['inflation'] > threshold['inflation_threshold']).astype(int)

  df['High_Total_Fertility'] = (df['total_fer'] > threshold['total_fer_threshold']).astype(int)

  df['Export_Import_Ratio'] = df['exports'] / df['imports']

  df['Health_GDP_Ratio'] = df['health'] / df['gdpp']

  return df

if st.button("Predict"):
    dict = {
        'child_mort' : [child_mortality],
        'exports' : [exports],
        'health' : [health],
        'imports' : [imports],
        'income' : [income],
        'inflation' : [inflation],
        'life_expec' : [life_expectancy],
        'total_fer' : [total_fertility],
        'gdpp' : [gdp]
    }
    df = pd.DataFrame(dict)

    df = feature_engineer(df, threshold)

    scale_columns = ['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec', 'total_fer',
                     'gdpp', 'Export_Import_Ratio', 'Health_GDP_Ratio']
    non_scale_columns = ['High_Child_Mortality', 'High_Inflation', 'High_Total_Fertility']

    df_scale = df[scale_columns]

    df_scaled = pd.DataFrame(scaler.transform(df_scale), columns=df_scale.columns, index=df_scale.index)

    df_scaled[non_scale_columns] = df[non_scale_columns]

    category_map = {
        0 : 'Developing',
        1 : 'Developed',
        2 : 'Under developed'
    }
    category = model.predict(df_scaled)
    category_name = category_map[category[0]]

    if category == 0 or category == 1:
        st.text(f"This country seems to be a {category_name}, Hence they don't need financial aid")
    else:
        st.text(f"This country seems to be a {category_name}, Hence they need financial aid")


