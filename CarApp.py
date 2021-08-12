# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 12:54:25 2021

@author: Carmen
"""

import streamlit as st
import pandas as pd
import pickle
from PIL import Image

PATH = "AppFiles/"
image = Image.open(PATH + "logo4.png")
st.image(image, use_column_width=True)

st.title('CAR AUCTION SALE PREDICTION')

st.sidebar.header('User Input Features')

car2 = pd.read_csv(PATH + "CarSalePrices.csv")

def user_input_features():
    make_model = st.sidebar.selectbox('Make/Model', set(car2['make_model']))
    body = st.sidebar.selectbox('Body', set(car2['body']))
    state = st.sidebar.selectbox('State', set(car2['state']))
    transmission = st.sidebar.selectbox('Transmission', set(car2['transmission']))
    color = st.sidebar.selectbox('Color', set(car2['color']))
    interior = st.sidebar.selectbox('Interior', set(car2['interior']))
    year = st.sidebar.slider('Year', min_value=1982, max_value=2015)
    condition = st.sidebar.slider('Condition', step= 0.1, min_value=1.0, max_value=5.0)
    odometer = st.sidebar.slider('Odometer', min_value=1, max_value=100000)


    data1 = { 
        'year': year,
        'make_model': make_model,
        'body': body,
        'transmission': transmission,
        'state': state,
        'condition': condition,
        'odometer': odometer,
        'color': color,
        'interior': interior
            }
    

    features = pd.DataFrame(data1, index=[0])
    return features
input_df = user_input_features()


car_raw = pd.read_csv(PATH + "CarSalePrices.csv")

car = car_raw.drop(columns=['sellingprice'])
df = pd.concat([input_df, car], axis=0)


encode = ['make_model', 'body', 'transmission', 'state', 'color', 'interior']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1]
df.drop_duplicates()

col1 = pd.DataFrame(df.columns)

st.subheader('User Input Features')

st.write('Currently using example input parameters (shown below).')
st.write(df)

load_rgr = pickle.load(open(PATH + "CarAuctionPriceFinal.pkl", 'rb'))

prediction = load_rgr.predict(df)

st.subheader('Prediction')

st.write(round(prediction[-1], 2))

st.write('')
st.write('')


st.write("""
            python Libraries: streamlit, pandas, base64, matplotlib, seaborn, numpy
            
            Data Source: https://www.kaggle.com/tunguz/used-car-auction-prices
        """)




