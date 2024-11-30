import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Price Predictor")

with open('./pickle_files/df.pkl', 'rb') as file:
    df = pickle.load(file)

with open('./pickle_files/pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# st.write(df)

st.header('Enter your inputs')

sector = st.selectbox('Sector', sorted(df['sector'].unique().tolist()))

bedrooms = float(st.selectbox('Number of Bedrooms', sorted(df['bedRoom'].unique().tolist())))

bathrooms = float(st.selectbox('Number of Bathrooms', sorted(df['bathroom'].unique().tolist())))

balcony = st.selectbox('Balconies', sorted(df['balcony'].unique().tolist()))

property_age = st.selectbox('Property Age', sorted(df['agePossession'].unique().tolist()))

built_up_area = float(st.number_input('Built Up Area (sq.ft.)'))

extra_rooms = float(st.selectbox('Number of extra_rooms (eg: servant room, puja room)', sorted(df['extra_rooms'].unique().tolist())))

furnishing_type = st.selectbox('Furnishing Type', sorted(df['furnishing_type'].unique().tolist()))

luxury_category = st.selectbox('Luxury Category', sorted(df['luxury_category'].unique().tolist()))

floor_category = st.selectbox('Floor Category', sorted(df['floor_category'].unique().tolist()))

if st.button('Predict'):
    data = [[sector, built_up_area, bedrooms, bathrooms, balcony, extra_rooms, property_age, furnishing_type, luxury_category, floor_category]]
    columns = ['sector', 'built_up_area', 'bedRoom', 'bathroom', 'balcony',
               'extra_rooms', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']
    
    test_df = pd.DataFrame(data, columns=columns)

    # predict
    base_price = np.expm1(pipeline.predict(test_df))[0]
    lowest_price = base_price - 0.22
    highest_price = base_price + 0.22

    # display
    st.text("The price of the flat is between {} Cr and {} Cr".format(round(lowest_price, 2),round(highest_price, 2)))