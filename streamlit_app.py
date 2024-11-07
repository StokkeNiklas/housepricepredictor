# Import necessary libraries
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('housing_price_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Define the correct order of features
feature_order = [
    'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
    'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
    'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea'
]

# Set a fixed exchange rate from USD to NOK
usd_to_nok_rate = 10.95  # Update this with the latest exchange rate

# Define your Streamlit interface
st.title("Housing Price Estimator")
st.write("Enter the housing details to get an estimated price in NOK.")

# Input fields for each feature
data = {
    'LotArea': st.number_input("Lot Area (square meters)", min_value=0.0) * 10.764,
    'OverallQual': st.slider("Overall Quality (1-10)", min_value=1, max_value=10),
    'OverallCond': st.slider("Overall Condition (1-10)", min_value=1, max_value=10),
    'YearBuilt': st.number_input("Year Built", min_value=1800, max_value=2024),
    'YearRemodAdd': st.number_input("Year Remodeled", min_value=1800, max_value=2024),
    'GrLivArea': st.number_input("Ground Living Area (square meters)", min_value=0.0) * 10.764,
    'FullBath': st.slider("Number of Full Bathrooms", min_value=0, max_value=5),
    'HalfBath': st.slider("Number of Half Bathrooms", min_value=0, max_value=5),
    'BedroomAbvGr': st.slider("Bedrooms Above Ground", min_value=0, max_value=10),
    'KitchenAbvGr': st.slider("Kitchens Above Ground", min_value=0, max_value=5),
    'TotRmsAbvGrd': st.slider("Total Rooms Above Ground", min_value=0, max_value=20),
    'Fireplaces': st.slider("Number of Fireplaces", min_value=0, max_value=5),
    'GarageCars': st.slider("Number of Cars Garage Can Hold", min_value=0, max_value=5),
    'GarageArea': st.number_input("Garage Area (square meters)", min_value=0.0) * 10.764
}

# Convert data to DataFrame
input_data = pd.DataFrame([data], columns=feature_order)

# Predict house price
if st.button("Estimate Price"):
    try:
        prediction_usd = loaded_model.predict(input_data)[0]
        prediction_nok = prediction_usd * usd_to_nok_rate * 1.446
        formatted_prediction_nok = f"{prediction_nok:,.2f} NOK"
        st.success(f"Estimated Price: {formatted_prediction_nok}")
    except Exception as e:
        st.error("Error in making the prediction: " + str(e))
