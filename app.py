import streamlit as st
import pandas as pd
import pickle


model = pickle.load(open("model.pkl", "rb"))


st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Predictor")
st.markdown("### Predict house prices using Machine Learning")


col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input("Total Sqft", min_value=300, max_value=10000, value=1000)
    bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)

with col2:
    bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)


data = pd.read_csv("bengaluru_house_prices.csv")
locations = sorted(data['location'].dropna().unique())

location = st.selectbox("Select Location", locations)


if st.button("Predict Price"):

    
    if sqft <= 0:
        st.error("❌ Please enter valid square feet")
    else:
        input_df = pd.DataFrame({
            'total_sqft': [sqft],
            'bath': [bath],
            'bhk': [bhk],
            'location': [location]
        })

        try:
            prediction = model.predict(input_df)
            st.success(f"💰 Estimated Price: ₹ {prediction[0]:.2f} Lakhs")
        except Exception as e:
            st.error("Something went wrong!")
            st.write(e)


st.markdown("---")
st.markdown("Built with ❤️ by Aryan Pardhi")