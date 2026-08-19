import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd

# Load saved model, scaler, and feature column order
model = joblib.load('housing_model.pkl')
scaler = pickle.load(open('housing_scaler.pkl', 'rb'))
feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))
st.set_page_config(page_title="Housing Price Predictor", page_icon="🏡", layout="wide")

# Custom CSS - Warm Earthy theme (terracotta / sand / sage)
st.markdown("""
    <style>
    @keyframes gradientShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .stApp {
        background: linear-gradient(-45deg, #F5EBDD, #E8D5B7, #EFE2CC, #F5EBDD);
        background-size: 400% 400%;
        animation: gradientShift 20s ease infinite;
    }
    section[data-testid="stSidebar"] {
        background: rgba(74, 55, 40, 0.95);
    }
    section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #F5EBDD !important;
}
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stTextInput input {
    background-color: #FFFFFF !important;
    color: #4A3728 !important;
}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #4A3728 !important;
}
    h1 { color: #4A3728 !important; font-weight: 800; }
    .subtitle { color: #8A6D4E; font-size: 16px; margin-bottom: 20px; font-weight: 500; }

    .glass-card {
        background: rgba(255, 255, 255, 0.45);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.6);
        border-radius: 22px;
        padding: 30px;
        box-shadow: 0 8px 28px rgba(74, 55, 40, 0.18);
        margin-top: 15px;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #C1662F, #8F4A1F);
        color: white; border: none; border-radius: 12px;
        padding: 12px 30px; font-size: 16px; font-weight: 700;
        width: 100%; transition: 0.3s;
    }
    div.stButton > button:hover { transform: scale(1.02); }

    .result-box {
        background: linear-gradient(135deg, rgba(135,168,120,0.25), rgba(193,102,47,0.15));
        backdrop-filter: blur(10px);
        border: 1.5px solid #87A878;
        color: #4A3728;
        padding: 30px; border-radius: 20px; text-align: center;
        margin-top: 20px;
    }
    .result-box h1 { font-size: 42px; margin: 0; color: #C1662F !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🏡 California Housing Price Predictor")
st.markdown('<p class="subtitle">Enter district details to estimate the median house value</p>', unsafe_allow_html=True)

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("📋 District Details")

longitude = st.sidebar.slider("Longitude", -124.5, -114.0, -119.5, 0.01)
latitude = st.sidebar.slider("Latitude", 32.5, 42.0, 36.5, 0.01)
housing_median_age = st.sidebar.slider("Median House Age (years)", 1, 52, 25)
total_rooms = st.sidebar.number_input("Total Rooms (in block)", min_value=1, value=2000)
total_bedrooms = st.sidebar.number_input("Total Bedrooms (in block)", min_value=1, value=400)
population = st.sidebar.number_input("Population (in block)", min_value=1, value=1000)
households = st.sidebar.number_input("Households (in block)", min_value=1, value=400)
median_income = st.sidebar.slider("Median Income (in $10,000s)", 0.5, 15.0, 3.5, 0.1)
ocean_proximity = st.sidebar.selectbox("Ocean Proximity", 
    ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

predict_btn = st.sidebar.button("🔍 Predict Price")

# ---------------- Main Area ----------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📍 Selected Location")
    map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
    st.map(map_df, zoom=5, size=200, color='#C1662F')
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🏘️ District Summary")
    st.write(f"**Rooms per household:** {total_rooms/households:.2f}")
    st.write(f"**Bedrooms per room:** {total_bedrooms/total_rooms:.2f}")
    st.write(f"**Population per household:** {population/households:.2f}")
    st.write(f"**Ocean Proximity:** {ocean_proximity}")
    st.markdown('</div>', unsafe_allow_html=True)

if predict_btn:
    rooms_per_household = total_rooms / households
    bedrooms_per_room = total_bedrooms / total_rooms
    population_per_household = population / households

    ocean_dummies = {
        'ocean_proximity_INLAND': 1 if ocean_proximity == 'INLAND' else 0,
        'ocean_proximity_ISLAND': 1 if ocean_proximity == 'ISLAND' else 0,
        'ocean_proximity_NEAR BAY': 1 if ocean_proximity == 'NEAR BAY' else 0,
        'ocean_proximity_NEAR OCEAN': 1 if ocean_proximity == 'NEAR OCEAN' else 0,
    }

    row = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        **ocean_dummies,
        'rooms_per_household': rooms_per_household,
        'bedrooms_per_room': bedrooms_per_room,
        'population_per_household': population_per_household,
    }

    input_df = pd.DataFrame([row])[feature_columns]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.markdown(f"""
    <div class="result-box">
        <p style="margin:0; font-size:16px;">Estimated Median House Value</p>
        <h1>${prediction:,.0f}</h1>
    </div>
    """, unsafe_allow_html=True)