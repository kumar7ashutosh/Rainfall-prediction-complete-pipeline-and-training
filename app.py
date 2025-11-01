import ssl, certifi, warnings
warnings.filterwarnings("ignore")

# ---- Fix SSL errors ----
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from meteostat import Point, Hourly
from streamlit_autorefresh import st_autorefresh
from streamlit_folium import st_folium
import folium

# ===============================
# Page Setup
# ===============================
st.set_page_config(page_title="Realtime Rainfall Predictor", page_icon="🌧️", layout="centered")
st.title("🌦️ Realtime Rainfall Prediction Dashboard")
st.caption("Fetch live weather data from any location and predict rainfall in real time.")

# ===============================
# Cached Loaders
# ===============================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load("best_model.pkl")
    pipeline = joblib.load("pipeline.pkl")
    return model, pipeline

try:
    model, pipeline = load_artifacts()
except Exception as e:
    st.error(f"❌ Could not load model/pipeline: {e}")
    st.stop()

# ===============================
# Auto Refresh every 10 minutes
# ===============================
st_autorefresh(interval=10 * 60 * 1000, key="auto_refresh")

# ===============================
# Sidebar: Location Selection
# ===============================
st.sidebar.header("🗺️ Select Location on Map")
st.sidebar.write("Click anywhere on the map to fetch live data for that location.")

# Persistent state
if "lat" not in st.session_state:
    st.session_state.lat = None
if "lon" not in st.session_state:
    st.session_state.lon = None

lat = st.session_state.lat
lon = st.session_state.lon

# ===============================
# Map Click Mode Only
# ===============================
default_lat, default_lon = 22.5726, 88.3639
m = folium.Map(location=[default_lat, default_lon], zoom_start=5)
folium.LatLngPopup().add_to(m)
map_data = st_folium(m, width=700, height=500)

if map_data and map_data.get("last_clicked"):
    clicked = map_data["last_clicked"]
    new_lat, new_lon = clicked["lat"], clicked["lng"]

    if new_lat != st.session_state.lat or new_lon != st.session_state.lon:
        st.session_state.lat, st.session_state.lon = new_lat, new_lon
        st.rerun()

# ===============================
# Fetch Live Weather + Predict
# ===============================
lat = st.session_state.lat
lon = st.session_state.lon

if lat is not None and lon is not None:
    try:
        location = Point(lat, lon)
        end = datetime.utcnow()
        start = end - timedelta(hours=6)
        data = Hourly(location, start, end).fetch()

        if data.empty:
            st.warning("⚠️ No live data available for this location right now.")
        else:
            latest = data.tail(1).reset_index()
            fetch_time = latest["time"].iloc[0] if "time" in latest.columns else latest.index[0]

            st.success(f"✅ Live data fetched for ({lat:.4f}, {lon:.4f}) at {fetch_time:%Y-%m-%d %H:%M UTC}")
            st.dataframe(latest)

            # Optional chart
            if len(data) > 3:
                st.subheader("📈 Temperature & Humidity (Last 24h)")
                chart_cols = [c for c in ["temp", "rhum"] if c in data.columns]
                if chart_cols:
                    df_plot = data.copy().reset_index()
                    df_plot = df_plot[df_plot["time"] >= datetime.utcnow() - timedelta(hours=24)]
                    st.line_chart(df_plot.set_index("time")[chart_cols])

            # ---- Map Meteostat → Model Features ----
            feature_map = {
                "temp": "temparature",
                "pres": "pressure",
                "rhum": "humidity",
                "dwpt": "dewpoint",
                "wspd": "windspeed"
            }
            for m_col, mdl_col in feature_map.items():
                if m_col in latest.columns:
                    latest.rename(columns={m_col: mdl_col}, inplace=True)

            model_features = getattr(pipeline, "feature_names_in_", None)
            if model_features is None:
                st.error("Your pipeline doesn’t expose feature names.")
                st.stop()

            for col in model_features:
                if col not in latest.columns:
                    latest[col] = 0

            X_live = latest[model_features]
            X_scaled = pipeline.transform(X_live)
            prediction = model.predict(X_scaled)[0]

            st.subheader("🌤️ Rainfall Prediction")
            if int(prediction) == 1:
                st.success("☔ **Rain likely at this location!**")
            else:
                st.info("🌞 **No rain expected right now.**")

    except Exception as e:
        st.error(f"Error fetching data: {e}")

else:
    st.info("👆 Click on the map to select a location and fetch live data.")

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption("Built with ❤️ by ashutosh • Powered by Meteostat, Streamlit & your trained model.")
