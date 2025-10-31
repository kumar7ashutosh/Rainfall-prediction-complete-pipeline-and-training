import pandas as pd, streamlit as st, joblib
pipeline=joblib.load('pipeline.pkl')
model=joblib.load('best_model.pkl')
st.title('Rainfall predictor')
st.write('Enter weather conditions')
pressure = st.number_input("Pressure (hPa)", value=1020.8)
humidity = st.number_input("Humidity (%)", value=77)
cloud = st.number_input("Cloud (%)", value=34)
sunshine = st.number_input("Sunshine (hours)", value=9.1)
winddirection = st.number_input("Wind Direction (°)", value=30.0)
windspeed = st.number_input("Wind Speed (km/h)", value=24.4)
if st.button('Predict RainFall'):
  input_data = pd.DataFrame([[
        pressure, humidity, cloud, sunshine, winddirection, windspeed
    ]], columns=['pressure', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed'])
  processed = pipeline.transform(input_data)
    
    prediction = model.predict(processed)
    
    if prediction[0] == 1:
        st.success("Rainfall expected!")
    else:
        st.info("No rainfall predicted.")
