import pandas as pd, streamlit as st, joblib
pipeline=joblib.load('pipeline.pkl')
model=joblib.load('best_model.pkl')
st.title('Rainfall predictor')
st.write('Enter weather conditions')
pressure = st.number_input("Pressure (hPa)")
humidity = st.number_input("Humidity (%)")
cloud = st.number_input("Cloud (%)")
sunshine = st.number_input("Sunshine (hours)")
winddirection = st.number_input("Wind Direction (°)")
windspeed = st.number_input("Wind Speed (km/h)")
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
