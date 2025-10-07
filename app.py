import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime

# Load model and columns
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# ✅ Rain effect
st.markdown("""
<style>
.rain-container {
  position: fixed;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 1;
}
.drop {
  position: absolute;
  color: #3498db;
  animation: fall linear infinite;
  opacity: 0.6;
}
@keyframes fall {
  0% { transform: translateY(-100px); }
  100% { transform: translateY(100vh); }
}
</style>
<div class="rain-container">
""" +
"\n".join([
    f"<div class='drop' style='left:{np.random.randint(0,100)}vw; font-size:{np.random.randint(15,25)}px; animation-duration:{np.random.uniform(2, 6):.2f}s;'>💧</div>"
    for _ in range(40)
]) + "</div>", unsafe_allow_html=True)

# Title and header
st.title("💧 Water Pollutants Predictor")
st.markdown("🚰 *Predict pollutant levels using Year and Station ID*")

# Sidebar inputs
st.sidebar.header("🧮 Input Parameters")
year_input = st.sidebar.number_input("Enter Year", min_value=2000, max_value=2100, value=2022)
station_id = st.sidebar.text_input("Enter Station ID", value='1')

if st.sidebar.button('🔍 Predict'):
    if not station_id:
        st.warning('⚠️ Please enter a valid Station ID')
    else:
        input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        # Add missing columns
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        predicted = model.predict(input_encoded)[0]
        pollutants = ['NH4', 'BSK5', 'Suspended', 'O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
        max_val = max(predicted)
        percent_values = [(val / max_val) * 100 if max_val != 0 else 0 for val in predicted]

        st.toast("🎉 Prediction Complete!", icon="✅")
        st.subheader(f"📊 Results for Station `{station_id}` in `{year_input}`")
        
        result_df = pd.DataFrame({
            'Pollutant': pollutants,
            'Predicted Value': [round(val, 2) for val in predicted],
            'Estimated % Level': [f"{round(p, 1)}%" for p in percent_values]
        })

        st.dataframe(result_df, use_container_width=True)

        # 📥 Download button
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv,
            file_name=f'pollutant_prediction_{station_id}_{year_input}.csv',
            mime='text/csv'
        )

        # 📅 Prediction time
        st.caption(f"🕒 Prediction generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 🔄 Animated pollutant levels
        st.markdown("📈 **Visual Pollutant Levels**")
        for p, val, perc in zip(pollutants, predicted, percent_values):
            counter = st.empty()
            for i in range(0, int(val), max(1, int(val)//10)):
                counter.markdown(f"🌡 **{p} Level:** {i:.2f}")
                time.sleep(0.01)
            counter.markdown(f"🌡 **{p} Level:** {val:.2f}")
            st.progress(min(int(perc), 100))

# ✅ Footer (final version)
st.markdown("""
    <hr style="border-top: 1px solid #bbb;">
    <center>
        Made by <strong>Krisha Chavan</strong>
    </center>
""", unsafe_allow_html=True)
