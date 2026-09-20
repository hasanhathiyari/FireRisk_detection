import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="FireRisk AI", page_icon="🔥", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("fire_risk_model.joblib")

model = load_model()

st.title("🔥 FireRisk AI")
st.subheader("Weather-Based Fire Activity Prediction")
st.write(
    "Enter the weather and fire-weather index values below. "
    "The trained Random Forest model will classify the observation as "
    "**Fire** or **Not Fire**."
)

st.info(
    "Educational/statistical ML prototype. This model is based on the "
    "UCI Algerian Forest Fires dataset (Bejaia region, 2012) and is not "
    "an operational emergency-warning system."
)

with st.form("prediction_form"):
    st.markdown("### 🌡️ Weather conditions")
    c1, c2 = st.columns(2)

    with c1:
        temperature = st.number_input("Temperature (°C)", 0.0, 60.0, 30.0, 0.5)
        rh = st.number_input("Relative Humidity RH (%)", 0.0, 100.0, 45.0, 1.0)
        ws = st.number_input("Wind Speed (km/h)", 0.0, 100.0, 15.0, 0.5)
        rain = st.number_input("Rain (mm)", 0.0, 200.0, 0.0, 0.1)
        ffmc = st.number_input("FFMC", 0.0, 110.0, 85.0, 0.1)

    with c2:
        dmc = st.number_input("DMC", 0.0, 300.0, 25.0, 0.1)
        dc = st.number_input("DC", 0.0, 1000.0, 100.0, 0.5)
        isi = st.number_input("ISI", 0.0, 100.0, 8.0, 0.1)
        bui = st.number_input("BUI", 0.0, 300.0, 30.0, 0.1)
        fwi = st.number_input("FWI", 0.0, 100.0, 10.0, 0.1)

    submitted = st.form_submit_button(
        "🚀 Predict Fire Activity", use_container_width=True
    )

if submitted:
    input_data = pd.DataFrame([{
        "Temperature": temperature,
        "RH": rh,
        "Ws": ws,
        "Rain": rain,
        "FFMC": ffmc,
        "DMC": dmc,
        "DC": dc,
        "ISI": isi,
        "BUI": bui,
        "FWI": fwi,
    }])

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    st.markdown("---")
    st.markdown("### Prediction")

    if prediction == 1:
        st.error("🔥 FIRE ACTIVITY PREDICTED")
    else:
        st.success("✅ NO FIRE ACTIVITY PREDICTED")

    st.metric("Model probability of Fire", f"{probability * 100:.1f}%")

    st.write(
        "This classification is based on patterns learned from the training dataset. "
        "It is not a real-time fire alarm or emergency prediction."
    )

    with st.expander("View input values"):
        st.dataframe(input_data, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("FireRisk AI • BSc IT Statistics Mini Project • Random Forest")
