import streamlit as st
import pickle
import pandas as pd

# Load models and encoders
model = pickle.load(open('pred.pkl', 'rb'))
symptom_encoders = pickle.load(open('symptom_encoders.pkl', 'rb'))
disease_encoder = pickle.load(open('disease_encoder.pkl', 'rb'))

clf_pipeline = pickle.load(open('clf_pipeline.pkl', 'rb'))
reg_pipeline = pickle.load(open('reg_pipeline.pkl', 'rb'))

# Utility functions
def categorize_bp(bp_input):
    try:
        parts = bp_input.split('/')
        if len(parts) == 2:
            systolic = int(parts[0].strip())
            diastolic = int(parts[1].strip())
        else:
            systolic = int(bp_input.strip())
            diastolic = 0
    except Exception:
        return 'Unknown'
    if systolic < 90 or diastolic < 60:
        return 'Low'
    elif 90 <= systolic <= 120 and 60 <= diastolic <= 80:
        return 'Normal'
    elif systolic > 120 or diastolic > 80:
        return 'High'
    else:
        return 'Unknown'

# Get all symptom classes (union of all symptom_encoders)
all_symptoms = set()
for le in symptom_encoders.values():
    all_symptoms.update(le.classes_)
all_symptoms = sorted(all_symptoms)

st.title("Disease and Treatment Prediction")

st.header("Step 1: Predict Disease from Symptoms")

symptoms = []
cols = st.columns(3)
for i in range(6):
    with cols[i % 3]:
        symptom = st.selectbox(f"Symptom {i+1}", options=[""] + all_symptoms, key=f"symptom_{i}")
        symptoms.append(symptom)

if st.button("Predict Disease"):
    if "" in symptoms:
        st.error("Please select all 6 symptoms")
    else:
        try:
            encoded = []
            for i, symptom in enumerate(symptoms):
                col_name = f'Symptom_{i+1}'
                le = symptom_encoders[col_name]
                if symptom in le.classes_:
                    encoded.append(le.transform([symptom])[0])
                else:
                    st.error(f"Invalid symptom: {symptom}")
                    break
            else:
                pred = model.predict([encoded])[0]
                disease = disease_encoder.inverse_transform([pred])[0]
                st.success(f"Predicted Disease: {disease}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.header("Step 2: Predict Treatment Plan")

disease_input = st.text_input("Disease (from above or type manually):")

age = st.number_input("Age", min_value=0, max_value=120, value=30)
bp = st.text_input("Blood Pressure (Systolic/Diastolic, e.g., 120/80):", value="120/80")
heart_beats = st.number_input("Heart Beats per minute", min_value=30, max_value=200, value=70)
oxygen_level = st.number_input("Oxygen Level (%)", min_value=50, max_value=100, value=98)
past_history = st.text_area("Past History (optional)")

if st.button("Predict Treatment"):
    if disease_input.strip() == "":
        st.error("Please enter the disease name.")
    else:
        bp_category = categorize_bp(bp)
        combined_text = f"{disease_input} {past_history} {bp_category}"
        input_df = pd.DataFrame({
            'combined_text': [combined_text],
            'Age': [age],
            'Heart_Beats': [heart_beats],
            'Oxygen_Level': [oxygen_level]
        })

        pred_class = clf_pipeline.predict(input_df)[0]
        pred_reg = reg_pipeline.predict(input_df)[0]

        medicines = [m for m in pred_class[:4] if isinstance(m, str) and m.strip() and m.lower() != 'nan']
        exercises = [e for e in pred_class[4:7] if isinstance(e, str) and e.strip() and e.lower() != 'nan']

        st.subheader("Treatment Plan")
        st.write(f"Blood Pressure Category: {bp_category}")
        st.write(f"Expected Recovery Days: {round(pred_reg, 2)}")
        st.write("Medicines:")
        for med in medicines:
            st.write(f"- {med}")
        st.write("Exercises:")
        for ex in exercises:
            st.write(f"- {ex}")
