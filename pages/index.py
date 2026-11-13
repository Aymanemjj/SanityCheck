import joblib
import pandas as pd
import streamlit as st

def ginki_dama(data):
    data = data.copy()
    data["Start_Age"] = data["Age"] - data["Year_of_Study"].round()
    data["Social_Media_Hours"] = data["Social_Media_Hours"].clip(upper=data["Screen_Time_Hours"])
    # mask = data["Social_Media_Hours"] > data["Screen_Time_Hours"]
    #
    # data.loc[mask, ["Social_Media_Hours", "Screen_Time_Hours"]] = (
    #     data.loc[mask, ["Screen_Time_Hours", "Social_Media_Hours"]].values
    # )
    return data

def ginki_names(transformer, input_features):
    return list(input_features) + ["Start_Age"]

MODEL_PATH = "../notebooks/LRP.joblib"
LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}

FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Year_of_Study",
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Physical_Activity",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate",
    "Part_Time_Job",
    "Diet_Quality",
    "Previous_Mental_Health_History",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Student Mental Health Predictor")
st.title("Student Mental Health Predictor")
st.caption("Fill in the student's details and click Predict.")

model = load_model()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=20, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Non-Binary"])
        year_of_study = st.number_input("Year of study", min_value=1, max_value=8, value=2, step=1)
        sleep_hours = st.number_input("Sleep hours (per night)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        study_hours = st.number_input("Study hours per week", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
        gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.0, step=0.1)
        social_media = st.number_input("Social media hours (per day)", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
        screen_time = st.number_input("Screen time hours (per day)", min_value=0.0, max_value=24.0, value=6.0, step=0.5)

    with col2:
        physical_activity = st.selectbox("Physical activity", ["Low", "Moderate", "High"])
        financial_stress = st.slider("Financial stress", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
        family_support = st.slider("Family support", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
        attendance = st.slider("Attendance rate (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
        part_time_job = st.selectbox("Part-time job", ["No", "Yes"])
        diet_quality = st.selectbox("Diet quality", ["Poor", "Average", "Good"])
        previous_history = st.selectbox("Previous mental health history", ["No", "Yes"])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame(
        [
            {
                "Age": age,
                "Gender": gender,
                "Year_of_Study": year_of_study,
                "Sleep_Hours": sleep_hours,
                "Study_Hours_Per_Week": study_hours,
                "GPA": gpa,
                "Social_Media_Hours": social_media,
                "Screen_Time_Hours": screen_time,
                "Physical_Activity": physical_activity,
                "Financial_Stress": financial_stress,
                "Family_Support": family_support,
                "Attendance_Rate": attendance,
                "Part_Time_Job": part_time_job,
                "Diet_Quality": diet_quality,
                "Previous_Mental_Health_History": previous_history,
            }
        ]
    )[FEATURE_COLUMNS]

    try:
        pred = model.predict(input_df)[0]

        label = LABEL_MAP[int(pred)]
        st.subheader("Prediction")
        st.metric("Predicted level", label)

    except Exception as e:
        st.error(f"Prediction failed: {e}")