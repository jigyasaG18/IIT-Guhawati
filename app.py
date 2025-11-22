import base64
import joblib
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import random
import os
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import numpy as np
import re

# ---------- CONFIG ----------
st.set_page_config(page_title="DV Risk & Support", layout="centered", initial_sidebar_state="expanded")
if not os.path.exists("user_history"):
    os.makedirs("user_history")
if not os.path.exists("audio_records"):
    os.makedirs("audio_records")

# ---------- LOAD MODELS / DATA ----------
# Primary DV risk model & scaler (existing files)
model = joblib.load("domestic_violence_model.pkl")
scaler = joblib.load("scaler.pkl")

@st.cache_data
def load_police_stations():
    df = pd.read_csv("dataset/Police_Stations_India.csv")
    df["Phone Number"] = df["Phone Number"].astype(str)
    return df

df_police_stations = load_police_stations()

# ---------- UTILITIES ----------
def save_pdf(bytes_io, filename):
    with open(filename, "wb") as f:
        f.write(bytes_io.getbuffer())

def create_safety_plan_pdf(user_info: dict, recommendations: list):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Empower Sakhi - Personalized Safety Plan")
    c.setFont("Helvetica", 11)
    y = height - 90
    c.drawString(50, y, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 20
    for k, v in user_info.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 16
    y -= 8
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Recommended Safety Steps:")
    y -= 18
    c.setFont("Helvetica", 11)
    for idx, rec in enumerate(recommendations, 1):
        if y < 80:
            c.showPage()
            y = height - 80
            c.setFont("Helvetica", 11)
        c.drawString(60, y, f"{idx}. {rec}")
        y -= 16
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def generate_report_pdf(result_record, user_inputs):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Empower Sakhi - Assessment Report")
    c.setFont("Helvetica", 11)
    y = height - 90
    c.drawString(50, y, f"Timestamp: {result_record['timestamp']}")
    y -= 20
    c.drawString(50, y, f"Risk Score: {result_record['risk_score']}%")
    y -= 24
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Inputs Provided:")
    y -= 18
    c.setFont("Helvetica", 11)
    for k, v in user_inputs.items():
        if y < 80:
            c.showPage()
            y = height - 80
            c.setFont("Helvetica", 11)
        c.drawString(50, y, f"- {k}: {v}")
        y -= 14
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def keyword_message_score(text):
    text_low = text.lower()

    # English + Hindi danger keywords
    danger_keywords = [
        # English
        "kill", "beat", "hit", "threat", "rape", "hurt", "stab", "break", "leave",
        "die", "disappear", "shut up", "control", "isolate", "no one will help",
        "kill you", "die alone",

        # Hindi threats (transliterated & Hindi words)
        "maar dunga", "mar dungi", "mar dalunga", "mar dalu", "zinda nahi", 
        "zinda nahi chhodunga", "khatam kar dunga", " jaan se maar", 
        "tod dunga", "pita dunga", "thappad", "laat mar", "tumhari zindagi barbaad",
        "ghar se nikal", "nikal ja", "bahar ja", "chup raho", "chup kar",
        "koi madad nahi karega", "akeli chhod dunga", "main sab control karta hoon"
    ]

    abusive_keywords = [
        # English
        "idiot", "stupid", "worthless", "useless",

        # Hindi insults
        "kameeni", "kamina", "bewakoof", "pagal", "nalayak", "haramzada",
        "ghatiya", "kichad", "kachra", "faltu"
    ]

    score = 0

    # Danger scoring
    for kw in danger_keywords:
        if kw in text_low:
            score += 3

    for kw in abusive_keywords:
        if kw in text_low:
            score += 1

    # Pattern-based Hindi threats
    if re.search(r"(agar tum\s+.*?\s+to\s+)", text_low):  # "agar tum ... to"
        score += 2

    if re.search(r"(tumhari\s+.*?\s+kar dunga)", text_low):  # "tumhari ... kar dunga"
        score += 2

    return min(score, 20)

def dv_type_detector(inputs: dict):
    # rule-based detection of abuse types
    types = set()
    if inputs.get("partner_alcoholic") == "Yes" or inputs.get("self_substance_abuse") == "Yes":
        types.add("Substance-related")
    if inputs.get("past_violence") == "Yes" or inputs.get("previous_reports",0) > 0:
        types.add("Physical")
    # if controlling indicators present
    if inputs.get("has_support_system") == "No" or inputs.get("housing_situation") in ["Shelter", "Homeless", "With relatives"]:
        types.add("Isolation / Financial")
    # simple text-based fields could be used later
    if not types:
        types.add("Emotional/Verbal")
    return ", ".join(types)

def approximate_feature_impact(input_df, model, scaler, delta=0.05):
    # small sensitivity test: perturb each feature by delta fraction of its value (or +1 for categorical) and measure probability change
    base_scaled = scaler.transform(input_df)
    base_prob = model.predict_proba(base_scaled)[0][1]
    impacts = {}
    for col in input_df.columns:
        perturbed = input_df.copy()
        val = perturbed.at[0, col]
        if isinstance(val, (int, float, np.integer, np.floating)):
            change = val * delta if abs(val) > 0 else delta
            perturbed.at[0, col] = val + change
        else:
            # for categorical encoded ints
            try:
                perturbed.at[0, col] = val + 1
            except:
                perturbed.at[0, col] = val
        new_prob = model.predict_proba(scaler.transform(perturbed))[0][1]
        impacts[col] = round((new_prob - base_prob) * 100, 3)  # percentage point change
    # sort by absolute impact
    impacts_sorted = dict(sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True))
    return impacts_sorted, round(base_prob*100,3)

# ---------- MAIN UI COMPONENTS ----------

def risk_assessment_form():
    st.header("👩 Risk Assessment Form")
    # Inputs (same as original)
    age = st.slider("Age", 18, 50, 30)
    education_levels = ["None", "Primary", "Lower Secondary", "Upper Secondary", "Diploma/Technical", "Undergraduate", "Postgraduate"]
    education = st.selectbox("Education", education_levels)
    income = st.selectbox("Monthly Income", ["<5000", "5000-10000", "10000-20000", ">20000"])
    marital_status = st.selectbox("Marital Status", ["Married", "Unmarried", "Divorced", "Widowed"])
    children = st.number_input("Number of Children", 0, 10, 0)
    has_partner = st.selectbox("Do you currently have a partner?", ["Yes", "No"])
    partner_alcoholic = st.selectbox("Partner alcoholic?", ["Yes", "No"])
    has_support = st.selectbox("Has support system?", ["Yes", "No"])
    past_violence = st.selectbox("Past violence?", ["Yes", "No"])
    mental_issues = st.selectbox("Mental health concerns?", ["Yes", "No"])
    employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed", "Part-time", "Student", "Homemaker"])
    housing_situation = st.selectbox("Housing Situation", ["Own", "Rent", "Shelter", "Homeless", "With relatives"])
    disability = st.selectbox("Disability?", ["Yes", "No"])
    self_substance_abuse = st.selectbox("Substance Abuse?", ["Yes", "No"])
    previous_reports = st.number_input("Previous Reports", 0, 10, 0)

    encode_map = {
        "None": 0, "Primary": 1, "Lower Secondary": 2, "Upper Secondary": 3,
        "Diploma/Technical": 4, "Undergraduate": 5, "Postgraduate": 6,
        "<5000": 0, "5000-10000": 1, "10000-20000": 2, ">20000": 3,
        "Married": 0, "Unmarried": 1, "Divorced": 2, "Widowed": 3,
        "Yes": 1, "No": 0,
        "Employed": 0, "Unemployed": 1, "Part-time": 2, "Student": 3, "Homemaker": 4,
        "Own": 0, "Rent": 1, "Shelter": 2, "Homeless": 3, "With relatives": 4
    }

    input_dict = {
        "age": age,
        "education": encode_map[education],
        "income": encode_map[income],
        "marital_status": encode_map[marital_status],
        "number_of_children": children,
        "has_partner": encode_map[has_partner],
        "partner_alcoholic": encode_map[partner_alcoholic],
        "has_support_system": encode_map[has_support],
        "past_violence": encode_map[past_violence],
        "mental_health_issues": encode_map[mental_issues],
        "employment_status": encode_map[employment_status],
        "housing_situation": encode_map[housing_situation],
        "disability": encode_map[disability],
        "self_substance_abuse": encode_map[self_substance_abuse],
        "previous_reports": previous_reports
    }

    input_data = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_data)

    if st.button("🔍 Assess Risk"):
        prediction = model.predict(input_scaled)[0]
        risk_score = model.predict_proba(input_scaled)[0][1]
        label = "High" if prediction == 1 else "Low"

        if prediction == 1:
            st.error("⚠️ High risk of domestic violence.")
        else:
            st.success("✅ Low risk of domestic violence.")
        st.metric("Risk Score", f"{risk_score * 100:.1f}%")

        # save to history
        history_file = "user_history/risk_history.csv"
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "risk_score": round(risk_score * 100, 2),
            "prediction": label
        }
        record_df = pd.DataFrame([record])
        if os.path.exists(history_file):
            old = pd.read_csv(history_file)
            new = pd.concat([old, record_df], ignore_index=True)
            new.to_csv(history_file, index=False)
        else:
            record_df.to_csv(history_file, index=False)
        st.success("📌 Saved to your risk history.")

        # show approximate feature impact (SHAP-like sensitivity)
        with st.expander("Why this score? (Feature sensitivity)"):
            impacts, base = approximate_feature_impact(input_data, model, scaler, delta=0.05)
            st.write(f"Base Risk Probability: {base}%")
            top_n = 5
            for i, (feat, val) in enumerate(impacts.items()):
                if i >= top_n:
                    break
                sign = "+" if val >= 0 else ""
                st.write(f"{i+1}. **{feat}**: {sign}{val} pp change if increased slightly")

        # DV type detection
        abuse_types = dv_type_detector({
            "partner_alcoholic": "Yes" if partner_alcoholic=="Yes" else "No",
            "self_substance_abuse": "Yes" if self_substance_abuse=="Yes" else "No",
            "past_violence": "Yes" if past_violence=="Yes" else "No",
            "previous_reports": previous_reports,
            "has_support_system": has_support,
            "housing_situation": housing_situation
        })
        st.info(f"Likely abuse types detected: **{abuse_types}**")

        # quick safety plan recommendation (simple rules)
        recs = []
        if prediction == 1:
            recs.append("Contact emergency helpline immediately (112 / 1091).")
            recs.append("Move to a safe place; avoid isolated areas.")
            recs.append("Tell a trusted friend or neighbor about your situation.")
            recs.append("Document injuries and incidents (photos, dates).")
            recs.append("Consider filing a police report / FIR at nearest station.")
        else:
            recs.append("Maintain a safety contact and keep evidence backups.")
            recs.append("Seek counseling if you notice escalation in behavior.")
        # show download safety plan
        if st.button("📝 Download Personalized Safety Plan (PDF)"):
            user_info = {
                "Age": age, "Marital Status": marital_status, "Children": children,
                "Employment": employment_status, "Housing": housing_situation
            }
            pdf_buffer = create_safety_plan_pdf(user_info, recs)
            st.download_button("Download Safety Plan PDF", data=pdf_buffer, file_name="safety_plan.pdf", mime="application/pdf")

        # offer to generate full report
        if st.button("📄 Download Assessment Report (PDF)"):
            history_file = "user_history/risk_history.csv"
            last_record = pd.read_csv(history_file).iloc[-1].to_dict() if os.path.exists(history_file) else record
            pdf_buf = generate_report_pdf(last_record, input_dict)
            st.download_button("Download Assessment Report", data=pdf_buf, file_name="assessment_report.pdf", mime="application/pdf")

def hear_her_stories():
    st.subheader("📖 Hear Her Stories")
    st.markdown("Real, anonymous words from brave women — just like you.")
    stories = [
        "💬 'I thought I couldn't live without him. But I learned to live for me.'",
        "💬 'I was silenced for years. Speaking up was the first breath I took.'",
        "💬 'Leaving was hard. Healing was harder. But I did both.'",
    ]
    if st.button("🔄 Show Me a Story"):
        st.markdown(f"<div style='padding:15px;background:#f0f0f5;border-left:4px solid #e91e63;border-radius:6px;'>{random.choice(stories)}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.info("💡 Email us your story at empowersakhi@mail.com")

# ---------- NEW FEATURES PAGES ----------
def message_danger_detector_page():
    st.header("💬 Message Danger Detector")
    st.markdown("Paste a suspicious message (WhatsApp / SMS). The tool provides a danger score and highlights risky phrases.")
    text = st.text_area("Paste message here", height=150)
    if st.button("Analyze Message"):
        if not text.strip():
            st.warning("Please paste a message to analyze.")
            return
        score = keyword_message_score(text)
        st.metric("Danger Score (0-20)", score)
        if score >= 8:
            st.error("High danger indicators detected.")
        elif score >= 4:
            st.warning("Moderate danger indicators detected.")
        else:
            st.success("Low danger indicators detected.")
        # highlight keywords
        keywords_found = []
        for kw in ["kill", "beat", "hit", "threat", "rape", "hurt", "pack", "isolate", "die", "if you leave"]:
            if kw in text.lower():
                keywords_found.append(kw)
        if keywords_found:
            st.markdown("**Risk keywords found:** " + ", ".join(keywords_found))

def dv_type_page():
    st.header("🧭 Abuse Type Detector")
    st.markdown("Quickly estimate probable abuse type based on inputs.")
    col1, col2 = st.columns(2)
    with col1:
        partner_alcoholic = st.selectbox("Partner alcoholic?", ["Yes","No"])
        past_violence = st.selectbox("Past violence?", ["Yes","No"])
        has_support = st.selectbox("Has support system?", ["Yes","No"])
    with col2:
        self_substance = st.selectbox("Self substance abuse?", ["Yes","No"])
        housing = st.selectbox("Housing situation", ["Own","Rent","Shelter","Homeless","With relatives"])
        previous_reports = st.number_input("Previous reports", 0, 10, 0)
    if st.button("Detect Types"):
        types = dv_type_detector({
            "partner_alcoholic": partner_alcoholic,
            "self_substance_abuse": self_substance,
            "past_violence": past_violence,
            "previous_reports": previous_reports,
            "has_support_system": has_support,
            "housing_situation": housing
        })
        st.success(f"Likely abuse types: {types}")

def audio_sos_page():
    st.header("🎙️ Audio SOS Recorder")
    st.markdown("You can upload a short audio clip (or record externally and upload). The file will be saved locally so you can use it as evidence.")
    uploaded = st.file_uploader("Upload or record audio (mp3/wav)", type=["mp3","wav","m4a"])
    if uploaded:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_records/sos_{ts}_{uploaded.name}"
        with open(filename, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Audio saved as {filename}")
        st.audio(uploaded)

def daily_checkin_page():
    st.header("📅 Daily Safety & Mood Check-in")
    mood = st.radio("How are you feeling today?", ["Safe", "Somewhat Safe", "Unsafe", "Scared", "Injured"])
    note = st.text_area("Optional note (private)", height=80)
    if st.button("Save Check-in"):
        hist_file = "user_history/checkins.csv"
        rec = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "mood": mood, "note": note}
        df_rec = pd.DataFrame([rec])
        if os.path.exists(hist_file):
            old = pd.read_csv(hist_file)
            new = pd.concat([old, df_rec], ignore_index=True)
            new.to_csv(hist_file, index=False)
        else:
            df_rec.to_csv(hist_file, index=False)
        st.success("Check-in saved.")
    if os.path.exists("user_history/checkins.csv"):
        df = pd.read_csv("user_history/checkins.csv")
        st.subheader("Recent Check-ins")
        st.dataframe(df.tail(10))

def knowledge_hub_page():
    st.header("📚 Knowledge Hub — Rights & Legal Steps (India)")
    st.markdown("""
    **What is Domestic Violence?**  
    Domestic violence includes physical, emotional, sexual, economic, and psychological abuse.

    **Key Actions:**  
    - Call emergency helpline: **112 / 1091**  
    - File an FIR at the nearest police station (use 'Nearby Resources' page)  
    - Contact legal aid at NALSA or local NGOs  
    """)
    st.markdown("**Useful Links:**")
    st.markdown("- [NCW Online Complaint](https://ncwapps.nic.in/onlinecomplaintsv2/)")
    st.markdown("- [NALSA Legal Aid](https://nalsa.gov.in/)")
    st.markdown("- [Women Shelters List (India)](https://spuwac.in/shelterhomesw.html)")

# ---------- HEADER / SIDEBAR ----------
st.markdown("<h1 style='text-align: center; font-size: 40px; font-weight: bold;'>🛡️ EMPOWER SAKHI 🛡️</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; font-style: italic;'>Your All-Time Support for Domestic Violence Risk & Safety</h3>", unsafe_allow_html=True)

with st.sidebar:
    st.header("🆘 Options")
    if st.button("🚨 Emergency Help"):
        st.session_state.page = "help"
    if st.button("🔍 Risk Assessment"):
        st.session_state.page = "risk_assessment"
    if st.button("📈 Risk Progress Tracker"):
        st.session_state.page = "progress"
    if st.button("📍 Nearby Resources"):
        st.session_state.page = "nearby_resources"
    if st.button("💬 Message Detector"):
        st.session_state.page = "message_detector"
    if st.button("🧭 Abuse Type"):
        st.session_state.page = "dv_type"
    if st.button("🎙️ Audio SOS"):
        st.session_state.page = "audio_sos"
    if st.button("📅 Daily Check-in"):
        st.session_state.page = "daily_checkin"
    if st.button("📚 Knowledge Hub"):
        st.session_state.page = "knowledge_hub"
    if st.button("📖 Hear Her Stories"):
        st.session_state.page = "hear_her_stories"
    if st.button("🚪 Quick Exit"):
        st.session_state.page = "quick_exit"

# ---------- PAGE HANDLER ----------
if "page" in st.session_state:
    if st.session_state.page == "help":
        st.error("🚨 Emergency — Immediate Help")
        st.markdown("- 📞 112  -  Emergency")
        st.markdown("- 📞 1091 - Women Helpline")
        st.markdown("- 📞 181  - Women Power Line")
        st.markdown("- 📞 100  - Police")
        st.image("https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExbmd2c3R1ZHYzczYyYzRhYmFoZTNucDkwZjlxbnlrbjExenM1b210YyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/oMS7ZYPF1jtfV48Lxr/giphy.gif", use_container_width=True)

    elif st.session_state.page == "risk_assessment":
        risk_assessment_form()

    elif st.session_state.page == "progress":
        st.header("📈 Your Risk Progress Over Time")
        history_file = "user_history/risk_history.csv"
        if not os.path.exists(history_file):
            st.info("No history found. Complete a risk assessment first.")
        else:
            df_hist = pd.read_csv(history_file)
            st.subheader("Assessment History")
            st.dataframe(df_hist.tail(20))
            st.subheader("Risk Trend")
            st.line_chart(df_hist.set_index("timestamp")["risk_score"])

    elif st.session_state.page == "nearby_resources":
        st.subheader("Enter Your Location")
        location_input = st.text_input("City/State (e.g., South Delhi, Delhi)")
        if location_input:
            parts = location_input.split(",")
            city = parts[0].strip().lower()
            state = parts[1].strip().lower() if len(parts) == 2 else None
            if state:
                mask = df_police_stations["State"].str.lower() == state
            else:
                mask = df_police_stations["City"].str.lower().str.contains(city)
            results = df_police_stations[mask]
            if not results.empty:
                st.success("📍 Showing results:")
                st.dataframe(results[["Police Station Name","City","State","Email","Phone Number"]])
            else:
                st.warning("No matching resources found.")

    elif st.session_state.page == "message_detector":
        message_danger_detector_page()

    elif st.session_state.page == "dv_type":
        dv_type_page()

    elif st.session_state.page == "audio_sos":
        audio_sos_page()

    elif st.session_state.page == "daily_checkin":
        daily_checkin_page()

    elif st.session_state.page == "knowledge_hub":
        knowledge_hub_page()

    elif st.session_state.page == "hear_her_stories":
        hear_her_stories()

    elif st.session_state.page == "quick_exit":
        # Quick exit: clear page and redirect to neutral site
        st.markdown("<script>window.location.href='https://www.google.com';</script>", unsafe_allow_html=True)

else:
    st.markdown("<h4 style='text-align:center;'>Select an option from the sidebar.</h4>", unsafe_allow_html=True)