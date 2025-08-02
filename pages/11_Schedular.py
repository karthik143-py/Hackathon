import streamlit as st
import requests
import fitz  # PyMuPDF
from datetime import date
import json

st.set_page_config(page_title="📅 Study Scheduler", layout="wide")
st.title("📅 Personalized Study Scheduler")

# === Upload Section ===
uploaded_file = st.file_uploader("📄 Upload PDF or DOCX", type=["pdf", "docx"])
text_input = st.text_area("Or paste your custom text here:")

# === Extract Text ===
extracted_text = ""

if uploaded_file is not None:
    if uploaded_file.name.endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            extracted_text += page.get_text()
    elif uploaded_file.name.endswith(".docx"):
        from docx import Document
        from io import BytesIO
        document = Document(BytesIO(uploaded_file.read()))
        extracted_text += "\n".join([para.text for para in document.paragraphs])

elif text_input:
    extracted_text = text_input

if extracted_text:
    st.success("✅ Text ready for scheduling!")

    # === User Inputs ===
    mode = st.radio("What do you want to learn?", ["pdf", "concepts"], horizontal=True)
    iq = st.slider("📊 Rate your IQ (approx)", 20, 160, 100)
    hours_per_day = st.slider("⏱️ How many hours can you study daily?", 1.0, 12.0, 2.0)
    end_date = st.date_input("📆 Select your end date:", min_value=date.today())

    # === Schedule Button ===
    if st.button("📌 Generate Study Plan"):
        with st.spinner("Generating your schedule..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/schedule/",
                    json={
                        "text": extracted_text,
                        "mode": mode,
                        "iq": iq,
                        "hours_per_day": hours_per_day,
                        "end_date": str(end_date),
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    plan = result.get("plan", [])

                    if not plan:
                        st.warning("No plan generated. Try adjusting parameters.")
                    else:
                        st.success(f"✅ {len(plan)} days scheduled!")

                        # === Table ===
                        st.markdown("### 📖 Study Plan")
                        for day in plan:
                            st.markdown(f"""
**Day {day['day']}**
- 🧠 Concept: `{day['concept']}`
- 📌 Summary: {day['summary']}
""")

                        # === Downloadable File ===
                        plan_txt = "\n\n".join([f"Day {d['day']}\nConcept: {d['concept']}\nSummary: {d['summary']}" for d in plan])
                        st.download_button("📥 Download Study Plan (.txt)", data=plan_txt, file_name="study_plan.txt")

                else:
                    st.error(f"❌ Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"🚫 Failed to fetch study plan: {e}")
else:
    st.info("📂 Upload a file or paste text to begin.")
