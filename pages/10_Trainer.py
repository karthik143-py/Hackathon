import streamlit as st
import fitz
import docx
import requests
from io import BytesIO
import re

st.set_page_config(page_title="🧪 Interactive Q&A Trainer", layout="wide")
st.title("🧪 Interactive Q&A Trainer")

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(BytesIO(uploaded_file.read()))
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# State setup
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = []
    st.session_state.answers = {}
    st.session_state.current_q = 0
    st.session_state.correct_answers = 0

uploaded_file = st.file_uploader("📄 Upload a PDF or DOCX file", type=["pdf", "docx"])

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(uploaded_file)
    else:
        extracted_text = extract_text_from_docx(uploaded_file)

    st.success("✅ Text extracted!")

    num_q = st.number_input("How many questions to generate?", min_value=1, max_value=20, value=5)

    if st.button("🎯 Generate Quiz"):
        with st.spinner("Generating questions..."):
            response = requests.post("http://127.0.0.1:8000/quiz/", json={"text": extracted_text, "n": num_q})
            if response.status_code == 200:
                raw = response.json()["quiz"]
                if raw.lower().startswith("not enough"):
                    st.warning("Not enough content.")
                else:
                    # Parse quiz
                    quiz_blocks = re.findall(r"Q\d+\..+?(?=(?:Q\d+\.|$))", raw, re.DOTALL)
                    for q in quiz_blocks:
                        lines = q.strip().split("\n")
                        question = lines[0]
                        options = [opt for opt in lines[1:5]]
                        answer_line = [l for l in lines if l.lower().startswith("answer")]
                        correct = answer_line[0][-1].lower() if answer_line else "a"
                        st.session_state.quiz_data.append({
                            "question": question,
                            "options": options,
                            "correct": correct
                        })
                    st.session_state.current_q = 0
                    st.session_state.answers = {}
                    st.session_state.correct_answers = 0
                    st.success("✅ Quiz ready! Start below.")

# === Test UI ===
if st.session_state.quiz_data:
    current = st.session_state.current_q
    total = len(st.session_state.quiz_data)

    if current < total:
        q = st.session_state.quiz_data[current]
        st.markdown(f"### ❓ {q['question']}")
        choice = st.radio("Choose your answer:", options=q["options"], key=f"q{current}")

        if st.button("Next"):
            # Extract chosen label (a/b/c/d)
            selected = choice[0].lower()
            st.session_state.answers[current] = selected
            if selected == q["correct"]:
                st.session_state.correct_answers += 1
            st.session_state.current_q += 1
            st.rerun()
    else:
        st.success(f"🎉 Quiz complete! You scored **{st.session_state.correct_answers} / {total}**")
        if st.download_button("📥 Download Results", 
                              data=f"Score: {st.session_state.correct_answers}/{total}\n" +
                                   "\n".join([f"{q['question']}\nYour: {st.session_state.answers[i]} | Correct: {q['correct']}" 
                                              for i, q in enumerate(st.session_state.quiz_data)]),
                              file_name="quiz_results.txt"):
            pass
