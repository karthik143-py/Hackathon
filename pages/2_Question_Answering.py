import streamlit as st
# import fitz  # PyMuPDF
import requests

st.title("❓ Question Answering (PDF or Text)")
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        import fitz  # PyMuPDF
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])

    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        from docx import Document
        from io import BytesIO
        doc = Document(BytesIO(uploaded_file.read()))
        return "\n".join([para.text for para in doc.paragraphs])

    else:
        return ""
# File uploader for PDF
uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx"])

extracted_text = ""

if uploaded_file is not None:
    extracted_text = extract_text_from_file(uploaded_file)
    st.success("✅ Text extracted successfully.")
    
    
final_text = extracted_text

# Question input
question = st.text_input("Your Question:")

# Button to get answers
if st.button("Get Answers"):
    if final_text.strip() == "" or question.strip() == "":
        st.warning("❗ Please upload a PDF or enter text, and provide a question.")
    else:
        with st.spinner("Searching for answers..."):
            try:
                response = requests.post("http://127.0.0.1:8000/question/", json={"text": final_text, "question": question})
                if response.status_code == 200:
                    answers = response.json().get("answers", [])
                    if not answers:
                        st.info("No answers found.")
                    else:
                        st.success("✅ Answers:")
                        for ans in answers:
                            st.write(f"**{ans['answer']}**  _(Confidence: {ans['score']:.2f})_")
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Request failed: {e}")
