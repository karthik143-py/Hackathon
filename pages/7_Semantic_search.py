import streamlit as st
import fitz  # PyMuPDF
import requests

st.set_page_config(page_title="🔍 Semantic Search", layout="wide")
st.title("🔍 Semantic Search (Chat with your PDF)")

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

# === Upload PDF ===
uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx"])
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

if uploaded_file:
    text = extract_text_from_file(uploaded_file)
    st.success("✅ Text extracted successfully.")
    st.session_state.pdf_text = text

# === Search Input ===
query = st.text_input("Ask a question or type a search query:")
if st.button("Search"):
    if not st.session_state.pdf_text:
        st.warning("❗ Please upload a PDF first.")
    elif not query.strip():
        st.warning("❗ Please enter a query.")
    else:
        with st.spinner("🔎 Searching..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/search/",
                    json={"text": st.session_state.pdf_text, "query": query}
                )
                if response.status_code == 200:
                    results = response.json()["results"]
                    st.success("✅ Top Results:")
                    for idx, res in enumerate(results, 1):
                        st.markdown(f"**{idx}.** {res['text']} _(Score: {res['score']:.2f})_")
                else:
                    st.error(f"❌ Error: {response.status_code}")
            except Exception as e:
                st.error(f"🚫 Request failed: {e}")
