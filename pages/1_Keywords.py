import streamlit as st # PyMuPDF
import requests
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
st.title("🔑 Keyword Extraction from PDF")

# === Helper: Split long text into chunks ===
def chunk_text(text, max_words=400):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx"])


if uploaded_file:
    extracted_text = extract_text_from_file(uploaded_file)
    st.success("✅ Text extracted successfully.")

    # === Process chunks ===
    chunks = list(chunk_text(extracted_text, max_words=400))
    st.info(f"Text split into {len(chunks)} chunks.")

    if st.button("🔍 Extract Keywords"):
        all_keywords = set()
        progress = st.progress(0)

        for i, chunk in enumerate(chunks):
            try:
                response = requests.post("http://127.0.0.1:8000/keywords/", json={"text": chunk})
                if response.status_code == 200:
                    keywords = response.json().get("keywords", [])
                    all_keywords.update(keywords)
                else:
                    st.warning(f"Chunk {i+1}: ❌ Error {response.status_code}")
            except Exception as e:
                st.warning(f"Chunk {i+1}: ❌ Request failed - {e}")

            progress.progress((i + 1) / len(chunks))

        st.subheader("🧠 Final Extracted Keywords")
        if all_keywords:
            st.write(", ".join(sorted(all_keywords)))
            st.download_button("📥 Download Keywords", data=", ".join(sorted(all_keywords)), file_name="keywords.txt")
        else:
            st.info("No keywords found.")
