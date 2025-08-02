# pages/outline.py

import streamlit as st
import fitz  # PyMuPDF
import requests

st.set_page_config(page_title="🧭 Smart Outline", layout="wide")
st.title("🧭 Smart Outline from PDF")

# === PDF Upload ===
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
    full_text = extract_text_from_file(uploaded_file)
    st.success("✅ Text extracted successfully.")


    st.success("✅ PDF loaded and text extracted.")
    st.text_area("Extracted Text (Preview)", full_text[:1000] + "...", height=200)

    # === Generate Outline ===
    if st.button("🪄 Generate Smart Outline"):
        with st.spinner("Using Gemini to generate outline..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/outline/",
                    json={"text": full_text}
                )
                if response.status_code == 200:
                    outline = response.json().get("outline", [])
                    if outline:
                        st.success("✅ Outline generated!")

                        # Display outline
                        st.subheader("📚 Outline")
                        for i, item in enumerate(outline, start=1):
                            st.markdown(f"**{i}. {item}**")

                        # Allow download
                        outline_text = "\n".join(f"{i}. {item}" for i, item in enumerate(outline, 1))
                        st.download_button(
                            label="📥 Download Outline as .txt",
                            data=outline_text,
                            file_name="document_outline.txt",
                            mime="text/plain"
                        )

                        # Sidebar navigation
                        
                    else:
                        st.warning("⚠️ Gemini didn't return any outline.")
                else:
                    st.error(f"❌ FastAPI Error: {response.status_code}")
            except Exception as e:
                st.error(f"🚫 Request failed: {e}")
