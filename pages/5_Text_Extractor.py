import streamlit as st
# import fitz  # PyMuPDF
import streamlit.components.v1 as components

st.title("📄 PDF Text Extractor with Copy Feature")

# File uploader
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

# extracted_text = ""

if uploaded_file:
    extracted_text = extract_text_from_file(uploaded_file)
    st.success("✅ Text extracted successfully.")


    st.subheader("📄 Extracted PDF Text")
    st.text_area("PDF Content", extracted_text, height=300, key="pdf_text")

    # Copy button (inject JS)
    copy_code = f"""
    <script>
    function copyToClipboard(text) {{
        navigator.clipboard.writeText(text);
        alert("✅ Text copied to clipboard!");
    }}
    </script>
    <button onclick="copyToClipboard(`{extracted_text}`)">📋 Copy to Clipboard</button>
    """
    components.html(copy_code, height=80)
