import streamlit as st
# import fitz  # PyMuPDF
import requests
from gtts import gTTS
import tempfile
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

st.title("📄 PDF & Text Summarizer")

# === Helper: Split long text into chunks ===
def chunk_text(text, max_words=500):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

# === File Upload ===
uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx"])


if uploaded_file:
    full_text = extract_text_from_file(uploaded_file)
    st.success("✅ Text extracted successfully.")


    # Break into chunks
    chunks = list(chunk_text(full_text, max_words=450))
    st.info(f"✅ PDF loaded and split into {len(chunks)} chunks.")

    # Show progress and summarize
    summaries = []
    progress = st.progress(0)
    for i, chunk in enumerate(chunks):
        try:
            res = requests.post("http://127.0.0.1:8000/summarize/", json={"text": chunk})
            if res.status_code == 200:
                summary = res.json().get("summary", "")
                summaries.append(f"{summary}")
            else:
                summaries.append(f"❌ API Error {res.status_code}")
        except Exception as e:
            summaries.append(f"Request failed - {e}")
        progress.progress((i + 1) / len(chunks))

    # Display combined summary
    full_summary = "\n\n".join(summaries)
    st.subheader("🧠 Final Summary")
    st.text_area("Summary Output", full_summary, height=400)

    # Optional: Download button
    st.download_button("📥 Download Summary", data=full_summary, file_name="summary.txt")

    # 🔊 Convert summary to audio
    if st.button("🔊 Listen to Summary"):
        try:
            tts = gTTS(text=full_summary, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format='audio/mp3')
                with open(fp.name, "rb") as audio_file:
                    st.download_button(label="📥 Download Summary Audio", data=audio_file, file_name="summary.mp3", mime="audio/mpeg")
        except Exception as e:
            st.error(f"❌ Failed to generate audio: {e}")
