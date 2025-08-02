import streamlit as st
import fitz  # PyMuPDF
from gtts import gTTS
import tempfile

st.set_page_config(page_title="📢 Text-to-Audio", layout="wide")
st.title("📢 Convert PDF Text to Audio")

uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")

if uploaded_file:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    extracted_text = ""
    for page in doc:
        extracted_text += page.get_text()

    st.success("✅ Text extracted successfully.")
    st.text_area("📄 Extracted Text", extracted_text, height=200)

    if st.button("🔊 Convert to Audio"):
        with st.spinner("Generating audio..."):
            try:
                tts = gTTS(text=extracted_text, lang='en')
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    st.audio(fp.name, format='audio/mp3')
                    with open(fp.name, "rb") as audio_file:
                        st.download_button(label="📥 Download MP3", data=audio_file, file_name="pdf_audio.mp3", mime="audio/mpeg")
            except Exception as e:
                st.error(f"❌ Error generating audio: {e}")
