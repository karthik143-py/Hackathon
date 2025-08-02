import streamlit as st
import fitz  # PyMuPDF
import requests

st.set_page_config(page_title="💬 Chat with PDF", layout="wide")

# === Session state ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

# === Custom CSS ===
st.markdown(
    """
    <style>
    .chat-container {
        max-height: 70vh;
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 80px;
    }
    .chat-bubble-user {
        background-color: #DCF8C6;
        color: black;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: right;
        align-self: flex-end;
        width: fit-content;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-bubble-assistant {
        background-color: #F1F0F0;
        color: black;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: left;
        align-self: flex-start;
        width: fit-content;
        max-width: 80%;
        margin-right: auto;
    }
    .fixed-input {
        position: sticky;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem;
        background-color: white;
        border-top: 1px solid #ccc;
        z-index: 9999;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# === Title and PDF Upload ===
st.title("💬 Chat with your PDF")
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

# === Chat History Display ===
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-bubble-user"><b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-assistant"><b>Gemini:</b> {msg["content"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# === Sticky Input Box ===
st.markdown('<div class="fixed-bottom-input">', unsafe_allow_html=True)

# Two columns: input and ask button
col1, col2 = st.columns([8, 1])

with col1:
    user_input = st.text_input("Type your question and hit Ask", key="input_box", label_visibility="collapsed")

with col2:
    if st.button("Ask"):
        if not st.session_state.pdf_text:
            st.warning("❗ Please upload a PDF first.")
        elif user_input.strip() == "":
            st.warning("❗ Please type a question.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Call FastAPI backend
            with st.spinner("🤖 Thinking..."):
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/conversational_qa/",
                        json={
                            "text": st.session_state.pdf_text,
                            "chat_history": st.session_state.chat_history,
                        },
                    )
                    if response.status_code == 200:
                        answer = response.json().get("answer", "No answer found.")
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {response.status_code}")
                except Exception as e:
                    st.error(f"🚫 Request failed: {e}")

st.markdown('</div>', unsafe_allow_html=True)
