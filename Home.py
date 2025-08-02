# home.py

import streamlit as st
import threading
import uvicorn
# from fastapi import FastAPI

import requests

# === Launch FastAPI backend in background ===


# === Page config ===
st.set_page_config(page_title="PDF Factory", layout="wide")

# === Hero Section ===
st.markdown("""
<style>
h1 {
    text-align: center;
    font-size: 3em;
    margin-bottom: 0;
}
h4 {
    text-align: center;
    font-weight: 400;
    color: #bbb;
}
.feature-card {
    background-color: #0e1117;
    border-radius: 1rem;
    padding: 1.5rem;
    color: white;
    box-shadow: 0px 0px 12px rgba(0, 255, 255, 0.1);
    transition: 0.3s ease;
}
.feature-card:hover {
    box-shadow: 0px 0px 20px rgba(0, 255, 255, 0.3);
    transform: scale(1.02);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚀 PDF Factory</h1>", unsafe_allow_html=True)
st.markdown("<h4>Your Intelligent PDF Assistant for Text, QA, Graphs & More</h4>", unsafe_allow_html=True)
st.markdown("---")



# === Feature Sections ===
st.subheader("🔧 Available Tools")
cols = st.columns(3)

with cols[0]:
    st.markdown("""
    <div class="feature-card">
        <h3>📄 PDF Summarizer</h3>
        <p>Extract and summarize long academic or business PDFs using AI.</p>
        <a href="/Summarization" target="_self">👉 Go to Summarizer</a>
    </div>
    """, unsafe_allow_html=True)

with cols[1]:
    st.markdown("""
    <div class="feature-card">
        <h3>💬 Chat with PDF</h3>
        <p>Ask questions conversationally about your uploaded document. Powered by Gemini/LLM.</p>
        <a href="/Chat_with_PDF" target="_self">👉 Start Chat</a>
    </div>
    """, unsafe_allow_html=True)

with cols[2]:
    st.markdown("""
    <div class="feature-card">
        <h3>🔑 Keyword Extractor</h3>
        <p>Extract key concepts and named entities intelligently from PDFs.</p>
        <a href="/Keywords" target="_self">👉 Extract Keywords</a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

cols2 = st.columns(3)

with cols2[0]:
    st.markdown("""
    <div class="feature-card">
        <h3>📚 Knowledge Graph</h3>
        <p>Visualize semantic relationships between concepts in your document.</p>
        <a href="/Knowledge_Graph" target="_self">👉 Generate Graph</a>
    </div>
    """, unsafe_allow_html=True)

with cols2[1]:
    st.markdown("""
    <div class="feature-card">
        <h3>🔍 Semantic Search</h3>
        <p>Ask natural queries and retrieve the most relevant passages.</p>
        <a href="/Semantic_search" target="_self">👉 Semantic Search</a>
    </div>
    """, unsafe_allow_html=True)

with cols2[2]:
    st.markdown("""
    <div class="feature-card">
        <h3>🔊 Text to Audio</h3>
        <p>Convert extracted or summarized PDF content into audio speech.</p>
        <a href="/Text_to_Audio" target="_self">👉 Hear Content</a>
    </div>
    """, unsafe_allow_html=True)

cols3 = st.columns(3)

with cols3[0]:
    st.markdown("""
    <div class="feature-card">
        <h3>🧭 Smart Outline</h3>
        <p>It gives you the outline of your document.</p>
        <a href="/get_Outline" target="_self">👉 Brief your Doc</a>
    </div>
    """, unsafe_allow_html=True)

with cols3[1]:
    st.markdown("""
    <div class="feature-card">
        <h3>📅 Personalized Study Scheduler</h3>
        <p>Gives you the best study plan to work with your document.</p>
        <a href="/Schedular" target="_self">👉 Get youe study plan</a>
    </div>
    """, unsafe_allow_html=True)

with cols3[2]:
    st.markdown("""
    <div class="feature-card">
        <h3>🧪 Interactive Q&A Trainer</h3>
        <p>Train your self with multipule choice based question answerings.</p>
        <a href="/Trainer" target="_self">👉 Take a test</a>
    </div>
    """, unsafe_allow_html=True)

# === Footer ===
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888;'>© 2025 PDF Factory | Built for IBM Hackathon 🚀</p>",
    unsafe_allow_html=True
)
# from main import app as fastapi_app
# def run_fastapi():
#     uvicorn.run(fastapi_app, host="127.0.0.1", port=8000)

# threading.Thread(target=run_fastapi, daemon=True).start()
# if "api_started" not in st.session_state:
#     def start_api():
#         uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="warning")

#     threading.Thread(target=start_api, daemon=True).start()
#     st.session_state.api_started = True
#     import requests
# try:
#     res = requests.get("http://127.0.0.1:8000/")
#     if res.status_code == 404:
#         st.success("✅ FastAPI backend is running!")
# except:
#     st.warning("⚠️ Backend not responding.")




