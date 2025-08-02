import streamlit as st
import requests
import streamlit.components.v1 as components

FASTAPI_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="📚 Knowledge Graph Generator", layout="wide")
st.title("📚 Knowledge Graph Generator")

input_text = st.text_area("Enter your text for graph generation:", height=300)

if st.button("Generate Knowledge Graph"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating graph..."):
            response = requests.post(f"{FASTAPI_URL}/graph/", json={"text": input_text})

            if response.status_code == 200:
                # Save and read HTML content
                with open("knowledge_graph.html", "wb") as f:
                    f.write(response.content)

                with open("knowledge_graph.html", "r", encoding="utf-8") as f:
                    html_content = f.read()

                st.success("✅ Graph generated successfully!")

                # Display graph inside Streamlit
                components.html(html_content, height=700, scrolling=True)

                # Optional download button
                with open("knowledge_graph.html", "rb") as fbin:
                    st.download_button(
                        label="📥 Download Knowledge Graph",
                        data=fbin,
                        file_name="knowledge_graph.html",
                        mime="text/html"
                    )
            else:
                st.error("Failed to generate graph. Please try again.")
