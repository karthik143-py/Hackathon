# 📘 PDF Intelligence Copilot

A unified,AI assistant that helps students and professionals interact with documents more intelligently — summarize, extract keywords, chat, search semantically, generate knowledge graphs, listen to audio, and even create study plans or train using MCQs.

> 🧠 Built for the IBM Hackathon | Team: Team 72

---

## 🚀 Features

- 📄 **Text Extraction**: View full content of PDF or DOCX with copy feature
- 🧠 **Summarizer**: Chunk + summarize long documents using BART
- 🔑 **Keyword Extractor**: Extract top concepts using IBM Granite-3.3-2B LLM
- 💬 **Chat with PDF**: Contextual Q&A powered by Gemini
- 🔍 **Semantic Search**: Query and retrieve most relevant sentences
- 📚 **Knowledge Graph**: Generate semantic relationships between concepts
- 🔊 **Text to Audio**: Real-time voice output with download support
- 🧭 **Smart Outline**: Get a structured outline of the document using LLM
- 📅 **Personalized Study Plan**: Generate day-wise learning roadmap
- 🧪 **Interactive Trainer**: Auto-generate MCQs for concept practice

---

## 🧰 Tech Stack

| Layer        | Tools & Models                                                                 |
|--------------|----------------------------------------------------------------------------------|
| Backend      | **FastAPI**, LangChain, FAISS, Hugging Face Transformers                       |
| Frontend     | **Streamlit** (multi-page layout)                                              |
| Models       | `facebook/bart-large-cnn`, `granite-3.3-2b-instruct`, `gemini-1.5-flash`, `MiniLM` |
| Document I/O | `PyMuPDF`, `python-docx`, `gTTS`, `pyttsx3`, `PyVis`                            |

---

## 🛠️ API Endpoints

| Endpoint            | Function                            | Model Used                       |
|---------------------|-------------------------------------|----------------------------------|
| `/summarize/`       | Summarizes content                  | `facebook/bart-large-cnn`        |
| `/keywords/`        | Extracts keyphrases                 | `granite-3.3-2b-instruct`        |
| `/question/`        | Answers factual questions           | `gemini-1.5-flash`               |
| `/conversationalqa/`| Multi-turn contextual chat          | `gemini-1.5-flash`               |
| ` /Audio/`          | Pdf to audio convertor              | `gtts`                           |
| `/TextEXtractor/`   | Extract text from doc and pdf       | `fitz`                           |
| `/search/`          | Semantic similarity search          | `MiniLM, FAISS`                  |
| `/graph/`           | Knowledge graph generation          | `LangChain + PyVis`                |
| `/outline/`         | Document outline extraction         | `gemini-1.5-flash`               |
| `/trainer/`         | Generates MCQs from concepts        | `gemini-1.5-flash`               |
| `/schedule/`        | Personalized study roadmap          | `gemini-1.5-flash`               |

---

## 📁 Project Structure

```bash
📁 pdf-intelligence-copilot
├── main.py                  # FastAPI backend
├── home.py                  # Streamlit launcher
├── .env                     # Environment variables
├── requirements.txt
├── .gitignore
├── 📁 pages/                # Streamlit app pages
│   ├── Chat_with_pdf.py
│   ├── keywords.py
│   ├── knowledge_graph.py
│   ├── question_answering.py
│   ├── semantic_search.py
│   ├── summarization.py
│   ├── text_extraction.py
│   ├── text_to_audio.py
│   ├── document_outliner.py
│   ├── trainer.py
│   └── personalized_scheduler.py
```

---



## 📈 Performance Highlights

- Models like Granite are loaded **once globally** to save memory
- Uses **FAISS** for blazing-fast search
- Runs **fully offline** except Gemini API
- Modular backend + clean API layer
- Optimized for multi-threaded async inference
- Simple and minimal design

---


## 🧩 Future Improvements

- 🔐 Document-level security (login/auth)
- 🗃️ Multi-document workspace
- 🎙️ Voice input
- 🌍 Multi-language support
- 🧠 RAG-enhanced answers using vector DB
- ☁️ Watsonx optional cloud backend

---

## 🏁 Conclusion

**PDF Intelligence Copilot** is an end-to-end AI system for intelligent document interaction. It compresses hours of reading, question answering, and study planning into minutes — built using only open-source models and APIs.

> 🔍 Ideal for education, research, legal reading, or enterprise knowledge management.

---

## 📜 License

MIT License © 2025 – Built with ❤️ by Team 72 for IBM Hackathon
