from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional,Literal
import torch
from datetime import datetime, timedelta
import os
from fastapi.responses import JSONResponse
import math
import json
import re


from dotenv import load_dotenv
# import numpy as np
# import faiss

# === Create FastAPI app ===
app = FastAPI()

# === Global model variables ===
tokenizer = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nlp = None
embedder = None
llm = None
graph_transformer = None
granite_tokenizer = None
granite_model = None

# === Input schemas ===
class TextInput(BaseModel):
    text: str
class TextInputs(BaseModel):
    text: str
    n: Optional[int] = 5
class ScheduleRequest(BaseModel):
    text: str
    mode: Literal["pdf", "concepts"]
    iq: int
    hours_per_day: float
    end_date: str


class QAInput(BaseModel):
    text: str
    question: str

class ChatRequest(BaseModel):
    text: str
    chat_history: List[Dict[str, str]]

class SearchInput(BaseModel):
    text: str
    query: str

# === /keywords/ (Granite 3.3 2B) ===
@app.post("/keywords/")
async def keywords(data: TextInput):
    

    prompt = (
        "Extract the top 5 keywords from the following academic text. "
        "Only return comma-separated keywords. Do not explain.\n\n"
        f"{data.text.strip()}\n\nKeywords:"
    )

    inputs = granite_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    outputs = granite_model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
        pad_token_id=granite_tokenizer.eos_token_id
    )
    result = granite_tokenizer.decode(outputs[0], skip_special_tokens=True)
    keywords_text = result.split("Keywords:")[-1].strip()
    keywords = [kw.strip().lower() for kw in keywords_text.split(",") if kw.strip()]

    return {"keywords": keywords}

# === /summarize/ ===
@app.post("/summarize/")
async def summarize(data: TextInput):
    input_text = data.text.strip().replace("\n", " ")
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(input_ids, max_length=200, min_length=30, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"summary": summary}

# === /question/ (Gemini direct question answering) ===
@app.post("/question/")
async def question(data: QAInput):
    # import google.generativeai as genai
    # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    # model = genai.GenerativeModel("models/gemini-1.5-flash")
    prompt = f"Text: {data.text}\n\nQuestion: {data.question}"
    response = modelgen.generate_content(prompt)
    return {"answers": [{"answer": response.text.strip(), "score": 1.0}]}

# === /conversational_qa/ ===
@app.post("/conversational_qa/")
async def conversational_qa(data: ChatRequest):
    
    prompt = f"You are a document assistant. Use this text:\n\n{data.text}\n\nAnd this chat history:\n"
    for msg in data.chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        prompt += f"{role}: {msg['content']}\n"
    prompt += "Assistant:"
    
    response = modelgen.generate_content(prompt)
    return {"answer": response.text.strip()}

# === /search/ ===
@app.post("/search/")
async def semantic_search(data: SearchInput):
    sentences = [s.strip() for s in data.text.split('.') if len(s.strip()) > 20]
    sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)
    query_embedding = embedder.encode(data.query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
    top_results = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)[:3]
    top_passages = [{"text": sent, "score": float(score)} for sent, score in top_results]
    return {"results": top_passages}

# === /graph/ ===
@app.post("/graph/")
async def graph(data: TextInput):
    graph_documents = await extract_graph_data(data.text)
    output_file = "knowledge_graph.html"
    visualize_graph(graph_documents, output_file)
    return FileResponse(output_file, media_type='text/html', filename="knowledge_graph.html")

# === Helpers for /graph/ ===
async def extract_graph_data(text):
    from langchain_core.documents import Document
    documents = [Document(page_content=text)]
    return await graph_transformer.aconvert_to_graph_documents(documents)

def visualize_graph(graph_documents, output_file):
    from pyvis.network import Network
    import random

    # Initialize Pyvis Network
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#0e1117",         # Dark background
        font_color="#f5f5f5",       # Light font
        directed=True,
        cdn_resources="remote"
    )

    nodes = graph_documents[0].nodes
    relationships = graph_documents[0].relationships

    node_dict = {node.id: node for node in nodes}
    valid_edges = []
    valid_node_ids = set()

    for rel in relationships:
        if rel.source.id in node_dict and rel.target.id in node_dict:
            valid_edges.append(rel)
            valid_node_ids.update([rel.source.id, rel.target.id])

    # Assign unique color per node type
    color_map = {}

    def get_color_for_type(type_):
        if type_ not in color_map:
            color_map[type_] = "#%06x" % random.randint(0x111111, 0xFFFFFF)
        return color_map[type_]

    # Add nodes to the network
    for node_id in valid_node_ids:
        node = node_dict[node_id]
        try:
            label = node.properties.get("name", node.id)
            title = f"ID: {node.id}<br>Type: {node.type}<br>Properties: {node.properties}"

            net.add_node(
                node.id,
                label=label,
                title=title,
                group=node.type,
                color={
                    "background": get_color_for_type(node.type),
                    "border": "#f39c12",
                    "highlight": {
                        "background": "#f39c12",
                        "border": "#e67e22"
                    }
                },
                font={"size": 16}
            )
        except Exception:
            continue

    # Add edges to the network
    for rel in valid_edges:
        try:
            net.add_edge(
                rel.source.id,
                rel.target.id,
                label=rel.type.lower(),
                arrows="to",
                font={"size": 15, "color": "#ff0000"},
                color="gray"
            )
        except Exception:
            continue

    # Set visual options
    net.set_options("""
    const options = {
      "layout": {
        "improvedLayout": true
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.01,
          "springLength": 120,
          "springConstant": 0.05
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      },
      "interaction": {
        "hover": true,
        "zoomView": true,
        "dragView": true,
        "navigationButtons": true
      },
      "nodes": {
        "shape": "dot",
        "scaling": {
          "min": 10,
          "max": 30
        }
      },
      "edges": {
        "smooth": true,
        "arrows": {
          "to": {
            "enabled": true
          }
        }
      }
    }
    """)

    # Save the output
    net.save_graph(output_file)
    # print(f"✅ Graph successfully saved to {output_file}")
@app.post("/outline/")
async def generate_outline(data: TextInput):
#     import google.generativeai as genai
    # genai.configure(api_key=api_key)
    # model = genai.GenerativeModel("models/gemini-1.5-flash")

    prompt = f"""
You are an intelligent academic assistant. Analyze the following text and extract a structured outline with multiple relevant sections like:

- Title
- Concepts
- Applications
- Key Terms
- Content (summarize in 1-2 words)
- Use Cases
- Challenges
- Advantages
- Limitations

🟡 Important Instructions:
- Only include sections that are **present and meaningful** in the text.
- If a section like "Applications" or "Use Cases" is not found in the text, **do not include** it in the output.
- Format your response like this:

Title: ...
Concepts: ...
Key Terms: ...
Use Cases: ...
Content: ...
(etc.)

Text to analyze:
\"\"\"
{data.text}
\"\"\"

Return only the final structured outline. Do not include any explanation or preamble.
"""


    response = modelgen.generate_content(prompt)
    raw = response.text.strip()
    outline_lines = [line.strip("•- \n") for line in raw.splitlines() if line.strip()]
    return {"outline": outline_lines}
@app.post("/quiz/")
async def generate_quiz(data: TextInputs):
    

    text = data.text.strip()
    num_q = data.n or 5


    prompt = f"""
You are an educational AI assistant.

From the following text, generate {num_q} multiple choice questions (MCQs).

Each MCQ must include:
- A clear question
- Four answer options labeled a, b, c, d
- Correct answer letter (e.g., "Answer: c")

Return only in this format (no explanation):

Q1. [Question]
a) Option A
b) Option B
c) Option C
d) Option D
Answer: b

Q2...

If content is too short for {num_q} MCQs, generate as many as possible.

Text:
\"\"\"
{text}
\"\"\"
"""
    response = modelgen.generate_content(prompt)
    return {"quiz": response.text.strip()}

@app.post("/schedule/")
async def schedule(data: ScheduleRequest):
    try:
        today = datetime.today().date()
        end = datetime.strptime(data.end_date, "%Y-%m-%d").date()
        total_days = (end - today).days

        if total_days <= 0:
            return {"error": "End date must be in the future."}

        prompt = f"""
You are a helpful assistant. Generate a day-wise study schedule.

INPUT:
Text:
\"\"\"{data.text}\"\"\"
Mode: {data.mode}
IQ: {data.iq}
Daily Hours: {data.hours_per_day}
End Date: {data.end_date} ({total_days} days)

Rules:
- Output must be a pure JSON array like:
[
  {{
    "day": 1,
    "concept": "Topic 1",
    "summary": "1-2 lines",
  }},
  ...
]
- Do NOT add explanations.
- Do NOT include markdown.
- Always respond in valid JSON.
        """

        # Gemini Call
        response = modelgen.generate_content(prompt)
        raw_output = response.text.strip()

        # 1. Handle empty response
        if not raw_output:
            return {"error": "Gemini returned an empty response."}

        # 2. Attempt to extract only JSON part (in case extra junk exists)
        json_start = raw_output.find("[")
        json_end = raw_output.rfind("]") + 1

        if json_start == -1 or json_end == -1:
            return {"error": "Gemini output does not contain a valid JSON array.", "raw": raw_output[:300]}

        cleaned_json = raw_output[json_start:json_end]

        # 3. Parse it
        try:
            plan = json.loads(cleaned_json)
            return {"plan": plan}
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse JSON: {e}", "raw": cleaned_json[:300]}

    except Exception as e:
        return {"error": str(e)}

# === Model initialization ===
if __name__ != "__main__":
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from langchain_google_genai import ChatGoogleGenerativeAI
    from sentence_transformers import SentenceTransformer, util
    import google.generativeai as genai
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM,BitsAndBytesConfig
    # import google.generativeai as genai
    # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    granite_model_id = "ibm-granite/granite-3.3-2b-instruct"
    granite_tokenizer = AutoTokenizer.from_pretrained(granite_model_id,device_map="auto",quantization_config=bnb_config)
    granite_model = AutoModelForCausalLM.from_pretrained(granite_model_id,torch_dtype=torch.float16).to(device)
    _ = granite_model.generate(granite_tokenizer("Hello", return_tensors="pt").to(device)["input_ids"])

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)

    # Summarizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(device)
    modelgen = genai.GenerativeModel("models/gemini-1.5-flash")
    # Embeddings for search
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Graph transformer
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=api_key)
    graph_transformer = LLMGraphTransformer(llm=llm)

    # Granite for keyword extraction
    

    
