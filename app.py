from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
import json

app = FastAPI()

# Groq endpoint (Ollama-compatible)
OLLAMA_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = os.getenv("OLLAMA_API_KEY")

class QueryInput(BaseModel):
    query: str

@app.post("/extract")
async def extract_travel_details(data: QueryInput):
    if not API_KEY:
        return {"error": "❌ Missing API Key. Set OLLAMA_API_KEY in Render environment variables."}

    inp = data.query

    prompt = f"""
    Extract travel details from the following user query.

    Rules:
    - Output only valid JSON (no extra text).
    - Fields must always exist: from, to, mode, time, emotion, miles, rating, via, inbetween.
    - If a field is **not explicitly mentioned** in the query, set it to "" (or [] for inbetween).
    - Do NOT infer or guess values.
    - Example:
      Query: "How far is Edmonton from Jasper by car"
      Output: {{
        "from": "Edmonton",
        "to": "Jasper",
        "mode": "car",
        "time": "",
        "emotion": "",
        "miles": "",
        "rating": "",
        "via": "",
        "inbetween": []
      }}

    Query: {inp}
    """

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a strict JSON travel details extractor."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    try:
        response = requests.post(OLLAMA_URL, headers=headers, json=payload)

        if response.status_code == 200:
            data = response.json()
            try:
                answer = data["choices"][0]["message"]["content"].strip()
                extracted = json.loads(answer)  # enforce JSON validity
            except Exception as e:
                return {"error": "❌ Failed to parse model output", "details": str(e), "raw": data}

            return {"query": inp, "extracted": extracted}

        return {"error": f"❌ API Error {response.status_code}", "details": response.text}

    except Exception as e:
        return {"error": "❌ Could not connect to Groq API", "details": str(e)}
