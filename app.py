from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
import json

app = FastAPI()

# Groq endpoint
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = os.getenv("OLLAMA_API_KEY")  # Set in Render environment variables

class QueryInput(BaseModel):
    query: str

@app.post("/extract")
async def extract_travel_details(data: QueryInput):
    if not API_KEY:
        return {"error": "❌ Missing API Key. Set OLLAMA_API_KEY in Render environment variables."}

    inp = data.query

    prompt = f"""
    Extract only 'from' and 'to' locations from the following query.
    
    Rules:
    - Output valid JSON only.
    - Fields required: from, to.
    - If a location is not found, set it as "".

    Query: {inp}
    """

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a strict JSON extractor. Always return only {\"from\":..., \"to\":...}."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }

    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload)

        if response.status_code == 200:
            data = response.json()
            try:
                answer = data["choices"][0]["message"]["content"].strip()
                parsed = json.loads(answer)  # Ensure valid JSON
            except Exception:
                parsed = {"from": "", "to": ""}
            return {"query": inp, "extracted": parsed}

        return {"error": f"❌ API Error {response.status_code}", "details": response.text}

    except Exception as e:
        return {"error": "❌ Could not connect to Groq API", "details": str(e)}
