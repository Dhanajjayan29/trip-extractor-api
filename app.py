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
    Extract structured travel details from the following query.
    
    Rules:
    - Output valid JSON only.
    - Always include all fields: from, to, mode, time, emotion, miles, rating, via, inbetween.
    - If a field is not explicitly present, set it to "" (empty string), except "inbetween" which should be [].
    - Do not assume or guess values. Only extract what's mentioned.
    - Example output:
      {{
        "from": "Toronto",
        "to": "Montreal",
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
            {
                "role": "system",
                "content": "You are a strict JSON extractor. Always return {from,to,mode,time,emotion,miles,rating,via,inbetween}. Never add defaults or guess. Missing fields must be empty."
            },
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
                parsed = {
                    "from": "",
                    "to": "",
                    "mode": "",
                    "time": "",
                    "emotion": "",
                    "miles": "",
                    "rating": "",
                    "via": "",
                    "inbetween": []
                }
            return {"query": inp, "extracted": parsed}

        return {"error": f"❌ API Error {response.status_code}", "details": response.text}

    except Exception as e:
        return {"error": "❌ Could not connect to Groq API", "details": str(e)}
