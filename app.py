from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

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
    - Fields: from, to, mode, time, emotion, miles, rating, via, inbetween.
    - "from": starting point (remove words like 'my location', 'pickup point').
    - "to": destination.
    - "mode": transport type if available (car, train, bus, flight, etc.).
    - "time": use format MM/dd/yyyy hh:mm:ss. Default: 03/26/2025 11:04:22 if not found.
    - "emotion": choose from [Surprised, Angry, Sad, Fearful, Happy, Love, Nervous, Mischievous,
      Silly, Dizzy, Confused, Injured, Dreamy, Neutral, Hungry, Speechless, Embarrassed,
      Scared, Uncomfortable, Relieved, Respectful, Quiet, Robotic, Pleading, Sleepy,
      Sly, Sick, Excited, Exhausted, Thoughtful, Flirty, Worried, None].
    - "miles": extract distance in miles if present (default "1").
    - "rating": extract rating if present, else "".
    - "via": extract intermediate stops (single).
    - "inbetween": extract list of all extra stops if available.

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
        "temperature": 0.2
    }

    try:
        response = requests.post(OLLAMA_URL, headers=headers, json=payload)

        if response.status_code == 200:
            data = response.json()
            # Safe parsing
            try:
                answer = data["choices"][0]["message"]["content"].strip()
            except Exception:
                answer = data
            return {"query": inp, "extracted": answer}

        return {"error": f"❌ API Error {response.status_code}", "details": response.text}

    except Exception as e:
        return {"error": "❌ Could not connect to Groq API", "details": str(e)}
