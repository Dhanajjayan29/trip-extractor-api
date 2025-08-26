from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

# Ollama API (Cloud)
OLLAMA_URL = "https://api.ollama.ai/v1/chat/completions"
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")  # set this in Render Environment

class QueryInput(BaseModel):
    query: str

@app.post("/extract")
async def extract_travel_details(data: QueryInput):
    inp = data.query

    prompt = f"""
    Extract travel details from the following user query.

    Rules:
    - Output only valid JSON (no extra text).
    - Fields: from, to, mode, time, emotion, miles, rating, via, inbetween.
    - "from": starting point.
    - "to": destination.
    - "mode": transport type.
    - "time": use format MM/dd/yyyy hh:mm:ss. Default: 03/26/2025 11:04:22 if not found.
    - "emotion": choose from [Surprised, Angry, Sad, Fearful, Happy, Love, Nervous, Mischievous,
      Silly, Dizzy, Confused, Injured, Dreamy, Neutral, Hungry, Speechless, Embarrassed,
      Scared, Uncomfortable, Relieved, Respectful, Quiet, Robotic, Pleading, Sleepy,
      Sly, Sick, Excited, Exhausted, Thoughtful, Flirty, Worried, None].
    - "miles": distance in miles if present (default "1").
    - "rating": rating if present else "".
    - "via": single intermediate stop.
    - "inbetween": list of extra stops if available.

    Query: {inp}
    """

    headers = {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral",  # or llama3
        "messages": [
            {"role": "system", "content": "You are a travel info extractor."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(OLLAMA_URL, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            return {"query": inp, "extracted": answer}
        else:
            return {"error": f"Ollama API error {response.status_code}", "details": response.text}

    except Exception as e:
        return {"error": "Could not connect to Ollama API", "details": str(e)}
