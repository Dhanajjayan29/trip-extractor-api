from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

# Replace with actual Ollama/Router API endpoint
OLLAMA_URL = "https://api.openrouter.ai/v1/chat/completions"
API_KEY = os.getenv("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")  # Set in Render Dashboard

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

    payload = {
        "model": "mistral:latest",  # change if needed
        "messages": [
            {"role": "system", "content": "You are a JSON extractor."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload
        )

        if response.status_code == 200:
            data = response.json()
            answer = data["choices"][0]["message"]["content"].strip()
            return {"query": inp, "extracted": answer}
        else:
            return {"error": f"Ollama API Error {response.status_code}", "details": response.text}

    except Exception as e:
        return {"error": "Could not connect to Ollama API", "details": str(e)}
