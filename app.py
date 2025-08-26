from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests

app = FastAPI()

# Change this to your Ollama server URL (local/ngrok/VM)
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

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
        "model": "mistral:latest",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)

        if response.status_code == 200:
            data = response.json()
            answer = data.get("response", "").strip()
            return {"query": inp, "extracted": answer}
        else:
            return {"error": f"❌ Ollama Error {response.status_code}", "details": response.text}

    except Exception as e:
        return {"error": "❌ Could not connect to Ollama server", "details": str(e)}
