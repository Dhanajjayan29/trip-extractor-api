from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json

app = FastAPI()

# Change this to your Ollama server URL
# On Render, you cannot access localhost:11434 (unless Ollama runs inside same service).
# For now, keep local testing:
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


# ---------------------------
# Root route (health check)
# ---------------------------
@app.get("/")
async def root():
    return {"message": "🚀 Trip Extractor API with Ollama is running"}


# ---------------------------
# Request model
# ---------------------------
class QueryInput(BaseModel):
    query: str


# ---------------------------
# /extract route
# ---------------------------
@app.post("/extract")
async def extract_travel_details(data: QueryInput):
    inp = data.query

    # Prompt for Ollama
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
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)

        if response.status_code == 200:
            data = response.json()
            raw_answer = data.get("response", "").strip()

            # Try to extract clean JSON
            try:
                # Sometimes Ollama returns extra text → try json.loads
                extracted = json.loads(raw_answer)
            except json.JSONDecodeError:
                # Try to extract JSON substring
                try:
                    start = raw_answer.find("{")
                    end = raw_answer.rfind("}") + 1
                    extracted = json.loads(raw_answer[start:end])
                except Exception:
                    extracted = {"error": "Invalid JSON from Ollama", "raw": raw_answer}

            return {"query": inp, "extracted": extracted}

        else:
            return {"error": f"Ollama Error {response.status_code}", "details": response.text}

    except Exception as e:
        return {"error": "Could not connect to Ollama server", "details": str(e)}
