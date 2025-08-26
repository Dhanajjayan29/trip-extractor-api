from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
import json

app = FastAPI()

# Groq API endpoint (OpenAI-compatible)
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = os.getenv("OLLAMA_API_KEY")

class QueryInput(BaseModel):
    query: str

@app.post("/process")
async def process_query(data: QueryInput):
    if not API_KEY:
        return {"error": "❌ Missing API Key. Set OLLAMA_API_KEY in environment variables."}

    inp = data.query

    # Categories for classification
    categories = [
        "accounting","airport","amusement_park","aquarium","art_gallery","atm","bakery","bank","bar",
        "beauty_salon","bicycle_store","book_store","bowling_alley","bus_station","cafe","campground",
        "car_dealer","car_rental","car_repair","car_wash","casino","cemetery","church","city_hall",
        "clothing_store","convenience_store","courthouse","dentist","department_store","doctor","drugstore",
        "electrician","electronics_store","embassy","fire_station","florist","funeral_home","furniture_store",
        "gas_station","gym","hair_care","hardware_store","hindu_temple","home_goods_store","hospital",
        "insurance_agency","jewelry_store","laundry","lawyer","library","light_rail_station","liquor_store",
        "local_government_office","locksmith","lodging","meal_delivery","meal_takeaway","mosque","movie_rental",
        "movie_theater","moving_company","museum","night_club","painter","park","parking","pet_store","pharmacy",
        "physiotherapist","plumber","police","post_office","primary_school","real_estate_agency","restaurant",
        "roofing_contractor","rv_park","school","secondary_school","shoe_store","shopping_mall","spa","stadium",
        "storage","store","subway_station","supermarket","synagogue","taxi_stand","tourist_attraction",
        "train_station","transit_station","travel_agency","university","veterinary_care","zoo"
    ]

    # Unified prompt
    prompt = f"""
    You are a strict JSON extractor and classifier.

    Task:
    1. Extract travel details (from, to, mode, time, emotion, miles, rating, via, inbetween).
    2. Classify the query into ONE category from the provided list: {", ".join(categories)}.
    3. Always return JSON only, with this structure:

    {{
      "from": "",
      "to": "",
      "mode": "",
      "time": "",
      "emotion": "",
      "miles": "",
      "rating": "",
      "via": "",
      "inbetween": [],
      "category": ""
    }}

    Rules:
    - If a field is NOT explicitly in the query, leave it as "" (or [] for inbetween).
    - Do not infer or add extra values.
    - For classification, choose the best matching category from the list, or "" if no match.

    Example:
    Query: "I am not feeling well, where should I go?"
    {{
      "from": "",
      "to": "",
      "mode": "",
      "time": "",
      "emotion": "",
      "miles": "",
      "rating": "",
      "via": "",
      "inbetween": [],
      "category": "hospital"
    }}

    Now process:
    Query: {inp}
    """

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a strict JSON travel extractor and classifier."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload)

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
