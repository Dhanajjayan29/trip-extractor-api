from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import re
import logging
import urllib.parse
import requests
from datetime import date, timedelta, datetime as dt
from geopy.distance import geodesic
import os
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="AI Trip Planner API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MAX_FORECAST_DAYS = 5
AVAILABLE_INTERESTS = [
    "Nature", "History", "Foodie", "Adventure", "Arts & Culture",
    "Relaxation", "Shopping", "Family Fun", "Nightlife",
    "Romantic", "Spiritual", "Educational", "Accessible Travel Focus"
]

# API Configuration
CUSTOM_API_ENDPOINT = "https://5f25128a2ff2.ngrok-free.app/api/generate"
CUSTOM_MODEL_NAME = "gpt-oss:20b"
GOOGLE_API_KEY = "AIzaSyAORraX3Txse2MCNXea5UnxxCrqubrsRHI"
WEATHER_API_KEY = "76225259973dffdfb59be3b3ce30433d"

# Pydantic Models
class Travelers(BaseModel):
    adults: int = 1
    children: int = 0
    seniors: int = 0
    infants: int = 0
    paratransit: int = 0

class TripRequest(BaseModel):
    from_location: Optional[str] = None
    destination: str
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    travelers: Travelers
    interests: List[str]

class ChatRequest(BaseModel):
    message: str
    trip_context: Optional[Dict[str, Any]] = None

class TripResponse(BaseModel):
    trip_id: str
    status: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# In-memory storage for trip data (use database in production)
trip_storage = {}

# Core functions from your Streamlit app
def make_custom_api_request(prompt, stream=False, system_prompt=""):
    """Makes a request to the custom API endpoint."""
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "model": CUSTOM_MODEL_NAME,
        "prompt": prompt,
        "stream": stream
    }
    if system_prompt:
        data["system_prompt"] = system_prompt

    try:
        response = requests.post(CUSTOM_API_ENDPOINT, headers=headers, json=data, timeout=90)
        response.raise_for_status()
        
        try:
            response_json = response.json()
            # Remove extraneous keys
            for key in ['context', 'thinking', 'done', 'done_reason', 'total_duration',
                        'load_duration', 'prompt_eval_count', 'prompt_eval_duration',
                        'eval_count', 'eval_duration']:
                response_json.pop(key, None)
            return response_json
        except json.JSONDecodeError:
            logger.warning("Custom API returned non-JSON response.")
            return {"raw_response": response.text}
    except requests.exceptions.RequestException as e:
        logger.error(f"Custom API request failed: {e}")
        return {"error": f"Failed to connect to AI service: {e}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred during custom API request: {e}")
        return {"error": f"An unexpected error occurred: {str(e)}"}

def geocode_location(location, api_key):
    """Geocodes a location string to latitude and longitude."""
    if not location or not api_key:
        return None, None

    if location.lower() == "current location":
        return None, None

    encoded_location = urllib.parse.quote(location)
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={encoded_location}&key={api_key}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "OK" and data.get("results"):
            lat = data["results"][0]["geometry"]["location"]["lat"]
            lon = data["results"][0]["geometry"]["location"]["lng"]
            return lat, lon
        else:
            return None, None
    except Exception as e:
        logger.error(f"Geocoding failed for '{location}': {e}")
        return None, None

def get_distance_and_time(from_coords, to_coords):
    """Calculate distance and estimated travel time."""
    if not from_coords or not to_coords or from_coords[0] is None or from_coords[1] is None or to_coords[0] is None or to_coords[1] is None:
            return 0, 0, "unknown"
    
    try:
        distance_km = geodesic((from_coords[0], from_coords[1]), (to_coords[0], to_coords[1])).km
        
        if distance_km < 5:
            mode = "walk"
            time_estimate = distance_km * 12
        elif distance_km < 150:
            mode = "drive" 
            time_estimate = distance_km * 1.5
        else:
            mode = "flight"
            time_estimate = distance_km * 0.1 + 150

        return distance_km, time_estimate, mode
    except Exception as e:
        logger.error(f"Error calculating distance/time: {e}")
        return 0, 0, "error"

def get_weather_forecast(destination, travel_date_obj):
    """Get weather forecast for destination."""
    try:
        geo_lat, geo_lon = geocode_location(destination, GOOGLE_API_KEY)
        if not geo_lat or not geo_lon:
            return {'temperature': 'N/A', 'description': 'Could not get forecast', 'humidity': 'N/A', 'wind_speed': 'N/A'}

        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': geo_lat,
            'lon': geo_lon,
            'appid': WEATHER_API_KEY,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        return {
            'temperature': round(data['main']['temp'], 1),
            'description': data['weather'][0]['description'].title(),
            'humidity': data['main']['humidity'],
            'wind_speed': round(data['wind']['speed'], 2)
        }
    except Exception as e:
        logger.error(f"Weather API request failed: {e}")
        return {'temperature': 'N/A', 'description': 'Could not get forecast', 'humidity': 'N/A', 'wind_speed': 'N/A'}

def get_packing_list(weather_data, travelers: Travelers):
    """Generate packing list based on weather and traveler types."""
    base_items = ["Passport/ID", "Phone charger", "Camera", "Medications"]
    
    try:
        temp_str = weather_data.get('temperature', '20')
        if isinstance(temp_str, str) and 'N/A' in temp_str: 
            temp = 20
        else: 
            temp = float(temp_str)
    except (ValueError, TypeError): 
        temp = 20

    clothing = []
    if temp < 10: 
        clothing.extend(["Warm jacket", "Long pants", "Warm shoes", "Gloves", "Hat"])
    elif temp < 20: 
        clothing.extend(["Light jacket", "Long pants", "Comfortable shoes"])
    else: 
        clothing.extend(["Light clothing", "Shorts", "Comfortable shoes", "Sunglasses", "Sunscreen"])
    
    description_lower = weather_data.get('description', '').lower()
    if any(word in description_lower for word in ['rain', 'shower', 'drizzle']):
        clothing.extend(["Umbrella", "Rain jacket"])
    
    if travelers.children > 0: 
        base_items.extend(["Snacks for kids", "Entertainment for kids", "Extra clothes"])
    if travelers.paratransit > 0: 
        base_items.extend(["Mobility aid", "Medical documents", "Emergency contacts"])
    
    return base_items + clothing

def get_place_details(place_name, destination, api_key):
    """Fetches place details from Google Places API."""
    search_query = f"{place_name}, {destination}"
    encoded_query = urllib.parse.quote(search_query)
    
    search_url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={encoded_query}&key={api_key}"
    
    place_id = None
    photo_url = None
    map_query = search_query
    google_name = place_name
    address = "Address unavailable"

    try:
        search_response = requests.get(search_url, timeout=10)
        search_response.raise_for_status()
        search_data = search_response.json()

        if search_data.get("status") == "OK" and search_data.get("results"):
            first_result = search_data["results"][0]
            place_id = first_result.get("place_id")
            google_name = first_result.get("name", place_name)
            address = first_result.get("formatted_address", "Address unavailable")
            map_query = f"{google_name}, {address}"

            if first_result.get("photos") and len(first_result["photos"]) > 0:
                photo_reference = first_result["photos"][0]["photo_reference"]
                photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={api_key}"
    except Exception as e:
        logger.error(f"Google Places Search failed for '{place_name}': {e}")

    return {
        "name": google_name,
        "photo_url": photo_url,
        "map_query": map_query,
        "address": address
    }

def extract_trip_details_from_prompt(user_prompt):
    """Extracts structured trip details from natural language prompt."""
    system_prompt = f"""
    You are an expert travel information extraction system. Parse the user's text and return a valid JSON object.
    The current date is {date.today().strftime('%Y-%m-%d')}.
    The JSON object MUST have: "origin", "destination", "start_date", "end_date", "travelers" (a dict with "adults", "children", "seniors", "infants", "paratransit"), and "interests".
    Use null for any value that is not present in the user's prompt.
    Respond ONLY with the JSON object. Do NOT add any preamble or closing text.
    """
    
    response_data = make_custom_api_request(prompt=user_prompt, system_prompt=system_prompt)

    if response_data and "error" not in response_data:
        try:
            details = None
            if isinstance(response_data, dict):
                if "raw_response" in response_data:
                    return None, f"AI Service returned non-JSON text: {response_data['raw_response']}"
                elif "response" in response_data and isinstance(response_data["response"], str):
                    json_match = re.search(r'```json\n(.*?)\n```', response_data["response"], re.DOTALL)
                    if json_match:
                        details = json.loads(json_match.group(1))
                    else:
                        details = json.loads(response_data["response"])
                elif "response" in response_data and isinstance(response_data["response"], dict):
                    details = response_data["response"]
                else:
                    details = response_data
            
            if details is None:
                return None, "Received an empty or invalid structure from AI."

            # Validate and clean details
            required_keys = ["origin", "destination", "start_date", "end_date", "travelers", "interests"]
            for key in required_keys:
                if key not in details or details[key] is None:
                    details[key] = None
            
            if not isinstance(details.get("travelers"), dict):
                details["travelers"] = {}
            
            for traveler_type in ["adults", "children", "seniors", "infants", "paratransit"]:
                if traveler_type not in details["travelers"]:
                    details["travelers"][traveler_type] = 0

            # Set default dates if not provided
            current_date = date.today()
            if details.get("start_date") is None:
                details["start_date"] = current_date.strftime('%Y-%m-%d')
            if details.get("end_date") is None:
                duration_match = re.search(r'(\d+)\s*(?:days?|day|weekend)', user_prompt, re.IGNORECASE)
                if duration_match:
                    try:
                        duration_days = int(duration_match.group(1))
                        end_date_calc = dt.strptime(details["start_date"], '%Y-%m-%d').date() + timedelta(days=duration_days - 1)
                        details["end_date"] = end_date_calc.strftime('%Y-%m-%d')
                    except (ValueError, TypeError): 
                        pass
                if details.get("end_date") is None:
                    details["end_date"] = (current_date + timedelta(days=2)).strftime('%Y-%m-%d')

            if not details.get("destination") or details["destination"].lower() == 'null':
                return None, "Could not identify destination. Please specify it clearly."
            
            return details, None
            
        except json.JSONDecodeError:
            return None, "AI response for trip details was not valid JSON."
        except Exception as e:
            return None, f"An error occurred processing AI details: {str(e)}"
            
    elif response_data and "error" in response_data:
        return None, f"AI Service Error: {response_data['error']}"
    else:
        return None, "Failed to get a valid response from the AI service."

def generate_itinerary(trip_data):
    """Generates itinerary using AI and enriches with Google Places details."""
    destination = trip_data.get('destination', 'a city')
    days = trip_data.get('days', 3)
    travelers = trip_data.get('travelers', Travelers())
    interests = trip_data.get('interests', ['general'])
    origin = trip_data.get('from_location', 'current location')
    
    try:
        start_date_obj = dt.strptime(trip_data['start_date'], '%Y-%m-%d').date()
    except (ValueError, TypeError):
        start_date_obj = date.today()
    
    start_date_str = start_date_obj.strftime('%Y-%m-%d')
    end_date_str = trip_data.get('end_date', (start_date_obj + timedelta(days=days-1)).strftime('%Y-%m-%d'))

    travel_context = f"trip TO {destination} from {origin} from {start_date_str} to {end_date_str}."
    traveler_context = f"{travelers.adults} adult(s)"
    if travelers.children > 0: 
        traveler_context += f", {travelers.children} child(ren)"
    if travelers.paratransit > 0: 
        traveler_context += f", {travelers.paratransit} paratransit traveler(s)"
    
    interest_context = f"interests: {', '.join(interests)}."
    
    prompt = f"""
    Generate a detailed day-by-day travel itinerary for {days} days in {destination}.
    {travel_context}
    Profile: {traveler_context}. {interest_context}

    **Instructions:**
    1. Suggest specific, real-world places relevant to the interests and destination.
    2. Plan 2-3 activities per day with reasonable pacing.
    3. For each place, provide a 1-2 sentence description.
    4. Explicitly state accessibility status for each place.
    5. Output STRICTLY as a JSON object with this structure:
    {{
        "transportation": "Overall transportation suggestion...",
        "itinerary": [
            {{
                "day": "Day X (YYYY-MM-DD)",
                "places": [
                    {{
                        "name": "Specific Place Name",
                        "description": "Detailed description...",
                        "suitable_for": ["adults", "children", ...],
                        "accessibility": "wheelchair accessible / partially accessible / not wheelchair accessible"
                    }}
                ]
            }}
        ]
    }}
    **DO NOT output any text outside the JSON structure.**
    """
    
    response_data = make_custom_api_request(prompt=prompt, 
                                          system_prompt="You are an expert travel planner generating specific, dynamic itineraries.")

    if response_data and "error" not in response_data:
        try:
            itinerary_result_from_ai = None
            
            if "raw_response" in response_data:
                return {"raw_response": response_data["raw_response"]}, False

            if isinstance(response_data, dict):
                if "response" in response_data and isinstance(response_data["response"], str):
                    json_match = re.search(r'```json\n(.*?)\n```', response_data["response"], re.DOTALL)
                    if json_match:
                        itinerary_result_from_ai = json.loads(json_match.group(1))
                    else:
                        itinerary_result_from_ai = json.loads(response_data["response"])
                elif "response" in response_data and isinstance(response_data["response"], dict):
                    itinerary_result_from_ai = response_data["response"]
                elif "itinerary" in response_data:
                    itinerary_result_from_ai = response_data
                else:
                    return "AI response structure not recognized.", False
            
            if itinerary_result_from_ai is None or "itinerary" not in itinerary_result_from_ai:
                return "AI did not return a valid itinerary.", False
            
            # Enrich with Google Places
            enriched_itinerary = []
            for day_entry in itinerary_result_from_ai.get("itinerary", []):
                if not isinstance(day_entry, dict) or "places" not in day_entry:
                    continue

                enriched_places = []
                for place_data in day_entry.get("places", []):
                    if not isinstance(place_data, dict) or "name" not in place_data:
                        continue
                    
                    place_details_google = get_place_details(place_data["name"], destination, GOOGLE_API_KEY)

                    enriched_places.append({
                        "name": place_details_google["name"],
                        "description": place_data.get("description", "No description available."),
                        "suitable_for": place_data.get("suitable_for", ["unknown"]),
                        "accessibility": place_data.get("accessibility", "information not available"),
                        "photo_url": place_details_google["photo_url"],
                        "map_query": place_details_google["map_query"],
                        "address": place_details_google.get("address", "Address unavailable")
                    })
                
                enriched_itinerary.append({
                    "day": day_entry.get("day", f"Day {len(enriched_itinerary) + 1}"),
                    "places": enriched_places
                })
            
            return {
                "itinerary": enriched_itinerary,
                "transportation": itinerary_result_from_ai.get("transportation", "N/A")
            }, True

        except json.JSONDecodeError:
            return "AI response for itinerary was not valid JSON.", False
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}", False
            
    elif response_data and "error" in response_data:
        return f"AI Service Error: {response_data['error']}", False
    else:
        return "Received unexpected response from AI service.", False

def create_pdf(trip_data, filename="trip_plan.pdf"):
    """Generates a PDF of the trip plan."""
    try:
        doc = SimpleDocTemplate(filename, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=72)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = styles['h1']
        title_style.alignment = TA_CENTER
        story.append(Paragraph("Your Trip Plan", title_style))
        story.append(Spacer(1, 0.2 * inch))

        # Add content similar to your Streamlit PDF function
        # ... (include the PDF content generation logic from your Streamlit app)
        
        doc.build(story)
        return True
    except Exception as e:
        logger.error(f"Error building PDF: {e}")
        return False

# API Endpoints
@app.get("/")
async def root():
    return {"message": "AI Trip Planner API", "status": "running"}

@app.get("/interests")
async def get_available_interests():
    return {"interests": AVAILABLE_INTERESTS}

@app.post("/plan-trip", response_model=TripResponse)
async def plan_trip(trip_request: TripRequest):
    """Plan a trip using form-based input."""
    try:
        trip_id = str(uuid.uuid4())
        
        # Calculate duration
        start_date = dt.strptime(trip_request.start_date, '%Y-%m-%d').date()
        end_date = dt.strptime(trip_request.end_date, '%Y-%m-%d').date()
        days = (end_date - start_date).days + 1
        
        # Prepare trip data
        trip_data = {
            'from_location': trip_request.from_location,
            'destination': trip_request.destination,
            'start_date': trip_request.start_date,
            'end_date': trip_request.end_date,
            'days': days,
            'travelers': trip_request.travelers,
            'interests': trip_request.interests
        }
        
        # Get coordinates and distance
        from_coords = geocode_location(trip_data['from_location'], GOOGLE_API_KEY) if trip_data['from_location'] else None
        to_coords = geocode_location(trip_data['destination'], GOOGLE_API_KEY)
        
        if from_coords and to_coords:
            distance, travel_time, mode = get_distance_and_time(from_coords, to_coords)
            trip_data.update({
                'distance': distance,
                'travel_time': travel_time,
                'recommended_mode': mode
            })
        
        # Get weather
        weather = get_weather_forecast(trip_data['destination'], start_date)
        trip_data['weather'] = weather
        
        # Get packing list
        packing_list = get_packing_list(weather, trip_request.travelers)
        trip_data['packing_list'] = packing_list
        
        # Generate itinerary
        itinerary_result, success = generate_itinerary(trip_data)
        
        if not success:
            return TripResponse(
                trip_id=trip_id,
                status="error",
                error=itinerary_result
            )
        
        trip_data['itinerary'] = itinerary_result.get('itinerary', [])
        trip_data['transportation'] = itinerary_result.get('transportation', 'N/A')
        
        # Store trip data
        trip_storage[trip_id] = trip_data
        
        return TripResponse(
            trip_id=trip_id,
            status="success",
            data=trip_data
        )
        
    except Exception as e:
        logger.error(f"Error planning trip: {e}")
        return TripResponse(
            trip_id=str(uuid.uuid4()),
            status="error",
            error=f"Internal server error: {str(e)}"
        )

@app.post("/chat-plan", response_model=TripResponse)
async def chat_plan_trip(chat_request: ChatRequest):
    """Plan a trip using natural language chat input."""
    try:
        trip_id = str(uuid.uuid4())
        
        # Extract trip details from chat message
        extracted_details, error_msg = extract_trip_details_from_prompt(chat_request.message)
        
        if error_msg:
            return TripResponse(
                trip_id=trip_id,
                status="error",
                error=error_msg
            )
        
        if not extracted_details:
            return TripResponse(
                trip_id=trip_id,
                status="error",
                error="Could not extract trip details from message"
            )
        
        # Convert to Travelers object
        travelers_dict = extracted_details.get('travelers', {})
        travelers = Travelers(
            adults=travelers_dict.get('adults', 1),
            children=travelers_dict.get('children', 0),
            seniors=travelers_dict.get('seniors', 0),
            infants=travelers_dict.get('infants', 0),
            paratransit=travelers_dict.get('paratransit', 0)
        )
        
        # Calculate duration
        start_date = dt.strptime(extracted_details['start_date'], '%Y-%m-%d').date()
        end_date = dt.strptime(extracted_details['end_date'], '%Y-%m-%d').date()
        days = (end_date - start_date).days + 1
        
        # Prepare trip data
        trip_data = {
            'from_location': extracted_details.get('origin'),
            'destination': extracted_details['destination'],
            'start_date': extracted_details['start_date'],
            'end_date': extracted_details['end_date'],
            'days': days,
            'travelers': travelers,
            'interests': extracted_details.get('interests', ['general'])
        }
        
        # Get coordinates and distance
        from_coords = geocode_location(trip_data['from_location'], GOOGLE_API_KEY) if trip_data['from_location'] else None
        to_coords = geocode_location(trip_data['destination'], GOOGLE_API_KEY)
        
        if from_coords and to_coords:
            distance, travel_time, mode = get_distance_and_time(from_coords, to_coords)
            trip_data.update({
                'distance': distance,
                'travel_time': travel_time,
                'recommended_mode': mode
            })
        
        # Get weather
        weather = get_weather_forecast(trip_data['destination'], start_date)
        trip_data['weather'] = weather
        
        # Get packing list
        packing_list = get_packing_list(weather, travelers)
        trip_data['packing_list'] = packing_list
        
        # Generate itinerary
        itinerary_result, success = generate_itinerary(trip_data)
        
        if not success:
            return TripResponse(
                trip_id=trip_id,
                status="error",
                error=itinerary_result
            )
        
        trip_data['itinerary'] = itinerary_result.get('itinerary', [])
        trip_data['transportation'] = itinerary_result.get('transportation', 'N/A')
        
        # Store trip data
        trip_storage[trip_id] = trip_data
        
        return TripResponse(
            trip_id=trip_id,
            status="success",
            data=trip_data
        )
        
    except Exception as e:
        logger.error(f"Error in chat plan: {e}")
        return TripResponse(
            trip_id=str(uuid.uuid4()),
            status="error",
            error=f"Internal server error: {str(e)}"
        )

@app.post("/trip-chat")
async def trip_chat(chat_request: ChatRequest):
    """Chat about an existing trip."""
    try:
        if not chat_request.trip_context:
            return {"error": "Trip context is required for trip chat"}
        
        trip_context_summary = f"Current trip plan: {json.dumps(chat_request.trip_context, indent=2)}"
        full_prompt = f"{trip_context_summary}\n\nUser query: {chat_request.message}"
        
        response_data = make_custom_api_request(
            prompt=full_prompt,
            system_prompt="You are a helpful AI assistant that answers follow-up questions about a specific trip plan. Use the provided trip context to answer. Be concise and helpful."
        )
        
        if response_data and "error" not in response_data:
            response_text = ""
            if "raw_response" in response_data:
                response_text = response_data["raw_response"]
            elif "response" in response_data and isinstance(response_data["response"], str):
                response_text = response_data["response"]
            else:
                response_text = "Received an unexpected AI response format."
            
            return {"response": response_text}
        else:
            return {"error": response_data.get("error", "Failed to get AI response")}
            
    except Exception as e:
        return {"error": f"Chat error: {str(e)}"}

@app.get("/trip/{trip_id}")
async def get_trip(trip_id: str):
    """Retrieve a previously generated trip plan."""
    if trip_id not in trip_storage:
        raise HTTPException(status_code=404, detail="Trip not found")
    
    return {
        "trip_id": trip_id,
        "data": trip_storage[trip_id]
    }

@app.get("/download-pdf/{trip_id}")
async def download_pdf(trip_id: str, background_tasks: BackgroundTasks):
    """Download trip plan as PDF."""
    if trip_id not in trip_storage:
        raise HTTPException(status_code=404, detail="Trip not found")
    
    trip_data = trip_storage[trip_id]
    filename = f"trip_plan_{trip_id}.pdf"
    
    if create_pdf(trip_data, filename):
        background_tasks.add_task(lambda: os.remove(filename) if os.path.exists(filename) else None)
        return FileResponse(filename, media_type='application/pdf', filename=filename)
    else:
        raise HTTPException(status_code=500, detail="Failed to generate PDF")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)