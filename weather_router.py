from fastapi import APIRouter, HTTPException
import os
import requests

router = APIRouter()

@router.get("/weather")
async def get_weather(lat: float, lng: float):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        # Fallback if key missing, but strictly this is "Real" logic failing gracefully
        return {"temp": 20, "condition": "No API Key", "wind": 0}
        
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lng}&appid={api_key}&units=metric"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "temp": data["main"]["temp"],
                "wind": data["wind"]["speed"],
                "condition": data["weather"][0]["main"],
                "city": data.get("name", "Unknown")
            }
        else:
            return {"error": "Weather API Error"}
    except Exception as e:
        return {"error": str(e)}
