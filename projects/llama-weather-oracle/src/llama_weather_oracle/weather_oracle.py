import argparse
import os
import sys

import ollama
import requests


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

PERSONALITY = """
You are Llama Nimbus, a dramatic but useful weather oracle.
You answer with playful, slightly ominous humor, but you must be accurate.
Use only the weather data provided by the API result.
Do not invent alerts, forecasts, or details that are not in the data.
Keep the answer brief: 2 to 4 sentences.
"""


def geocode_city(city):
    response = requests.get(
        GEOCODING_URL,
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    results = data.get("results", [])
    if not results:
        raise ValueError(f"I could not find a location matching '{city}'.")

    result = results[0]
    return {
        "name": result["name"],
        "country": result.get("country", "Unknown country"),
        "latitude": result["latitude"],
        "longitude": result["longitude"],
        "timezone": result.get("timezone", "auto"),
    }


def get_current_weather(city, temperature_unit):
    location = geocode_city(city)
    response = requests.get(
        FORECAST_URL,
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "temperature_unit": temperature_unit,
            "wind_speed_unit": "mph",
            "timezone": location["timezone"],
        },
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    if os.getenv("ENV", "").lower() == "development":
        print(f"Weather API response: {data}", file=sys.stderr)

    return {
        "location": f"{location['name']}, {location['country']}",
        "current": data["current"],
        "units": data["current_units"],
    }


def ask_llama(weather, model):
    prompt = f"""
User asked for the current weather.

Weather API result:
{weather}

Give the user the current temperature, humidity, and wind speed.
"""

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": PERSONALITY.strip()},
            {"role": "user", "content": prompt.strip()},
        ],
    )
    return response["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="Ask local Llama for live weather.")
    parser.add_argument("city", help="City to check, for example: Miami")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name")
    parser.add_argument(
        "--unit",
        choices=["fahrenheit", "celsius"],
        default="fahrenheit",
        help="Temperature unit",
    )
    args = parser.parse_args()

    try:
        weather = get_current_weather(args.city, args.unit)
        print(ask_llama(weather, args.model))
    except requests.RequestException as exc:
        print(f"Weather API request failed: {exc}", file=sys.stderr)
        sys.exit(1)
    except ollama.ResponseError as exc:
        print(f"Ollama error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
