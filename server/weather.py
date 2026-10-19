"""Weather for the get_weather tool, from Open-Meteo (free, no key).

Much faster and more exact than a web search: one lookup for the place (cached)
and one for the forecast, usually well under a second.

    python weather.py Gandhinagar      check it works from this laptop
"""

from __future__ import annotations

import threading

import requests

import config

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather codes, in words Friday can say.
_CODES = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "foggy",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "heavy showers", 85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail",
}

_session = requests.Session()
_places: dict[str, dict] = {}
_lock = threading.Lock()


def _place(name: str) -> dict:
    key = name.strip().lower()
    with _lock:
        if key in _places:
            return _places[key]
    r = _session.get(GEO_URL, params={"name": name.split(",")[0].strip(), "count": 5, "language": "en",
                                      "format": "json"}, timeout=config.SEARCH_TIMEOUT)
    r.raise_for_status()
    results = (r.json() or {}).get("results") or []
    if not results:
        raise LookupError(f"no place called {name}")
    # "Gandhinagar, Gujarat" -> prefer the result in that state/country; else the biggest place.
    hint = ",".join(name.split(",")[1:]).strip().lower()
    best = None
    if hint:
        best = next((p for p in results if hint in f"{p.get('admin1', '')} {p.get('country', '')}".lower()), None)
    best = best or max(results, key=lambda p: p.get("population") or 0)
    with _lock:
        _places[key] = best
    return best


def get_weather(place: str) -> dict:
    """Now, today and tomorrow for `place`."""
    p = _place(place)
    r = _session.get(FORECAST_URL, timeout=config.SEARCH_TIMEOUT, params={
        "latitude": p["latitude"], "longitude": p["longitude"], "timezone": "auto", "forecast_days": 3,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,precipitation",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,sunrise,sunset",
    })
    r.raise_for_status()
    d = r.json()
    cur, day = d.get("current", {}), d.get("daily", {})

    def daily(i: int) -> dict:
        return {"sky": _CODES.get((day.get("weather_code") or [None] * 3)[i], "unknown"),
                "high_c": (day.get("temperature_2m_max") or [None] * 3)[i],
                "low_c": (day.get("temperature_2m_min") or [None] * 3)[i],
                "rain_chance_percent": (day.get("precipitation_probability_max") or [None] * 3)[i],
                "sunrise": ((day.get("sunrise") or [""] * 3)[i] or "")[-5:],
                "sunset": ((day.get("sunset") or [""] * 3)[i] or "")[-5:]}

    where = ", ".join(x for x in (p.get("name"), p.get("admin1"), p.get("country")) if x)
    return {
        "place": where,
        "now": {"sky": _CODES.get(cur.get("weather_code"), "unknown"), "temp_c": cur.get("temperature_2m"),
                "feels_like_c": cur.get("apparent_temperature"), "humidity_percent": cur.get("relative_humidity_2m"),
                "wind_kmh": cur.get("wind_speed_10m"), "rain_mm": cur.get("precipitation"),
                "local_time": (cur.get("time") or "").replace("T", " ")},
        "today": daily(0),
        "tomorrow": daily(1),
        "source": "Open-Meteo",
    }


if __name__ == "__main__":
    import json
    import sys
    import time

    t0 = time.time()
    print(json.dumps(get_weather(" ".join(sys.argv[1:]) or "Gandhinagar"), indent=2))
    print(f"{(time.time() - t0) * 1000:.0f} ms")
