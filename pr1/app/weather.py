"""Модуль інтеграції із зовнішнім API погоди.

Це єдине місце застосунку, яке знає про HTTP: адреси сервісів, параметри
запиту, коди відповіді й формат JSON. Веб-рівень (`app/main.py`) отримує
звідси готовий результат або зрозумілу помилку і нічого не знає про
`requests`.
"""

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 5


class WeatherError(Exception):
    """Помилка отримання погоди, зрозуміла веб-рівню."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


def find_city(name: str) -> tuple[float, float, str]:
    """Знайти координати міста за його назвою.

    Повертає кортеж (latitude, longitude, formatted_name).
    """
    if not name or not name.strip():
        raise WeatherError("Назва міста не може бути порожньою", status_code=400)

    try:
        response = requests.get(
            GEOCODING_URL,
            params={"name": name.strip(), "count": 1, "language": "uk"},
            timeout=TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results")
        if not results:
            raise WeatherError(f"Місто '{name}' не знайдено", status_code=404)

        city_data = results[0]
        return city_data["latitude"], city_data["longitude"], city_data.get("name", name)

    except requests.Timeout:
        raise WeatherError("Перевищено час очікування відповіді від сервісу геокодування", status_code=504)
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 500
        raise WeatherError(f"Помилка сервісу геокодування (код {status})", status_code=status)
    except requests.RequestException:
        raise WeatherError("Сервіс геокодування тимчасово недоступний", status_code=503)


def get_current_weather(city: str) -> dict:
    """Повернути поточну погоду в місті: температуру й швидкість вітру."""
    lat, lon, formatted_name = find_city(city)

    try:
        response = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current_weather": True
            },
            timeout=TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        current = data.get("current_weather")
        if not current:
            raise WeatherError("У відповіді погодного сервісу відсутні дані про поточну погоду", status_code=502)

        return {
            "city": formatted_name,
            "temperature": current.get("temperature"),
            "windspeed": current.get("windspeed")
        }

    except requests.Timeout:
        raise WeatherError("Перевищено час очікування відповіді від погодного сервера", status_code=504)
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 500
        raise WeatherError(f"Помилка погодного сервера (код {status})", status_code=status)
    except requests.RequestException:
        raise WeatherError("Погодний сервіс тимчасово недоступний", status_code=503)