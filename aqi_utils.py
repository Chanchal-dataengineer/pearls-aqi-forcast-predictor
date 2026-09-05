def get_aqi_category(aqi):
    """
    Convert normalized AQI value (1-5) into
    a human-readable project-specific category.
    """

    aqi = float(aqi)

    if aqi <= 1.5:
        return {
            "category": "Good",
            "level": "🟢",
            "message": "Air quality is generally good."
        }

    elif aqi <= 2.5:
        return {
            "category": "Moderate",
            "level": "🟡",
            "message": "Air quality is acceptable, but sensitive people should be cautious."
        }

    elif aqi <= 3.5:
        return {
            "category": "Unhealthy",
            "level": "🟠",
            "message": "Air quality may negatively affect sensitive individuals."
        }

    elif aqi <= 4.5:
        return {
            "category": "Poor",
            "level": "🔴",
            "message": "Air quality is poor. Consider reducing prolonged outdoor exposure."
        }

    else:
        return {
            "category": "Hazardous",
            "level": "☠️",
            "message": "Air quality is hazardous. Avoid unnecessary outdoor exposure."
        }


def get_aqi_alert(aqi):
    """
    Return a concise alert message for dashboard/API use.
    """

    result = get_aqi_category(aqi)

    return (
        f"{result['level']} {result['category']}: "
        f"{result['message']}"
    )