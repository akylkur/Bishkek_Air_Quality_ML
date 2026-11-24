def aqi_category(value):
    for upper, label in [(50, "Good"), (100, "Moderate"), (150, "Unhealthy for sensitive groups"),
                         (200, "Unhealthy"), (300, "Very unhealthy")]:
        if value <= upper:
            return label
    return "Hazardous"
