# utils.py

import unicodedata
import re
from math import radians, cos, sin, asin, sqrt

def clean_text(text):
    """Cleans station names by removing accents, converting to lowercase,
    and removing non-alphanumeric characters."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # Keep spaces for now
    text = text.strip() # Remove leading/trailing spaces
    # Optional: Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees) using the Haversine formula.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c # Radius of earth in kilometers. Use 6371 for average.
    return km