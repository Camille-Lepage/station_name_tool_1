# geocoding.py

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import ssl
import certifi
import time # Needed for delay
import streamlit as st # Keep for st.progress, st.warning etc.
import re # Added for response parsing

# --- Geopy SSL Context Workaround (Sometimes needed) ---
# Comment out if not needed
try:
    # Ensure you have installed certifi: pip install certifi
    ctx = ssl.create_default_context(cafile=certifi.where())
    # Using Nominatim.default_ssl_context as in your code
    Nominatim.default_ssl_context = ctx
except Exception as e:
    # Use st.warning here as this file might be imported by app.py
    st.warning(f"Could not set custom SSL context for geopy: {e}")
# --------------------------------------------------------

# Initialize Nominatim geocoder with a user_agent (important!)
# Use a more descriptive user_agent in a real application
# Using the user_agent you had in your code snippet
geolocator = Nominatim(user_agent="my_geocoding_app_v2") # Replace/update user_agent
# Using the RateLimiter parameters you had in your code snippet
geocode = RateLimiter(geolocator.reverse, min_delay_seconds=0.001, max_retries=3, error_wait_seconds=1.0)

def reverse_geocode_dataframe(df, lat_col, lon_col, output_col_address='geocoded_address', output_col_display_name='geocoded_display_name', output_col_raw='geocoded_address_raw_data'):
    """
    Performs reverse geocoding for a DataFrame using lat/lon columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        lat_col (str): Name of the latitude column.
        lon_col (str): Name of the longitude column.
        output_col_address (str): Name for the new column storing formatted addresses.
        output_col_display_name (str): Name for the new column storing display_name field.
        output_col_raw (str): Name for the new column storing raw geocoding data.

    Returns:
        pd.DataFrame: DataFrame with added geocoded address columns.
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        st.error(f"Latitude column '{lat_col}' or Longitude column '{lon_col}' not found in the DataFrame.")
        return df

    # Work on a copy to avoid modifying the original DataFrame passed in
    df_processed = df.copy()

    # Ensure columns for results exist on the copy
    if output_col_address not in df_processed.columns:
        df_processed[output_col_address] = None
    if output_col_display_name not in df_processed.columns:
        df_processed[output_col_display_name] = None
    if output_col_raw not in df_processed.columns:
         df_processed[output_col_raw] = None

    # Ensure columns are object type to store various results (strings, dicts, None)
    df_processed[output_col_address] = df_processed[output_col_address].astype(object)
    df_processed[output_col_display_name] = df_processed[output_col_display_name].astype(object)
    df_processed[output_col_raw] = df_processed[output_col_raw].astype(object)

    # Keeping the start message as in your code
    st.write("Starting reverse geocoding...")
    # Initialize progress bar with text
    progress_bar = st.progress(0, text="Geocoding in progress...")
    total_rows = len(df_processed)

    # Progress callback function
    def update_progress(current, total):
        progress = current / total
        progress_bar.progress(progress, text=f"Geocoding: {current}/{total}")

    # Iterate over the copied DataFrame
    for index, row in df_processed.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]

        # Ensure lat/lon are not NaN or None before attempting geocoding
        if pd.notna(lat) and pd.notna(lon):
            try:
                location = geocode(f"{lat}, {lon}")
                if location:
                    df_processed.at[index, output_col_address] = location.address
                    if hasattr(location, 'raw') and location.raw and 'display_name' in location.raw:
                        df_processed.at[index, output_col_display_name] = location.raw['display_name']
                    else:
                        df_processed.at[index, output_col_display_name] = location.address
                    df_processed.at[index, output_col_raw] = location.raw if location.raw is not None else {}
                else:
                    df_processed.at[index, output_col_address] = "No address found"
                    df_processed.at[index, output_col_display_name] = "No address found"
                    df_processed.at[index, output_col_raw] = {}
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                df_processed.at[index, output_col_address] = f"Geocoding Error: {e}"
                df_processed.at[index, output_col_display_name] = f"Geocoding Error: {e}"
                df_processed.at[index, output_col_raw] = {"error": str(e), "lat": lat, "lon": lon}
                st.warning(f"Geocoding failed for row with index {index} ({lat}, {lon}): {e}.")
            except Exception as e:
                df_processed.at[index, output_col_address] = f"Unexpected Error: {e}"
                df_processed.at[index, output_col_display_name] = f"Unexpected Error: {e}"
                df_processed.at[index, output_col_raw] = {"error": str(e), "lat": lat, "lon": lon}
                st.error(f"An unexpected error occurred during geocoding row with index {index} ({lat}, {lon}): {e}")
        else:
            df_processed.at[index, output_col_address] = "Missing Lat/Lon"
            df_processed.at[index, output_col_display_name] = "Missing Lat/Lon"
            df_processed.at[index, output_col_raw] = {}

        # Update progress bar
        update_progress(index + 1, total_rows)

    progress_bar.progress(1.0, text="Geocoding completed!")  # Ensure it reaches 100% at the end
    st.write("Reverse geocoding finished.")
    return df_processed # Return the modified copy
