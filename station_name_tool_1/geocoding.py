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
    progress_bar = st.progress(0)
    total_rows = len(df_processed)

    # Iterate over the copied DataFrame
    for index, row in df_processed.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]

        # Ensure lat/lon are not NaN or None before attempting geocoding
        if pd.notna(lat) and pd.notna(lon):
            try:
                # Use .at for setting single values by label for efficiency
                location = geocode(f"{lat}, {lon}")
                if location:
                    df_processed.at[index, output_col_address] = location.address
                    # Store display_name field from raw data if available
                    if hasattr(location, 'raw') and location.raw and 'display_name' in location.raw:
                        df_processed.at[index, output_col_display_name] = location.raw['display_name']
                    else:
                        df_processed.at[index, output_col_display_name] = location.address
                    # Store raw data as a dictionary, handle cases where it might be None
                    df_processed.at[index, output_col_raw] = location.raw if location.raw is not None else {}
                else:
                     # Handle cases where geocoding returns no location
                     df_processed.at[index, output_col_address] = "No address found"
                     df_processed.at[index, output_col_display_name] = "No address found"
                     df_processed.at[index, output_col_raw] = {}

            except (GeocoderTimedOut, GeocoderServiceError) as e:
                # Use .at for setting single values by label
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
             # Handle rows with missing lat/lon explicitly
             df_processed.at[index, output_col_address] = "Missing Lat/Lon"
             df_processed.at[index, output_col_display_name] = "Missing Lat/Lon"
             df_processed.at[index, output_col_raw] = {}

        # Update progress bar - Keeping the calculation as in your code
        # Ensure index is compatible with get_loc if the index is not standard RangeIndex
        try:
            current_pos = df_processed.index.get_loc(index) + 1
        except KeyError:
            # Fallback if index is not found (shouldn't happen with iterrows)
            current_pos = index + 1 # Assuming index is numeric if get_loc fails
        progress = current_pos / total_rows
        progress_bar.progress(min(progress, 1.0)) # Ensure progress doesn't exceed 1.0


    progress_bar.empty() # Hide the progress bar when done
    # Keeping the finish message as in your code
    st.write("Reverse geocoding finished.")
    return df_processed # Return the modified copy
