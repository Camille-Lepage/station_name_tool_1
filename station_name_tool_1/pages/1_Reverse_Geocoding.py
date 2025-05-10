# pages/1_Reverse_Geocoding.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import io # For download button
import re # Added for response parsing

# --- Imports from your modules ---
from utils import clean_text # Needed for address formatting logic
from geocoding import reverse_geocode_dataframe # The geocoding function
# Add imports for other utility functions used directly in this script if any
# -----------------------------------

# --- Session State Initialization (Needed in every script using state) ---
# Initialize state variables if they don't exist
# IMPORTANT: Copy the ENTIRE session state initialization block from Home.py here
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_geocoded' not in st.session_state:
    st.session_state.df_geocoded = None
if 'df_ai_named' not in st.session_state:
    st.session_state.df_ai_named = None
if 'df_clustered' not in st.session_state:
    st.session_state.df_clustered = None
if 'df_final' not in st.session_state:
    st.session_state.df_final = None
if 'clusters_for_review' not in st.session_state:
    st.session_state.clusters_for_review = {}
if 'potential_cluster_names' not in st.session_state:
    st.session_state.potential_cluster_names = {} # Store suggested names per cluster
if 'manual_cluster_names' not in st.session_state:
    st.session_state.manual_cluster_names = {} # Store user-edited names
if 'last_uploaded_filename' not in st.session_state:
    st.session_state.last_uploaded_filename = "data.csv"
if 'geocoding_params' not in st.session_state:
     st.session_state.geocoding_params = {
         'lat_col': None,
         'lon_col': None,
     }
if 'clustering_params' not in st.session_state:
    st.session_state.clustering_params = {
        'name_col': None,
        'lat_col': None,
        'lon_col': None,
        'distance_threshold_km': 0.1,
        'name_similarity_threshold': 0.8,
    }
if 'ai_params' not in st.session_state:
    st.session_state.ai_params = {
        'address_col': None,
        'name_col': None,
        'other_name_cols': [],
        'batch_size': 50,
    }
if 'selected_address_keys' not in st.session_state: # Also initialize address keys state
    st.session_state.selected_address_keys = ['city'] # Default keys
# --- End of Session State Initialization ---


# Set page configuration (Optional, but good practice)
st.set_page_config(page_title="Reverse Geocoding", layout="wide")


st.title("Reverse Geocoding")
st.markdown("""
Reverse geocode Latitude/Longitude to get addresses.

**Note:** This tool uses OpenStreetMap for reverse geocoding. It takes approximately 1 second per station, so processing 1000 stations will take around 17 minutes. If the address names are not suitable, consider using a map API (for example via Google Sheets). Documentation: [Google Sheets Map API Confluence](https://one2go.atlassian.net/wiki/x/CIBPu).
""")

# --- Geocoding Step ---
st.header("🗺️ Run Reverse Geocoding")
st.markdown("Select Lat/Lon columns and run the geocoding process.")

# Ensure the DataFrame is loaded into session state
if 'df' in st.session_state and st.session_state.df is not None:
    current_df = st.session_state.df
else:
    st.error("No data found! Please upload a dataset on the Home page first.")
    st.stop()

# Display Latitude and Longitude selection side by side
col1, col2 = st.columns(2)
with col1:
    lat_col_geocode = st.selectbox(
        "Select Latitude Column",
        list(current_df.columns),
        index=(
            list(current_df.columns).index('latitude')
            if 'latitude' in current_df.columns
            else next((i for i, col in enumerate(current_df.columns) if 'lat' in col.lower()), 0)
        ),
        key="lat_col_geocode_page1"
    )

with col2:
    lon_col_geocode = st.selectbox(
        "Select Longitude Column",
        list(current_df.columns),
        index=(
            list(current_df.columns).index('longitude')
            if 'longitude' in current_df.columns
            else next((i for i, col in enumerate(current_df.columns) if 'lng' in col.lower() or 'lng' in col.lower()), 0)
        ),
        key="lon_col_geocode_page1"
    )

st.session_state.geocoding_params['lat_col'] = lat_col_geocode
st.session_state.geocoding_params['lon_col'] = lon_col_geocode


if st.button("Run Reverse Geocoding", key="run_geocode_page1_button"):
    if lat_col_geocode and lon_col_geocode:
        # --- CALL THE IMPORTED GEOCODING FUNCTION ---
        st.session_state.df_geocoded = reverse_geocode_dataframe(current_df, lat_col_geocode, lon_col_geocode)
        if st.session_state.df_geocoded is not None:
            st.success("Geocoding complete! Review raw data below to format addresses.")
        # No preview shown immediately after running, will be shown in formatting section

    else:
        st.warning("Please select both Latitude and Longitude columns for geocoding.")


# --- Address Formatting Step ---
st.header("📝 Format Geocoded Addresses")
st.markdown("Review the raw geocoding data and select keys to format the main address column. It is recommended to select keys that correspond to the city, neighborhood, or district, and avoid keys like street, postcode, or country for better results.")

# This step is only relevant if geocoding was performed and data is in session state
if st.session_state.df_geocoded is not None:
    df_for_formatting = st.session_state.df_geocoded.copy()  # Work on a copy

    if 'geocoded_address_raw_data' in df_for_formatting.columns:
        st.write("Preview of raw geocoding data (dictionary structure):")

        # Collect all possible address keys from all rows
        all_address_keys = set()

        # Iterate through the geocoded data to find all address keys
        for idx, row in df_for_formatting.iterrows():
            raw_data = row['geocoded_address_raw_data']
            if isinstance(raw_data, dict) and 'address' in raw_data and isinstance(raw_data['address'], dict):
                all_address_keys.update(raw_data['address'].keys())

        # Find the first row with valid raw geocoding data to show as a sample
        sample_raw_data = None
        for idx, row in df_for_formatting.iterrows():
            raw_data = row['geocoded_address_raw_data']
            if isinstance(raw_data, dict) and 'address' in raw_data and isinstance(raw_data['address'], dict):
                sample_raw_data = raw_data['address']
                break  # Found a sample, exit loop

        if sample_raw_data:
            # Display the sample raw data with expander (closed by default)
            with st.expander("Sample keys available in raw data:", expanded=False):
                st.json(sample_raw_data)  # Display the dictionary structure

            # Convert address keys to list and sort for consistent display
            preferred_keys = ["suburb", "borough", "neighbourhood", "city", "town", "village", "hamlet"]

            # Prepare all keys in a logical order (preferred keys first, then others alphabetically)
            all_keys_list = []
            if all_address_keys:
                # Add preferred keys that exist in the data
                for key in preferred_keys:
                    if key in all_address_keys:
                        all_keys_list.append(key)

                # Add remaining keys sorted alphabetically
                remaining_keys = sorted([k for k in all_address_keys if k not in preferred_keys])
                all_keys_list.extend(remaining_keys)
            else:
                # Fallback to default preferred keys if no address keys found
                all_keys_list = preferred_keys
                st.warning("No address keys found in the raw data. Using default address key options.")

            # Create checkboxes for each key
            st.write("Select keys to include in the formatted address:")
            selected_keys = []

            # Create 3 columns to display checkboxes in a grid
            cols = st.columns(3)
            for i, key in enumerate(all_keys_list):
                col_idx = i % 3
                # Check if the key should be pre-checked (either in preferred keys or was previously selected)
                default_checked = key in preferred_keys or key in st.session_state.selected_address_keys
                if cols[col_idx].checkbox(key, value=default_checked, key=f"checkbox_{key}"):
                    selected_keys.append(key)

            # Update session state with current selection
            st.session_state.selected_address_keys = selected_keys

            if st.button("Apply Selected Address Keys", key="apply_keys_page1_button"):
                if selected_keys:
                    st.write("Applying selected keys to format addresses...")
                    progress_bar_format = st.progress(0)
                    total_rows_format = len(df_for_formatting)

                    # Create the formatted address based on selected keys
                    formatted_addresses = []
                    for index, row in df_for_formatting.iterrows():
                        raw_data = row['geocoded_address_raw_data']
                        formatted_address_dict = {}  # Dictionary to store key-value pairs

                        if isinstance(raw_data, dict) and 'address' in raw_data and isinstance(raw_data['address'], dict):
                            for key in selected_keys:
                                if key in raw_data['address'] and pd.notna(raw_data['address'][key]):
                                    formatted_address_dict[key] = raw_data['address'][key].strip()
                        else:
                            # If 'address' is not in raw_data or not a dictionary, use the original formatted address
                            if 'geocoded_address' in row and pd.notna(row['geocoded_address']):
                                formatted_address_dict["full_address"] = row['geocoded_address']

                        # Convert the dictionary to a JSON string
                        import json
                        formatted_address = json.dumps(formatted_address_dict, ensure_ascii=False)
                        formatted_addresses.append(formatted_address)

                        # Update progress bar
                        progress_bar_format.progress((index + 1) / total_rows_format)

                    # Assign the newly formatted addresses back to the column
                    df_for_formatting['geocoded_address'] = formatted_addresses

                    st.session_state.df_geocoded = df_for_formatting  # Update the session state with the formatted data

                    progress_bar_format.empty()
                    st.success("Addresses formatted based on selected keys.")

                    # Show real-time preview of the formatted data
                    st.subheader("Preview of Geocoded and Formatted Data")
                    st.dataframe(st.session_state.df_geocoded[['geocoded_address', 'geocoded_address_raw_data']].head(500))  # Show 500 rows

                else:
                    st.warning("Please select at least one key to format the address.")

        else:
            st.warning("No valid raw geocoding data found to display keys for formatting.")
            st.session_state.selected_address_keys = []  # Reset selected keys if no raw data

    else:
        st.warning("The 'geocoded_address_raw_data' column was not found after geocoding.")
        st.session_state.selected_address_keys = []  # Reset selected keys



    st.write("Download the geocoded results after completing this step to save your progress.")

    def create_download_button(df, filename_suffix, key):
        if df is not None:
            try:
                # Use BytesIO to handle the CSV data in memory
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue().encode('utf-8')

                download_filename = f"{st.session_state.last_uploaded_filename.split('.')[0]}_{filename_suffix}.csv"

                st.download_button(
                    label="Download Reverse Geocoding Address",
                    data=csv_bytes,
                    file_name=download_filename,
                    mime="text/csv",
                    key=key
                )
            except Exception as e:
                st.error(f"Error preparing download file for step {filename_suffix}: {e}")

    create_download_button(st.session_state.df_geocoded, "geocoded", "download_geocoded_page1")


else:
    st.info("Run the geocoding step above to process and format addresses.")
