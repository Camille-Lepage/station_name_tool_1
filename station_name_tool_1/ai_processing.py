# ai_processing.py

import google.generativeai as genai
import json
import time
import streamlit as st # Keep for st.info, st.error etc.
from math import ceil # Needed for batch calculation
import pandas as pd
import re

# Assuming clean_text is needed here, import it from utils
from utils import clean_text

def configure_gemini(api_key):
    """Configures the Google Gemini API with the provided API key."""
    try:
        if not api_key:
            st.error("No Gemini API key provided. Please enter a valid API key.")
            return False
            
        genai.configure(api_key=api_key)
        st.info("Gemini API configured successfully.")
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}")
        st.error("Please check your API key.")
        return False

# Function to parse Gemini response (using bracket-finding approach)
def parse_gemini_response(response_text):
    """
    Parses the text response from Gemini, extracting JSON or key-value pairs.
    Returns the parsed JSON object or None if parsing fails.
    """
    try:
        # Attempt to find and parse a JSON array
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        if (json_start >= 0 and json_end > json_start):
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)

        # If no JSON array is found, attempt to parse key-value pairs
        parsed_data = []
        pattern = re.compile(r"oi (\d+) pn ([^\n]+)")  # Use shorter keys: oi = original_index, pn = proposed_name
        matches = pattern.findall(response_text)
        for match in matches:
            original_index, proposed_name = match
            parsed_data.append({
                "original_index": int(original_index),
                "proposed_name": proposed_name.strip()
            })
        if parsed_data:
            return parsed_data

        # If no valid data is found, log an error
        st.error("No valid data found in the Gemini response")
        st.text(f"Raw response: {response_text}")
        return None

    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON from Gemini response: {e}")
        st.text(f"Raw response: {response_text}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during response parsing: {e}")
        st.text(f"Raw response: {response_text}")
        return None

# New function to detect address format
def detect_address_format(sample):
    """
    Detect whether a sample address is structured (Nominatim) or plain text (Maps API).
    Returns 'nominatim' or 'maps'
    """
    if isinstance(sample, dict) or (isinstance(sample, str) and sample.startswith('{')):
        return 'nominatim'
    else:
        return 'maps'

# Function to create prompt based on format type
def create_prompt_for_format(batch_data, address_col, other_name_cols, prompt_type, custom_template=None):
    """
    Create an appropriate prompt based on the address format (Nominatim or Maps API) or use a custom template
    """
    if prompt_type == 'custom' and custom_template:
        # Check if the template contains the {batch_data} placeholder
        if '{batch_data}' in custom_template:
            # Format the batch data as nicely formatted JSON
            formatted_batch = json.dumps(batch_data, indent=2)
            return custom_template.replace('{batch_data}', formatted_batch)
        else:
            # If no placeholder is found, add the data at the end
            st.warning("No {batch_data} placeholder found in your custom prompt. Adding data at the end.")
            return custom_template + "\n\nHere is the data to process:\n" + json.dumps(batch_data, indent=2)
    elif prompt_type == 'maps':
        # Improved prompt for Maps API (plain text addresses)
        prompt = """
        Analyze the following structured address data and remote names to determine the best name (`pn`) for each transportation station.

        1. IMPORTANT : AVOID word duplication in the name (`pn`), If a word in the address key is the same as a word in the remote name, consider them similair, choose another adress key. So the name is not redundant. 
        2. Combine the remote_name with a relevant address component that is address key that is NOT similar
           and is contextually suitable (By priority: 1. town, 2. city, 3. village, 4. suburb, 5. neighbourhood), (eg. Center is a good adress value for the name).
           If no suitable address component is available, use only the remote_name
        3.  Remove all diacritics and accents from all parts of the name.
        4.  Ensure the final generated name (pn) does not exceed 10 words.
        5.  Don't keep station-related terms (e.g., "Terminal", "Gare", "Rodoviária", "Estación", ”station”)

        Here is the data to process:
        """

        # Add batch data to the prompt
        for i, item in enumerate(batch_data):
            address = item.get(address_col, '') if isinstance(item.get(address_col, ''), str) else ""
            remote_name = item.get('original_name', '') if isinstance(item.get('original_name', ''), str) else ""
            prompt += f"\nStation {i+1}:\n- A: \"{address}\"\n- R: \"{remote_name}\"\n"

        # Request a response in JSON format
        prompt += """
        Reply only with a JSON array where each element is an object with the following properties:
        - oi: The original index of the station in the input list (used for mapping back to the DataFrame).
        - pn: The single, clean, and standardized name you propose based on the address and other available names.

        For example:
        ```json
        [
          {
            "oi": 0,
            "pn": "Sao Paulo Terminal - Santana"
          },
          {
            "oi": 1,
            "pn": "New York - Main Street"
          }
        ]
        ```
        """
    else:  # 'nominatim' or other default
        # Prompt for Nominatim (structured addresses)
        prompt = f"""
        Analyze the following structured address data and remote names to determine the best name (`pn`) for each transportation station.

        1. IMPORTANT : AVOID word duplication in the name (`pn`), If a word in the address key is the same as a word in the remote name, consider them similair, choose another adress key. So the name is not redundant. 
        2. Combine the remote_name with a relevant address component that is address key that is NOT similar
           and is contextually suitable (By priority: 1. town, 2. city, 3. village, 4. suburb, 5. neighbourhood), (eg. Center is a good adress value for the name).
           If no suitable address component is available, use only the remote_name
        3.  Remove all diacritics and accents from all parts of the name.
        4.  Ensure the final generated name (pn) does not exceed 10 words.
        5.  Don't keep station-related terms (e.g., "Terminal", "Gare", "Rodoviária", "Estación", ”station”)

        Here is the data to process:
        {json.dumps(batch_data, indent=2)}

        Return the result as a JSON array of objects. Each object must have the following keys:
        - oi: The original index of the station in the input list (used for mapping back to the DataFrame).
        - pn: The single, clean, and standardized name you propose based on the address and other available names.

        Ensure the output is ONLY the JSON array. Do not include any other text before or after the JSON.
        ```json
        [
          {{"oi": 0, "pn": "Sao Paulo - Santana"}},
          {{"oi": 1, "pn": "New York - Main Street"}}
          // ... more objects
        ]
        ```
        """
    return prompt

# Function to process a batch of stations with Gemini
def process_batch_with_gemini(model, batch_data, address_col, other_name_cols, prompt_type='auto', max_retries=3, delay_seconds=1, custom_template=None):
    """Processes a batch of station data using the Gemini model."""
    
    # If prompt_type is 'auto', detect from first item in batch
    if prompt_type == 'auto' and batch_data:
        first_address = batch_data[0].get(address_col)
        detected_format = detect_address_format(first_address)
        if not hasattr(st.session_state, 'logged_prompt_type') or st.session_state.logged_prompt_type != detected_format:
            st.info(f"Auto-detected address format: {detected_format}")
            st.session_state.logged_prompt_type = detected_format  # Store detected format in session state
        prompt_type = detected_format
    
    # Create appropriate prompt based on detected or specified format
    prompt = create_prompt_for_format(batch_data, address_col, other_name_cols, prompt_type, custom_template)

    for attempt in range(max_retries):
        try:
            # Send request to Gemini
            response = model.generate_content(prompt)
            response_text = response.text

            # Find and extract the JSON
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                try:
                    results = json.loads(json_str)
                    # Process results based on their structure
                    processed_results = []
                    
                    for item in results:
                        # Handle both possible formats
                        if "oi" in item and "pn" in item:
                            processed_results.append({
                                "original_index": item["oi"],
                                "proposed_name": clean_text(item["pn"])
                            })
                        elif "station_name" in item:
                            # Handle old format for backward compatibility
                            idx = results.index(item)
                            processed_results.append({
                                "original_index": idx,
                                "proposed_name": clean_text(item["station_name"])
                            })
                        else:
                            # Handle format with direct index mapping
                            for key, value in item.items():
                                if isinstance(value, str):  # Assume this is the name
                                    try:
                                        idx = int(key) if key.isdigit() else results.index(item)
                                        processed_results.append({
                                            "original_index": idx,
                                            "proposed_name": clean_text(value)
                                        })
                                        break
                                    except ValueError:
                                        pass
                    
                    if processed_results:
                        return processed_results
                    else:
                        if attempt < max_retries - 1:
                            st.warning(f"No valid results found. Retrying ({attempt+2}/{max_retries})...")
                            time.sleep(delay_seconds * (2 ** attempt))  # Exponential backoff
                            continue
                        else:
                            st.error("Failed to extract valid results after all retries.")
                            return None
                            
                except json.JSONDecodeError as je:
                    st.error(f"JSON decoding error: {je}")
                    st.text(f"Received JSON: {json_str}")
                    if attempt < max_retries - 1:
                        st.warning(f"Retrying ({attempt+2}/{max_retries})...")
                        time.sleep(delay_seconds * (2 ** attempt))  # Exponential backoff
                        continue
                    else:
                        return None
            else:
                st.error(f"No JSON found in the response")
                st.text(f"Received text: {response_text[:500]}...")  # Show first 500 chars
                if attempt < max_retries - 1:
                    st.warning(f"Retrying ({attempt+2}/{max_retries})...")
                    time.sleep(delay_seconds * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    return None
        except Exception as e:
            st.error(f"Error with Gemini: {e}")
            if attempt < max_retries - 1:
                st.warning(f"Retrying ({attempt+2}/{max_retries})...")
                time.sleep(delay_seconds * (2 ** attempt))  # Exponential backoff
                continue
            else:
                return None
    
    return None  # If we get here, all retries failed

# Main function to process all stations with Gemini in batches
def process_stations_with_gemini(df, api_key, address_col, name_col, other_name_cols=[], batch_size=50, prompt_type='auto', custom_template=None):
    """
    Processes DataFrame rows in batches using the Gemini API to generate proposed names.

    Args:
        df (pd.DataFrame): Input DataFrame.
        api_key (str): Google Gemini API key.
        address_col (str): Name of the column containing geocoded addresses.
        name_col (str): Name of the original station name column.
        other_name_cols (list): List of other potential name columns to include in the prompt.
        batch_size (int): Number of rows to process in each batch.
        prompt_type (str): Type of prompt to use - 'auto', 'nominatim', 'maps', or 'custom'
        custom_template (str, optional): Custom prompt template when prompt_type is 'custom'

    Returns:
        pd.DataFrame: DataFrame with added 'proposed_name' column.
                      Returns None if API configuration fails.
    """
    if not configure_gemini(api_key):
        return None  # Stop if API configuration fails

    try:
        # Initialize the generative model
        model = genai.GenerativeModel('gemini-1.5-flash-latest')  # Use the appropriate model
    except Exception as e:
        st.error(f"Failed to initialize Gemini model: {e}")
        return None

    if address_col not in df.columns:
        st.error(f"Address column '{address_col}' not found in the DataFrame.")
        return df  # Return original df if address column is missing

    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()

    # Prepare data for processing
    temp_df = result_df.reset_index()  # Reset index but keep original as 'index' column
    index_map = temp_df['index'].to_dict()  # Map of new index to original index

    # Prepare batch data
    cols_to_include = [name_col, address_col] + [col for col in other_name_cols if col in df.columns]
    batch_df = temp_df[cols_to_include].copy()

    # Initialize proposed_name column
    result_df['proposed_name'] = None

    total_batches = ceil(len(batch_df) / batch_size)
    total_rows = len(batch_df)
    st.write(f"Starting AI processing in {total_batches} batches...")
    progress_bar = st.progress(0, text="AI Naming in progress...")

    processed_count = 0

    for i in range(0, len(batch_df), batch_size):
        current_batch = batch_df.iloc[i:i+batch_size]
        # Prepare batch data for JSON (convert NaNs to None for JSON compatibility)
        batch_data = current_batch.where(pd.notna(current_batch), None).to_dict('records')

        # Add original_name field to align with expected format
        for j, item in enumerate(batch_data):
            item['original_name'] = item.get(name_col)
            item['original_index'] = j  # Store batch position index

        results = process_batch_with_gemini(
            model,
            batch_data,
            address_col,
            other_name_cols,
            prompt_type,
            max_retries=3,
            delay_seconds=1,
            custom_template=custom_template
        )

        if results:
            for item in results:
                batch_idx = item.get('original_index')
                proposed_name = item.get('proposed_name')

                if batch_idx is not None and proposed_name is not None:
                    # Get the original DataFrame index
                    df_idx = i + batch_idx  # Calculate index in the temporary DataFrame
                    original_idx = index_map.get(df_idx)  # Map to the original index
                    if original_idx is not None:
                        # Update the result DataFrame
                        result_df.loc[original_idx, 'proposed_name'] = proposed_name
                        processed_count += 1
        else:
            # Handle case where a batch completely failed
            st.warning(f"Batch {i // batch_size + 1} failed to return results.")

        # Update progress bar with processed count
        progress_bar.progress(processed_count / total_rows, text=f"AI Naming: {processed_count}/{total_rows}")

        time.sleep(1)  # Small delay between batches

    progress_bar.progress(1.0, text="AI Naming completed!")  # Ensure it reaches 100% at the end
    st.write(f"AI processing finished. Successfully processed {processed_count} items.")

    return result_df
