# pages/2_AI_Naming.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import io # For download button
import re # Added for response parsing
import json # Pour formatter les données JSON dans le template personnalisé

# --- Imports from your modules ---
from ai_processing import process_stations_with_gemini # The AI naming function
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
        'prompt_type': 'auto', # New parameter for prompt type
    }
if 'selected_address_keys' not in st.session_state: # Also initialize address keys state
    st.session_state.selected_address_keys = ['road', 'city', 'postcode', 'country'] # Default keys
if 'logged_prompt_type' not in st.session_state:
    st.session_state.logged_prompt_type = None  # Initialize logged prompt type
if 'custom_prompt_template' not in st.session_state:
    st.session_state.custom_prompt_template = ""  # Initialize custom prompt template
# --- End of Session State Initialization ---


# Set page configuration (Optional, but good practice)
st.set_page_config(page_title="Station Name Processing Tool - AI Naming", layout="wide")

# Always display the title and description
st.title("🤖 3. AI Naming with Gemini")
st.markdown("""
Use Gemini to suggest standardized names based on address data.

Download the AI-named results after completing this step to save your progress.
""")

# Check if a file is uploaded
if st.session_state.df is None:
    st.warning("Please upload your data in Step 1 before proceeding.")
    st.stop()

# Use geocoded data if available, otherwise fallback to the uploaded data
if st.session_state.df_geocoded is not None:
    df_for_ai = st.session_state.df_geocoded.copy()
else:
    df_for_ai = st.session_state.df.copy()

# --- Gestion de la clé API Gemini ---
if 'gemini_api_key' not in st.session_state:
    try:
        # Essayer de récupérer depuis les secrets de Streamlit
        from streamlit import secrets
        st.session_state.gemini_api_key = secrets.get("GEMINI_API_KEY", "")
    except Exception as e:
        # Si aucun secret n'est trouvé, initialiser avec une chaîne vide
        st.session_state.gemini_api_key = ""

# Indicateur visuel de présence d'une clé API par défaut
if st.session_state.gemini_api_key:
    st.success("✓ A default Gemini API key is configured")
    st.info("You can use the default key or enter your own below")
else:
    st.warning("No default Gemini API key found. Please enter your API key below.")

# Input pour la clé Gemini API avec valeur par défaut
gemini_api_key = st.text_input(
    "Enter your Gemini API Key (or leave as is to use the default key)",
    value=st.session_state.gemini_api_key,
    type="password", 
    key="gemini_api_key_page2"
)

# Mise à jour de la session state avec la valeur saisie ou la valeur par défaut
st.session_state.gemini_api_key = gemini_api_key

# Remove restrictions for address column selection
address_col_ai = st.selectbox(
    "Select Address Column for AI Naming",
    list(df_for_ai.columns),
    index=(
        list(df_for_ai.columns).index('geocoded_address')
        if 'geocoded_address' in df_for_ai.columns
        else next((i for i, col in enumerate(df_for_ai.columns) if 'address' in col.lower()), 0)
    ),
    key="address_col_ai_page2"
)

original_name_col_ai = st.selectbox(
    "Select Remote Name Column",
    list(df_for_ai.columns),
    index=(
        list(df_for_ai.columns).index('remote_name')
        if 'remote_name' in df_for_ai.columns
        else next((i for i, col in enumerate(df_for_ai.columns) if 'name' in col.lower()), 0)
    ),
    key="original_name_col_ai_page2"
)

# Remove restrictions for other potential name columns
other_name_cols_ai = st.multiselect(
    "Select Other Potential Name Columns (Optional)",
    list(df_for_ai.columns),
    default=st.session_state.ai_params['other_name_cols'],
    key="other_name_cols_ai_page2"
)

# Modifier la sélection du type de prompt pour inclure l'option personnalisée
prompt_type = st.radio(
    "Select Address Format Source",
    ["Auto-detect", "Nominatim (Structured)", "Maps API (Plain Text)", "Custom Prompt"],
    index=0,
    key="prompt_type_radio"
)

# Map the radio options to values we'll use in code
prompt_type_value = {
    "Auto-detect": "auto",
    "Nominatim (Structured)": "nominatim",
    "Maps API (Plain Text)": "maps",
    "Custom Prompt": "custom"
}[prompt_type]

# Ajouter une zone d'aide pour les formats d'adresse
with st.expander("Address Format Help"):
    st.markdown("""
    **Nominatim (Structured)**: Addresses structured with keys like 'road', 'city', 'country' as separate fields.
    
    **Maps API (Plain Text)**: Addresses as a single text string like "123 Main Street, New York, NY 10001".
    
    **Auto-detect**: System will try to determine the format based on address column content.
    
    **Custom Prompt**: Create your own prompt template with complete control over instructions given to the AI.
    """)

# Zone de texte conditionnelle pour le prompt personnalisé
if prompt_type == "Custom Prompt":
    # Initialisation du prompt personnalisé s'il n'existe pas dans la session
    if 'custom_prompt_template' not in st.session_state:
        st.session_state.custom_prompt_template = ""
    
    # Expander pour montrer un exemple de prompt
    with st.expander("Custom Prompt Example (Click to expand)"):
        st.markdown("### Example Nominatim Prompt")
        st.code("""
        Analyze the following structured address data and associated remote names to determine the best single, standardized name (`pn`) for each transportation station.

        Context: These data represent transportation stations (bus, train, etc.) in different languages and countries. Your task is to generate a clean, consistent English name for each station based on the provided information and the following rules.

        Goal: For each station, generate a single, standardized name (`pn`) that is informative, concise, and adheres to all rules.

        For each station, apply the following steps and formatting rules:

        Steps for Name Construction:
        1.  Identify the city or town or village name from the address structure (e.g., "city", "town").
        2.  Identify any important specific location information from the address structure (by priority: "suburb", "quarter", "neighbourhood", "amenity"...).
        3.  Construct the initial station name by combining the identified city/town, specific location, and the `remote_name` according to the following logic:
          * If the `remote_name` is the same as or a close variation of the identified city/town name, the generated name should be "City/Town Name - Specific Location" (if a distinct specific location was identified in step 2). If no specific location was identified, use just the "City/Town Name".
          * If the `remote_name` is *not* the same as the identified city/town name, the generated name should integrate the `remote_name` and the city/town name or specific location in a clear and informative way. Prioritize "City/Town Name - Remote Name" if the city/town name is clearly identifiable. Otherwise, use "Specific Location - Remote Name" or simply the "Remote Name" if no other useful distinguishing information is available from the address. Aim for the most intuitive and concise combination.

        Formatting Rules:
        4.  Remove all diacritics and accents from all parts of the name.
        5.  Ensure the final generated name (`pn`) does not exceed 10 words.
        6.  Don't keep station-related terms (e.g., "Terminal", "Gare", "Rodoviária", "Estación").
        7.  Avoid including redundant or consecutive duplicate words in the name.
        8.  Ensure the final generated name (`pn`) is clean and standardized, applying all the above rules.

Here is the data to process:
{batch_data}

Return the result as a JSON array of objects. Each object must have the following keys:
- oi: The original index of the station in the input list (used for mapping back to the DataFrame).
- pn: The single, clean, and standardized name you propose based on the address and other available names.

Ensure the output is ONLY the JSON array. Do not include any other text before or after the JSON.
```json
[
  {"oi": 0, "pn": "Sao Paulo - Santana"},
  {"oi": 1, "pn": "New York - Main Street"}
  // ... more objects
]
        """)

    # Instructions pour l'utilisateur
    st.info("Create your custom prompt template. Use `{batch_data}` as a placeholder - it will be replaced with the actual address data.")

    # Zone de texte pour le prompt personnalisé
    custom_prompt = st.text_area(
        "Your Custom Prompt Template",
        value=st.session_state.custom_prompt_template if st.session_state.custom_prompt_template else """
Analyze the following address data and determine the best station name for each:
{batch_data}
For each station, please:

Create a standardized name based on the address and original name.
Follow these rules:

Use city name as first part when possible
Include distinctive location details
Remove accents and diacritics
Keep names concise (max 10 words)



Return ONLY a JSON array with this format:
[
{"oi": 0, "pn": "Proposed Name 1"},
{"oi": 1, "pn": "Proposed Name 2"}
]""",
        height=400,
        key="custom_prompt_input"
    )
    # Mettre à jour la session state avec le prompt saisi
    st.session_state.custom_prompt_template = custom_prompt

    # Aide supplémentaire pour créer un bon prompt
    with st.expander("Tips for Writing Effective Custom Prompts"):
        st.markdown("""
        ### Best Practices for Custom Prompts
        
        1. **Include Clear Instructions**: Be specific about naming conventions and rules.
        
        2. **Use the Placeholder**: Always include `{batch_data}` where you want the address data to be inserted.
        
        3. **Specify Output Format**: Request output in the proper JSON format with these specific fields:
           - `oi`: Original index (integer)
           - `pn`: Proposed name (string)
           
        4. **Provide Examples**: Show sample inputs and desired outputs.
        
        5. **Define Formatting Rules**: Specify requirements for:
           - Handling accents/diacritics
           - Translation needs
           - Word count limits
           - Name structure (e.g., "City - Location")
           
        6. **Avoid Additional Text**: Instruct the AI to return ONLY the JSON array.
        
        7. **Error Handling**: Consider instructing how to handle incomplete or unusual data.
        """)
                
    # Bouton pour prévisualiser comment serait formaté le prompt
    if st.button("Preview Prompt with Sample Data", key="preview_prompt_button"):
        if st.session_state.custom_prompt_template:
            # Créer un petit échantillon de données
            sample_data = [
                {
                    "original_name": "Sample Station",
                    "original_index": 0,
                    address_col_ai: {"road": "Main Street", "city": "Sample City", "country": "Sample Country"}
                }
            ]
            
            # Formater le prompt avec l'échantillon
            sample_formatted = st.session_state.custom_prompt_template
            if '{batch_data}' in sample_formatted:
                sample_formatted = sample_formatted.replace('{batch_data}', json.dumps(sample_data, indent=2))
            else:
                sample_formatted += "\n\nHere is the data to process:\n" + json.dumps(sample_data, indent=2)
            
            # Afficher le prompt formaté
            st.subheader("Preview of Formatted Prompt")
            st.code(sample_formatted, language="markdown")
                    
else:
    # Si le mode custom n'est pas activé, s'assurer que la variable est vide
    if 'custom_prompt_template' in st.session_state:
        st.session_state.custom_prompt_template = ""

# Batch size for AI processing
batch_size_ai = st.number_input(
    "Batch Size for AI Processing",
    min_value=10,  # Minimum value set to 10
    max_value=100,  # Maximum value set to 100
    value=40,  # Default value set to 40
    step=5,  # Step increments of 5
    key="batch_size_ai_page2"
)

# Update session state with selected parameters
st.session_state.ai_params['address_col'] = address_col_ai
st.session_state.ai_params['name_col'] = original_name_col_ai
st.session_state.ai_params['other_name_cols'] = other_name_cols_ai
st.session_state.ai_params['batch_size'] = batch_size_ai
st.session_state.ai_params['prompt_type'] = prompt_type_value

# Validate the Gemini API key before proceeding
if not gemini_api_key.strip():
    st.error("The Gemini API key is missing. Please enter a valid API key.")
    st.stop()

# --- Run AI Naming button ---
st.header("⚙️ Configure AI Naming Parameters")

if st.button("Run AI Naming", key="run_ai_naming_button_page2"):
    if address_col_ai and original_name_col_ai:
        # Passer le prompt personnalisé à la fonction
        custom_template = st.session_state.custom_prompt_template if prompt_type == "Custom Prompt" else None
        
        # Call the AI processing function with added custom_template parameter
        st.session_state.df_ai_named = process_stations_with_gemini(
            df_for_ai.copy(),  # Pass a copy
            gemini_api_key.strip(),  # Ensure the key is stripped of whitespace
            address_col_ai,
            original_name_col_ai,
            other_name_cols_ai,
            batch_size_ai,
            prompt_type_value,  # Pass the prompt type to the function
            custom_template  # Pass the custom template
        )
        if st.session_state.df_ai_named is not None:
            st.success("AI naming complete!")
            st.dataframe(
                st.session_state.df_ai_named[
                    [address_col_ai, original_name_col_ai, 'proposed_name']  # Include original_name for clarity
                ].head(300)  # Show up to 300 rows
            )
    else:
        st.warning("Please select Address and Remote Name columns for AI processing.")

def create_download_button(df, filename_suffix, key):
    if df is not None:
        try:
            # Use BytesIO to handle the CSV data in memory
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue().encode('utf-8')

            download_filename = f"{st.session_state.last_uploaded_filename.split('.')[0]}_{filename_suffix}.csv"

            st.download_button(
                label=f"Download {filename_suffix.replace('_', ' ').title()} Data",
                data=csv_bytes,
                file_name=download_filename,
                mime="text/csv",
                key=key
            )
        except Exception as e:
            st.error(f"Error preparing download file for step {filename_suffix}: {e}")

create_download_button(st.session_state.df_ai_named, "ai_named", "download_ai_named_page2")
