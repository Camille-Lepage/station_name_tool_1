# Home.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import re # Added for response parsing

# --- Session State Initialization (Needed in every script using state) ---
# Initialize state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None
# Initialize other state variables that will be used across pages
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
    st.session_state.selected_address_keys = ['road', 'city', 'postcode', 'country'] # Default keys


# Set page configuration (This applies to all pages, but set it here)
st.set_page_config(page_title="🤖 AI StationNaming Tool", layout="wide")

# Display title and intro for Home page
st.title("🚉 AI StationNaming")
st.markdown("""
Welcome to the **AI StationNaming Tool**!  
This application helps you create a more informative and accurate station name.

### 🌟 Features:
- **📍 Reverse Geocoding**:  Increase station name precision by integrating relevant address component.
- **🤖 AI Naming**: Use AI to suggest standardized station names based on remote name and address component.
- **🗺️ Clustering**: Group similar station names based on coordinates and radius to determine the most relevant station name for each cluster.

At each step, download the results to save your progress and avoid data loss.

---

### 🚀 Instructions:
1. **Upload your dataset**: Use this page to upload your initial CSV file. The file must include at least a remote name, and it is highly recommended to include coordinates.
2. **Follow the steps**: Navigate through the steps using the sidebar:
   - Reverse geocoding → AI Naming → Clustering.
3. **Download results**: Save your progress at each step.

""")

# --- File Upload ---
st.header("📂 File Upload")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df.copy() # Store a copy of the original data
        st.session_state.last_uploaded_filename = uploaded_file.name
        st.success("File uploaded successfully!")

        # Display the first 5 rows and columns
        st.write("### Preview of Uploaded Data:")
        st.dataframe(df)
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

        # Display column names for user to select
        st.write("### Available Columns:")
        st.write(list(df.columns))

        st.info("Your data has been successfully loaded! Use the sidebar to proceed to Step 1: Reverse Geocoding.")

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.session_state.df = None
