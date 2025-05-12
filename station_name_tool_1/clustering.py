# pages/3_Clustering.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import io # For download button

# --- Imports from your modules ---
from clustering import create_station_clusters, define_cluster_station_names, apply_manual_cluster_names, streamlit_cluster_selection_tabs
from utils import haversine_distance, clean_text # Needed for clustering logic if referenced directly

# --- Session State Initialization (Needed in every script using state) ---
# Initialize state variables if they don't exist
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
        'distance_threshold_m': 15,  # Default to 15 meters
        'remote_name_col': None,  # Optional column for remote names
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
if 'clustering_step' not in st.session_state:
    st.session_state.clustering_step = 1  # Track the current step in the workflow
if 'needs_manual_review' not in st.session_state:
    st.session_state.needs_manual_review = False  # Flag to indicate if manual review is needed
# --- End of Session State Initialization ---


# Set page configuration
st.set_page_config(page_title="🚉 Station Name Processing Tool - Clustering", layout="wide")

# Page Title
st.title("🗺️ 4. Clustering Stations")
st.markdown("""
### 🗂️ Group similar stations based on location.
""")

# --- Process Summary ---
st.markdown("## 🛠️ Process Overview")
st.markdown("""
1. **Clustering**: Group stations based on geographic proximity.
2. **Automatic Naming**: Define common names for each cluster where possible.
3. **Manual Review** (if needed): Assign names to clusters that couldn't be named automatically.
4. **Final Dataset**: Download the complete dataset with assigned cluster names.
""")
st.markdown("---")

# --- Helper function for download button ---
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
            st.error(f"Error preparing download file: {e}")


# --- MAIN UI CONTENT ---

# --- Select which dataset to use for clustering ---
st.markdown("### 📂 Select Dataset for Clustering")
available_datasets = []
dataset_labels = []

if st.session_state.df is not None:
    available_datasets.append("df")
    dataset_labels.append("Reverse Geocoding data")
    
if st.session_state.df_geocoded is not None:
    available_datasets.append("df_geocoded")
    dataset_labels.append("Geocoded data")
    
if st.session_state.df_ai_named is not None:
    available_datasets.append("df_ai_named")
    dataset_labels.append("AI Named data")

# Only show dataset selection if multiple options are available
if len(available_datasets) > 1:
    selected_dataset = st.selectbox(
        "Select dataset to use for clustering:",
        options=range(len(available_datasets)),
        format_func=lambda x: dataset_labels[x]
    )
    df_for_clustering = st.session_state[available_datasets[selected_dataset]].copy()
    st.write(f"Using {dataset_labels[selected_dataset]} for clustering")
elif len(available_datasets) == 1:
    df_for_clustering = st.session_state[available_datasets[0]].copy()
    st.write(f"Using {dataset_labels[0]} for clustering")
else:
    st.warning("Please complete Step 1: Reverse Geocoding first on the Home page.")
    st.stop()  # Stop execution if no dataset is available


# --- STEP 1: Configure Parameters and Run Clustering ---
if st.session_state.clustering_step == 1:
    st.markdown("### ⚙️ Step 1: Configure Clustering Parameters")

    name_col_cluster = st.selectbox(
        "Select Name Column for Clustering",
        list(df_for_clustering.columns),
        index=(
            list(df_for_clustering.columns).index('proposed_name')
            if 'proposed_name' in df_for_clustering.columns
            else next((i for i, col in enumerate(df_for_clustering.columns) if 'name' in col.lower()), 0)
        ),
        key="name_col_cluster_page3"
    )

    lat_col_cluster = st.selectbox(
        "Select Latitude Column for Clustering",
        list(df_for_clustering.columns),
        index=(
            list(df_for_clustering.columns).index('latitude')
            if 'latitude' in df_for_clustering.columns
            else next((i for i, col in enumerate(df_for_clustering.columns) if 'lat' in col.lower()), 0)
        ),
        key="lat_col_cluster_page3"
    )

    lon_col_cluster = st.selectbox(
        "Select Longitude Column for Clustering",
        list(df_for_clustering.columns),
        index=(
            list(df_for_clustering.columns).index('longitude')
            if 'longitude' in df_for_clustering.columns
            else next((i for i, col in enumerate(df_for_clustering.columns) if 'lng' in col.lower() or 'lng' in col.lower()), 0)
        ),
        key="lon_col_cluster_page3"
    )

    remote_name_col = st.selectbox(
        "Select Remote Name Column (Optional, for reference only)",
        ["None"] + list(df_for_clustering.columns),
        index=(
            1 + list(df_for_clustering.columns).index('remote_name')
            if 'remote_name' in df_for_clustering.columns
            else next((1 + i for i, col in enumerate(df_for_clustering.columns) if 'name' in col.lower()), 0)
        ),
        key="remote_name_col_page3"
    )
    if remote_name_col == "None":
        remote_name_col = None

    distance_threshold_m = st.number_input(
        "Distance Threshold (meters)", 
        min_value=1, 
        max_value=2000, 
        value=st.session_state.clustering_params.get('distance_threshold_m', 15), 
        step=1, 
        key="distance_threshold_m_page3"
    )

   # Update session state with all parameters
    st.session_state.clustering_params['name_col'] = name_col_cluster
    st.session_state.clustering_params['lat_col'] = lat_col_cluster
    st.session_state.clustering_params['lon_col'] = lon_col_cluster
    st.session_state.clustering_params['distance_threshold_m'] = distance_threshold_m
    st.session_state.clustering_params['remote_name_col'] = remote_name_col


    # --- Run Clustering Button ---
    if st.button("Run Clustering", key="run_clustering_button_page3"):
        if name_col_cluster and lat_col_cluster and lon_col_cluster:
            # Show a spinner while clustering is running
            with st.spinner("Clustering in progress..."):
                # Call the clustering function with all parameters
                df_clustered_result, clusters_dict = create_station_clusters(
                    df_for_clustering.copy(),
                    lat_col_cluster,
                    lon_col_cluster,
                    name_col_cluster,
                    distance_threshold_m
                )

                st.session_state.df_clustered = df_clustered_result
                st.session_state.clusters_for_review = clusters_dict

                if st.session_state.df_clustered is not None:
                    st.success(f"Clustering complete! Found {len(st.session_state.clusters_for_review)} potential clusters.")

                    # Automatically run the automatic naming process
                    with st.spinner("Automatically defining cluster names..."):
                        st.session_state.potential_cluster_names = define_cluster_station_names(
                            st.session_state.df_clustered,
                            st.session_state.clusters_for_review,
                            name_col_cluster
                        )

                        # Initialize manual names with potential names
                        st.session_state.manual_cluster_names = st.session_state.potential_cluster_names.copy()

                        # Check if manual review is needed
                        manual_review_needed = [name for name in st.session_state.potential_cluster_names.values() if name is None]
                        manual_review_count = len(manual_review_needed)

                        if manual_review_count > 0:
                            st.session_state.needs_manual_review = True
                            st.success(f"Automatic naming complete! {manual_review_count} clusters need manual review.")
                        else:
                            st.session_state.needs_manual_review = False
                            st.success("All clusters were automatically named! No manual review needed.")

                            # Apply the names automatically since no manual review is needed
                            st.session_state.df_final = apply_manual_cluster_names(
                                st.session_state.df_clustered.copy(),
                                st.session_state.manual_cluster_names
                            )

                # Move to the next step
                st.session_state.clustering_step = 2
                st.rerun()  # Changed from experimental_rerun to rerun
        else:
            st.warning("Please select Name, Latitude, and Longitude columns for clustering.")


# --- STEP 2: Show Results and Manual Review if Needed ---
elif st.session_state.clustering_step == 2:
    st.markdown("### 📊 Step 2: Review Clustering Results")
    if st.session_state.df_clustered is not None:
        # Create a container for the statistics section
        stats_container = st.container()
        
        with stats_container:
            st.subheader("Clustering Results")
            
            # Calculate statistics
            total_rows = len(st.session_state.df_clustered)
            clustered_rows = st.session_state.df_clustered[st.session_state.df_clustered['cluster_id'] > 0].shape[0]
            unclustered_rows = total_rows - clustered_rows
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Stations", total_rows)
            col2.metric("Clustered Stations", clustered_rows)
            col3.metric("Unclustered Stations", unclustered_rows)
        
        # Show automatically named clusters
        auto_named_clusters = {k: v for k, v in st.session_state.potential_cluster_names.items() 
                              if v is not None}
        
        if auto_named_clusters:
            st.subheader("Automatically Named Clusters")
            
            # Prepare the data for display
            auto_named_df = st.session_state.df_clustered.copy()
            auto_named_df = auto_named_df[auto_named_df['cluster_id'].isin(auto_named_clusters.keys())]
            auto_named_df['auto_suggested_name'] = auto_named_df['cluster_id'].map(auto_named_clusters)
            display_cols = ['cluster_id', st.session_state.clustering_params['remote_name_col'], 
                            st.session_state.clustering_params['name_col'], 'auto_suggested_name']
            display_cols = [col for col in display_cols if col in auto_named_df.columns]
            auto_named_df = auto_named_df[display_cols]

            # Display the table using st.dataframe
            st.dataframe(auto_named_df)
        else:
            st.info("No automatically named clusters available.")

        # Check if manual review is needed
        if st.session_state.needs_manual_review:
            st.subheader("Step 2: Manual Cluster Name Review")
            
            # Only display clusters that need manual review
            manual_review_needed = {
                k: v for k, v in st.session_state.clusters_for_review.items() 
                if k in st.session_state.potential_cluster_names and st.session_state.potential_cluster_names[k] is None
            }
            
            # Display the simplified cluster selection interface with checkboxes and text inputs
            for cluster_id, station_indices in manual_review_needed.items():
                name_col = st.session_state.clustering_params['name_col']
                remote_name_col = st.session_state.clustering_params['remote_name_col']
                
                st.markdown(f"#### Cluster {cluster_id}")
                selected_name = None
                
                # Create a DataFrame for just this cluster's stations
                cluster_stations = st.session_state.df_clustered.iloc[station_indices]
                station_names = []
                
                # Display each station's names
                for _, station in cluster_stations.iterrows():
                    station_info = f"{station[name_col]}"
                    if remote_name_col and pd.notna(station[remote_name_col]):
                        station_info = f"{station[name_col]} (Remote: {station[remote_name_col]})"
                    
                    if st.checkbox(f"Select: {station_info}", key=f"checkbox_{cluster_id}_{station[name_col]}"):
                        selected_name = station[name_col]

                # Allow custom name input
                custom_name = st.text_input(
                    f"Or enter a custom name for this cluster:",
                    value="",
                    key=f"custom_name_{cluster_id}"
                )

                # Update manual cluster names based on selection
                if selected_name:
                    st.session_state.manual_cluster_names[cluster_id] = selected_name
                elif custom_name:
                    st.session_state.manual_cluster_names[cluster_id] = custom_name
            
            # Button to apply manual names
            if st.button("Apply All Selections", key="apply_manual_names_button"):
                if st.session_state.df_clustered is not None and st.session_state.manual_cluster_names:
                    with st.spinner("Applying manual names to clusters..."):
                        # Merge automatic names with manual names
                        final_names = st.session_state.potential_cluster_names.copy()
                        for k, v in st.session_state.manual_cluster_names.items():
                            if v is not None:  # Only update if a value was entered
                                final_names[k] = v

                        st.session_state.df_final = apply_manual_cluster_names(
                            st.session_state.df_clustered.copy(),
                            final_names
                        )

                        if st.session_state.df_final is not None:
                            st.success("Manual names applied successfully!")
                            # Move to the final step
                            st.session_state.clustering_step = 3
                            st.rerun()

        else:
            # If no manual review is needed, show the final dataset and download button
            st.success("All clusters were automatically named! No manual review needed.")
            st.session_state.df_final = apply_manual_cluster_names(
                st.session_state.df_clustered.copy(),
                st.session_state.manual_cluster_names
            )

            if st.session_state.df_final is not None:
                st.subheader("Final Dataset")
                display_cols = ['cluster_id', st.session_state.clustering_params['name_col'], 'final_name']
                display_cols = [col for col in display_cols if col in st.session_state.df_final.columns]
                st.dataframe(st.session_state.df_final[display_cols])

                # Add download button
                create_download_button(st.session_state.df_final, "final_named", "download_final")

                # Option to restart the process
                if st.button("Restart Clustering Process", key="restart_process"):
                    st.session_state.clustering_step = 1
                    st.rerun()
    else:
        st.error("No clustered data available. Please run clustering first.")


# --- STEP 3: Final Results and Download ---
elif st.session_state.clustering_step == 3:
    st.markdown("### ✅ Step 3: Final Results and Download")
    
    name_col_cluster = st.session_state.clustering_params.get('name_col')
    
    if st.session_state.df_final is not None:
        # Show preview of final dataset
        display_cols = ['cluster_id', name_col_cluster, 'final_name']
        display_cols = [col for col in display_cols if col in st.session_state.df_final.columns]
        st.dataframe(st.session_state.df_final[display_cols])
        
        # Download button
        create_download_button(st.session_state.df_final, "final_named", "download_final")
        
        # Option to restart the process
        if st.button("Restart Clustering Process", key="restart_process"):
            st.session_state.clustering_step = 1
            st.rerun()  # Changed from experimental_rerun to rerun
    else:
        st.error("Final dataset not available.")
