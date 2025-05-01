# clustering.py
import pandas as pd
import numpy as np
import streamlit as st
from collections import Counter
from utils import haversine_distance, clean_text

def create_station_clusters(df, lat_col, lon_col, name_col, distance_threshold_m=15):
    """
    Creates clusters of stations based on geographic proximity.
    
    Args:
        df (pd.DataFrame): DataFrame containing station data
        lat_col (str): Name of latitude column
        lon_col (str): Name of longitude column
        name_col (str): Name of column containing station names
        distance_threshold_m (float): Distance threshold in meters
    
    Returns:
        tuple: (DataFrame with cluster_id column, dictionary of {cluster_id: indices})
    """
    # Convert distance threshold from meters to kilometers for haversine
    distance_threshold_km = distance_threshold_m / 1000.0
    
    # Create copy of dataframe to avoid modifying the original
    df_result = df.copy()
    
    # Initialize cluster ID column
    df_result['cluster_id'] = 0
    
    # Store indices of rows in each cluster
    clusters_dict = {}
    
    # Convert to numpy arrays for faster access
    lats = df_result[lat_col].values
    lons = df_result[lon_col].values
    
    # Validate coordinates - skip rows with invalid coordinates
    valid_coords = ~(np.isnan(lats) | np.isnan(lons))
    
    # Current cluster ID
    current_cluster_id = 1
    
    # Process all rows with valid coordinates
    n_rows = len(df_result)
    processed = np.zeros(n_rows, dtype=bool)
    
    for i in range(n_rows):
        # Skip if already processed or has invalid coordinates
        if processed[i] or not valid_coords[i]:
            continue
            
        # Create new cluster with this station
        cluster_indices = [i]
        df_result.loc[i, 'cluster_id'] = current_cluster_id
        processed[i] = True
        
        # Reference coordinates
        ref_lat, ref_lon = lats[i], lons[i]
        
        # Find all other points within distance threshold
        for j in range(n_rows):
            if j == i or processed[j] or not valid_coords[j]:
                continue
                
            # Calculate distance using haversine formula
            dist = haversine_distance(ref_lat, ref_lon, lats[j], lons[j])
            
            if dist <= distance_threshold_km:
                # Add to current cluster
                df_result.loc[j, 'cluster_id'] = current_cluster_id
                cluster_indices.append(j)
                processed[j] = True
        
        # Store cluster if it has more than one station
        if len(cluster_indices) > 1:
            clusters_dict[current_cluster_id] = cluster_indices
            current_cluster_id += 1
        else:
            # Revert single-station "clusters" back to 0
            df_result.loc[i, 'cluster_id'] = 0
    
    return df_result, clusters_dict

def define_cluster_station_names(df_clustered, clusters_dict, name_col):
    """
    Automatically determines the best name for each cluster based on name containment.
    Selects longer name only if it contains the shorter name.

    Args:
        df_clustered (pd.DataFrame): DataFrame with cluster_id column.
        clusters_dict (dict): Dictionary of {cluster_id: indices}.
        name_col (str): Column name containing station names.

    Returns:
        dict: Dictionary of {cluster_id: suggested_name}.
    """
    suggested_names = {}

    for cluster_id, indices in clusters_dict.items():
        # Get all names in this cluster
        names = df_clustered.iloc[indices][name_col].dropna().tolist()

        if not names:
            suggested_names[cluster_id] = None
            continue

        # Clean names for better matching
        clean_names = [clean_text(name) if isinstance(name, str) else "" for name in names]
        
        # Create a dictionary of clean_name: original_name mapping
        name_mapping = {clean: orig for clean, orig in zip(clean_names, names)}
        
        # Count frequency of each cleaned name
        name_counts = Counter(clean_names)
        
        if name_counts:
            # Get unique names ordered by frequency (highest first)
            unique_names = sorted(set(clean_names), key=lambda x: name_counts[x], reverse=True)
            
            # Start with the most frequent name
            best_name = unique_names[0]
            best_count = name_counts[best_name]
            
            # Compare with other names that have the same frequency
            same_freq_names = [name for name in unique_names if name_counts[name] == best_count]
            
            if len(same_freq_names) > 1:
                # Parmi les noms de même fréquence, chercher des relations d'inclusion
                found_contained = False
                for name1 in same_freq_names:
                    for name2 in same_freq_names:
                        if name1 != name2:
                            # Vérifie si un nom est contenu dans l'autre
                            if name1 in name2:  # Si name1 est contenu dans name2
                                best_name = name2  # Prendre le plus long
                                found_contained = True
                                break
                            elif name2 in name1:  # Si name2 est contenu dans name1
                                best_name = name1  # Prendre le plus long
                                found_contained = True
                                break
                    if found_contained:
                        break
                
                if not found_contained:
                    # Si aucune relation d'inclusion n'est trouvée, marquer pour revue manuelle
                    suggested_names[cluster_id] = None
                    continue

            suggested_names[cluster_id] = name_mapping[best_name]
        else:
            suggested_names[cluster_id] = None

    return suggested_names

def apply_manual_cluster_names(df, cluster_names_dict):
    """
    Applies the manually defined cluster names to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with cluster_id column
        cluster_names_dict (dict): Dictionary of {cluster_id: final_name}
        
    Returns:
        pd.DataFrame: DataFrame with final_name column added
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Initialize final_name column
    result_df['final_name'] = None
    
    # Apply cluster names where available
    for cluster_id, final_name in cluster_names_dict.items():
        if final_name is not None:
            # Apply to all rows in this cluster
            result_df.loc[result_df['cluster_id'] == cluster_id, 'final_name'] = final_name
    
    # For unclustered rows (cluster_id = 0) or clusters without a final name,
    # use the original name from the name column
    name_col = st.session_state.clustering_params.get('name_col')
    if name_col and name_col in result_df.columns:
        # Fill NaN final_name with original name
        mask = result_df['final_name'].isna()
        result_df.loc[mask, 'final_name'] = result_df.loc[mask, name_col]
    
    return result_df

def streamlit_cluster_selection_tabs(clusters_for_review_dict, original_df, potential_cluster_names, name_col='proposed_name', remote_name_col=None):
    """
    Generates Streamlit tabs for reviewing and manually renaming clusters.

    Args:
        clusters_for_review_dict (dict): {cluster_id: list_of_original_indices} for clusters to review.
        original_df (pd.DataFrame): The original DataFrame containing necessary columns.
        potential_cluster_names (dict): {cluster_id: potential_name} dictionary with auto-detected names.
        name_col (str): The column containing the proposed name.
        remote_name_col (str, optional): Column containing remote names for reference.

    Returns:
        dict: {cluster_id: manual_name} with potentially updated names from user input.
    """
    st.write("Please review and name the following clusters:")

    # Create a list of cluster IDs to review
    # Only include clusters that need manual review (None in potential_cluster_names)
    clusters_needing_review = []
    for cluster_id, indices in clusters_for_review_dict.items():
        if cluster_id in potential_cluster_names and potential_cluster_names[cluster_id] is None:
            clusters_needing_review.append(cluster_id)
    
    if not clusters_needing_review:
        st.info("No clusters need manual review.")
        return potential_cluster_names  # Return existing names

    # Sort cluster IDs for consistent tab order
    clusters_needing_review.sort()

    # Create tabs
    tabs = st.tabs([f"Cluster {cid}" for cid in clusters_needing_review])

    manual_cluster_names = potential_cluster_names.copy()

    for i, cluster_id in enumerate(clusters_needing_review):
        with tabs[i]:
            st.write(f"Details for Cluster {cluster_id}")

            original_indices = clusters_for_review_dict[cluster_id]
            # Select rows from the original DataFrame using the original indices
            cluster_df = original_df.loc[original_indices]

            if cluster_df.empty:
                st.warning(f"No data found for cluster {cluster_id}.")
                manual_cluster_names[cluster_id] = ""
                continue

            # Get all proposed names in this cluster
            display_cols = ['cluster_id', name_col]
            if remote_name_col and remote_name_col in cluster_df.columns:
                display_cols.append(remote_name_col)
            
            # Add other useful columns if they exist
            optional_cols = ['geocoded_address', 'geocoded_address_raw_data', 'explanation']
            display_cols += [col for col in optional_cols if col in cluster_df.columns]
            
            # Show all stations in this cluster
            st.dataframe(cluster_df[display_cols])

            # Get unique proposed names from this cluster for reference
            unique_names = cluster_df[name_col].dropna().unique().tolist()
            
            if unique_names:
                st.write("Available names in this cluster:")
                for name in unique_names:
                    st.write(f"- {name}")
            
            # Direct text input field (no radio button selection first)
            custom_name = st.text_input(
                "Enter name for this cluster:",
                value="",
                key=f"manual_name_{cluster_id}"
            )
            
            # Store the entered name
            if custom_name.strip():
                manual_cluster_names[cluster_id] = custom_name
            else:
                # If no custom name is provided but there are unique names, suggest the first one
                if unique_names:
                    manual_cluster_names[cluster_id] = unique_names[0]
                    st.info(f"Using '{unique_names[0]}' as default name. Edit the field above to change it.")
                else:
                    manual_cluster_names[cluster_id] = f"Cluster {cluster_id}"
                    st.info(f"Using 'Cluster {cluster_id}' as default name. Edit the field above to change it.")

    return manual_cluster_names