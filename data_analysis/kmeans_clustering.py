import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from race_data import RaceData
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
import json
import os

# ── Toggle this to switch between raw telemetry and PCA-reduced data ──
#PCA should be set to false for single variable clustering
USE_PCA = True


def _build_matrix(data_dict, drivers):
    all_laps = []
    lap_refs = []   # ← NEW: track (driver, lap_index)
    expected_shape = None
 
    for driver in drivers:
        for lap_idx, lap in enumerate(data_dict[driver]):
 
            if isinstance(lap, np.ndarray):
                flat = lap.flatten()
            else:
                flat = lap.values.flatten()
 
            if flat.size == 0:
                continue
 
            if expected_shape is None:
                expected_shape = flat.shape
 
            if flat.shape == expected_shape:
                all_laps.append(flat)
                lap_refs.append((driver, lap_idx))  # ← track it
            else:
                print(f"Skipping lap with inconsistent shape: {flat.shape}")
 
    return np.vstack(all_laps), lap_refs

def k_means_cluster(data_dict, drivers, cluster_num):
    print("\nClustering data (KMeans)...\n")
    X, lap_refs = _build_matrix(data_dict, drivers)

    km = KMeans(n_clusters=cluster_num, random_state=42)
    labels = km.fit_predict(X)
    sil_score = silhouette_score(X, labels)
    dbi_score = davies_bouldin_score(X, labels)
    cah_score = calinski_harabasz_score(X, labels)

    return labels, sil_score, dbi_score, cah_score, lap_refs

def attach_labels(interp_dict, lap_refs, labels):
    for i, (driver, lap_idx) in enumerate(lap_refs):
        interp_dict[driver][lap_idx]['cluster_label'] = labels[i]

#calculate percentage of laps a driver falls into each cluster
def driver_cluster_distribution(interp_dict, drivers):
    for driver in drivers:
        cluster_counts = {}

        for lap_df in interp_dict[driver]:
            if 'cluster_label' not in lap_df.columns:
                continue

            label = lap_df['cluster_label'].iloc[0]
            cluster_counts[label] = cluster_counts.get(label, 0) + 1
        
        total = sum(cluster_counts.values())
        print(f"{driver} Cluster distribution")

        for cluster_id in sorted(cluster_counts.keys()):
            proportion = cluster_counts[cluster_id] / total
            print(f"  cluster {cluster_id}: {proportion:.2f}")
    
    print()

#Calculates and returns the mean values of each cluster
def cluster_mean_telemetry(interp_dict, drivers, norm_tel_columns):
    cluster_laps = {}
    
    #stores the mean values of each cluster in a nested dictionary
    cluster_mean_data = {}
    
    for driver in drivers:
        for lap_df in interp_dict[driver]:
            if 'cluster_label' not in lap_df.columns:
                continue
            label = lap_df['cluster_label'].iloc[0]
            if label not in cluster_laps:
                cluster_laps[label] = []
            cluster_laps[label].append(lap_df)

    for cluster_id in sorted(cluster_laps.keys()):
        laps = cluster_laps[cluster_id]
        all_data = pd.concat(laps)

        #Initialize the nested dictionary for this specific cluster
        cluster_id_str = str(cluster_id)
        cluster_mean_data[cluster_id_str] = {
            "lap_count": len(laps),
            "telemetry_means": {}
        }
        

        print(f"Cluster {cluster_id} ({len(laps)} laps)")
        for col in norm_tel_columns:
            mean = all_data[col].mean()

            #Store the mean inside the sub-dictionary
            cluster_mean_data[cluster_id_str]["telemetry_means"][col] = mean

            print(f"  {col}: {mean:.3f}")
    
    #return the mean data dictionary
    return cluster_mean_data

#create cluster export (JSON 2) given the number of clusters per lap in a dictionary
def create_cluster_export(race_name, mean_cluster_data, filename="cluster_overview.json"):
    #Get the total number of laps
    total_laps = sum(info['lap_count'] for info in mean_cluster_data.values())

    #Used to store info for each cluster
    clusters_list = []

    #Compute the weighted avg of each metric
    average_values = {}
    for cluster_id, info in mean_cluster_data.items():
        lap_count = info['lap_count']
        lap_percentage = (lap_count / total_laps) if total_laps > 0 else 0
        for metric, mean_value in info['telemetry_means'].items():
            if metric not in average_values:
                average_values[metric] = 0
            average_values[metric] += (lap_percentage * mean_value)

    #round weighted averages of each metric
    for key in average_values:
        average_values[key] = round(average_values[key], 3)

    #Iterate through the dictionary items to build each cluster entry for the JSON file
    for cluster_id, info in mean_cluster_data.items():
        #get total number of laps for each cluster
        lap_count = info['lap_count']

        #get percentage of laps that fall within each cluster
        lap_percentage = round((lap_count / total_laps), 3) if total_laps > 0 else 0

        #calculate deviation from average for each cluster metric
        deviation_summary = {}
        for metric, mean_value in info['telemetry_means'].items():
            deviation_summary[metric] = round((mean_value - average_values[metric]), 3)

        #store values in a cluster_entry dictionary
        cluster_entry = {
            "cluster_id": cluster_id,
            "cluster_name": f"Cluster {cluster_id}",
            "lap_count": lap_count,
            "lap_percentage": lap_percentage,
            "deviation_summary": deviation_summary
        }

        #add cluster_entry to clusters_list
        clusters_list.append(cluster_entry)

    #create the final JSON structure
    export_data = {
        "race": race_name,
        "clusters": clusters_list
    }


    output_dir = os.path.join("../frontend/clustering_results", race_name)
    os.makedirs(output_dir, exist_ok=True)

    json_filename = os.path.join(output_dir, filename)

    #Save summary of cluster results to a .JSON
    export_driver_distribution_json(export_data, filename=json_filename)

    '''
    #save to a JSON file
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Successfully saved to {filename}")
    '''


def visualize_clusters(interp_dict, reduced_dict, drivers, use_pca=True):
    pc1_vals = []
    pc2_vals = []
    labels = []

    for driver in drivers:
        #Zip interp_dict (for labels) and reduced_dict (for coordinates)
        for lap_df, lap_data in zip(interp_dict[driver], reduced_dict[driver]):
            
            #Handle PCA vs Raw Telemetry data types
            #PCA data is a numpy array; interp_dict is a list of DataFrames
            if isinstance(lap_data, pd.DataFrame):
                #If using raw telemetry, we only want the columns we clustered on
                #Take the mean across the lap for visualization
                coords = lap_data.values 
            else:
                coords = lap_data

            #Extract first dimension (PC1 or first variable)
            pc1_vals.append(coords[:, 0].mean())

            #Extract second dimension if it exists
            if coords.shape[1] > 1:
                pc2_vals.append(coords[:, 1].mean())
            else:
                #For single variate, use a dummy value for the Y-axis
                pc2_vals.append(0)

            labels.append(lap_df['cluster_label'].iloc[0])

    # --- Plotting Logic ---
    plt.figure(figsize=(10, 6))
    
    #check if we actually have a second dimension to plot
    has_second_dim = any(v != 0 for v in pc2_vals)

    if has_second_dim:
        scatter = plt.scatter(pc1_vals, pc2_vals, c=labels, cmap='tab10', alpha=0.6, edgecolors='w')
        plt.ylabel('PC2 / Variable 2')
    else:
        #1D Visualization
        y_jitter = np.random.normal(0, 0.01, size=len(pc1_vals))
        scatter = plt.scatter(pc1_vals, y_jitter, c=labels, cmap='tab10', alpha=0.6, edgecolors='w')
        plt.yticks([]) # Hide Y-axis as it's just noise for 1D
        plt.ylim(-0.1, 0.1)

    plt.colorbar(scatter, label='Cluster Label')
    plt.xlabel('PC1 / Variable 1')
    
    title_suffix = "(PCA-Reduced)" if use_pca else "(Raw Features)"
    plt.title(f'F1 Lap Clustering {title_suffix}')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('clusters.png')
    plt.show()

#denormalizes the clustering data and saves it to an external csv file
def export_clusters_to_csv(race, clusterVariables, filename="clustering_results.csv"):
    print(f"\nExporting denormalized clustering results to {filename}...")
    
    all_laps_data = []

    for driver in race.drivers:
        
        prev_lap_increment = 0
        min_dict, max_dict = {}, {}
        
        for lap_index, lap_df in enumerate(race.interp_dict[driver], start=1):
            if 'cluster_label' not in lap_df.columns:
                continue
            #Recreate the 5-lap chunking logic from the _normalize function
            #Ensures we grab the exact min/max used for this specific lap
            lap_increment = lap_index + (5 - lap_index % 5)
            if lap_increment != prev_lap_increment:
                min_dict, max_dict = race._get_min_max_driver_lap(driver, lap_increment - 5, lap_increment)
                prev_lap_increment = lap_increment
            
            #make a copy so we don't overwrite the normalized data currently in memory
            export_df = lap_df.copy()
            
            #Denormalize the columns that have been clustered on
            for col in clusterVariables:
                # x = x_norm * (max - min) + min
                export_df[col] = (export_df[col] * (max_dict[col] - min_dict[col])) + min_dict[col]

            columns_to_keep = ['Driver', 'Lap_Index'] + clusterVariables + ['cluster_label']
            
            #Add identifying metadata to the front
            export_df.insert(0, 'Driver', driver)
            export_df.insert(1, 'Lap_Index', lap_index)

            export_df = export_df[columns_to_keep]
            
            all_laps_data.append(export_df)

    #combine everything and export
    if all_laps_data:
        final_csv_df = pd.concat(all_laps_data, ignore_index=True)
        final_csv_df.to_csv(filename, index=False)
        print(f"Successfully saved {len(final_csv_df)} rows to {filename}\n")
    else:
        print("Error. No data found to export.")

#export driver distribution json
def export_driver_distribution_json(all_driver_distributions, filename="Driver_Distribution.json"):
    print(f"Saving driver distributions to {filename}...")

    with open(filename, "w") as f:
        json.dump(all_driver_distributions, f, indent=2)

    print(f"Saved driver distributions to {filename}")

def main():

    # Load race metadata from cache
    with open('cache/race_metadata_cache.json', 'r') as f:
        race_metadata = json.load(f)

    available_races = list(race_metadata['races'].keys())

    print("\nF1 LAP CLUSTERING ANALYSIS")

    # Ask user if they want multivariate clustering
    print("\nClustering Mode:")
    print("1. Multivariate (cluster on multiple metrics)")
    print("2. Single Variable (cluster on one metric)")

    mode_choice = input("\nSelect mode (1 or 2): ").strip()

    if mode_choice == "1":
        # Multivariate clustering
        use_pca_mode = True
        clusterVariables = ["rpm", "speed", "throttle", "brake", "gear", "acc_x", "acc_y", "acc_z"]
        print(f"\nClustering on multivariate metrics: {clusterVariables}")

    else:
        # Single variable clustering
        use_pca_mode = False
        print("\nAvailable metrics for single-variable clustering:")
        available_metrics = ['rpm', 'gear', 'throttle', 'brake', 'speed', 'acc_x', 'acc_y', 'acc_z']

        for i, metric in enumerate(available_metrics, 1):
            print(f"  {i}. {metric}")

        metric_choice = input(f"\nSelect metric (1-{len(available_metrics)}): ").strip()

        try:
            metric_idx = int(metric_choice) - 1

            if 0 <= metric_idx < len(available_metrics):
                clusterVariables = [available_metrics[metric_idx]]
                print(f"\nClustering on single metric: {clusterVariables[0]}")

            else:
                print("Invalid selection. Using default: rpm")
                clusterVariables = ["rpm"]

        except ValueError:
            print("Invalid input. Using default: rpm")
            clusterVariables = ["rpm"]

    # Ask user about track selection
    print("Track Selection:")
    print("1. All available tracks")
    print("2. Specific track")

    track_choice = input("\nSelect option (1 or 2): ").strip()

    races_to_analyze = []

    if track_choice == "1":
        # All tracks
        races_to_analyze = available_races
        print(f"\nAnalyzing all {len(available_races)} tracks")

    else:
        # Specific track
        print("\nAvailable tracks:")
        for i, race in enumerate(available_races, 1):
            print(f"  {i}. {race}")

        race_selection = input(f"\nSelect track (1-{len(available_races)}): ").strip()

        try:
            race_idx = int(race_selection) - 1

            if 0 <= race_idx < len(available_races):
                races_to_analyze = [available_races[race_idx]]
                print(f"\nAnalyzing: {races_to_analyze[0]}")

            else:
                print("Invalid selection. Using default: Canadian_Grand_Prix")
                races_to_analyze = ['Canadian_Grand_Prix']

        except ValueError:
            print("Invalid input. Using default: Canadian_Grand_Prix")
            races_to_analyze = ['Canadian_Grand_Prix']

    # Ask about number of clusters
    cluster_num_input = input("\nNumber of clusters (default 5): ").strip()
    try:
        cluster_num = int(cluster_num_input) if cluster_num_input else 5
    except ValueError:
        cluster_num = 5
        print("Invalid input. Using default: 5 clusters")

    print("\nSTARTING ANALYSIS\n")

    # Process each selected race
    for race_name in races_to_analyze:

        all_driver_distributions = []

        print(f"Processing: {race_name}\n")

        # Create RaceData object with the defined columns
        race = RaceData(race_name, clusterVariables)

        drivers = race.drivers
        data_dict = {}

        if use_pca_mode:
            data_dict = race.reduced_dict
        else:
            data_dict = race.interp_dict

        # Create kmeans cluster
        labels, sil, dbi, cah, lap_refs = k_means_cluster(data_dict, drivers, cluster_num=cluster_num)

        attach_labels(race.interp_dict, lap_refs, labels)

        for driver in drivers:
            cluster_counts = {}

            for lap_df in race.interp_dict[driver]:
                if 'cluster_label' not in lap_df.columns:
                    continue

                label = int(lap_df['cluster_label'].iloc[0])
                cluster_counts[label] = cluster_counts.get(label, 0) + 1

            usable_laps = sum(cluster_counts.values())

            cluster_distribution = []
            for cluster_id in range(cluster_num):
                lap_count = cluster_counts.get(cluster_id, 0)
                lap_percentage = round(lap_count / usable_laps, 3) if usable_laps > 0 else 0.0

                cluster_distribution.append({
                    "cluster_id": cluster_id,
                    "lap_count": lap_count,
                    "lap_percentage": lap_percentage
                })

            sorted_clusters = sorted(
                cluster_distribution,
                key=lambda x: x["lap_count"],
                reverse=True
            )

            primary_style_cluster = sorted_clusters[0]["cluster_id"] if len(sorted_clusters) > 0 else None
            secondary_style_cluster = sorted_clusters[1]["cluster_id"] if len(sorted_clusters) > 1 else None

            driver_summary = {
                "race": race.race_name,
                "driver": driver,
                "usable_laps": usable_laps,
                "cluster_distribution": cluster_distribution,
                "primary_style_cluster": primary_style_cluster,
                "secondary_style_cluster": secondary_style_cluster
            }

            all_driver_distributions.append(driver_summary)

        # Calculate percentage of laps a driver falls into each cluster
        driver_cluster_distribution(race.interp_dict, drivers)

        #Print and retrieve the mean values of each cluster
        mean_cluster_data = cluster_mean_telemetry(race.interp_dict, drivers, race.norm_columns)

        #create cluster export (JSON 2) given the number of clusters per lap in a dictionary
        create_cluster_export(race_name, mean_cluster_data)

        # Print cluster performance metrics
        #print(f"\nCluster Performance Metrics:")
        #print(f"Silhouette: {sil:.4f}") #Range [-1, 1]. Higher score indicates better clustering
        #print(f"Davies Bouldin: {dbi:.4f}") #Lower score indicates better clustering
        #print(f"Calinski Harabasz: {cah:.4f}") #Higher score indicates better clustering
        

        output_dir = os.path.join("../frontend/clustering_results", race_name)
        os.makedirs(output_dir, exist_ok=True)

        json_filename = os.path.join(output_dir, "driver_distribution.json")

        # Save summary of cluster results to a .JSON
        export_driver_distribution_json(all_driver_distributions, filename=json_filename)
        # Export 1 – Track Summar
        race.exporter.export_track_summary()

    print("ANALYSIS COMPLETE")


if __name__ == '__main__':
    main()
