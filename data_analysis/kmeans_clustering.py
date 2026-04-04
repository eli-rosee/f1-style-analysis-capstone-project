import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from race_data import RaceData
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.mixture import GaussianMixture

# ── Toggle this to switch between raw telemetry and PCA-reduced data ──
#PCA should be set to false for single variable clustering
USE_PCA = True


def _build_matrix(data_dict, drivers):
    all_laps = []
    for driver in drivers:
        for lap in data_dict[driver]:
            if isinstance(lap, np.ndarray):
                all_laps.append(lap.flatten())
            else:
                all_laps.append(lap.values.flatten())
    return np.array(all_laps)

def k_means_cluster(data_dict, drivers, cluster_num):
    print("\nClustering data (KMeans)...\n")
    X = _build_matrix(data_dict, drivers)

    km = KMeans(n_clusters=cluster_num, random_state=42)
    labels = km.fit_predict(X)
    sil_score = silhouette_score(X, labels)
    dbi_score = davies_bouldin_score(X, labels)
    cah_score = calinski_harabasz_score(X, labels)

    return labels, sil_score, dbi_score, cah_score

def attach_labels(interp_dict, drivers, labels):
    lap_index = 0
    for driver in drivers:
        for lap_df in interp_dict[driver]:
            lap_df['cluster_label'] = labels[lap_index]
            lap_index += 1

#calculate percentage of laps a driver falls into each cluster
def driver_cluster_distribution(interp_dict, drivers):
    for driver in drivers:
        cluster_counts = {}

        for lap_df in interp_dict[driver]:
            label = lap_df['cluster_label'].iloc[0]
            cluster_counts[label] = cluster_counts.get(label, 0) + 1
        
        total = sum(cluster_counts.values())
        print(f"{driver} Cluster distribution")

        for cluster_id in sorted(cluster_counts.keys()):
            proportion = cluster_counts[cluster_id] / total
            print(f"  cluster {cluster_id}: {proportion:.2f}")
    
    print()

#calculate the mean values of each cluster
def cluster_mean_telemetry(interp_dict, drivers, norm_tel_columns):
    cluster_laps = {}
    for driver in drivers:
        for lap_df in interp_dict[driver]:
            label = lap_df['cluster_label'].iloc[0]
            if label not in cluster_laps:
                cluster_laps[label] = []
            cluster_laps[label].append(lap_df)

    for cluster_id in sorted(cluster_laps.keys()):
        laps = cluster_laps[cluster_id]
        all_data = pd.concat(laps)
        print(f"Cluster {cluster_id} ({len(laps)} laps)")
        for col in norm_tel_columns:
            mean = all_data[col].mean()
            print(f"  {col}: {mean:.3f}")

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

    import pandas as pd

#denormalizes the clustering data and saves it to an external csv file
def export_clusters_to_csv(race, clusterVariables, filename="clustering_results.csv"):
    print(f"\nExporting denormalized clustering results to {filename}...")
    
    all_laps_data = []

    for driver in race.drivers:
        prev_lap_increment = 0
        min_dict, max_dict = {}, {}
        
        for lap_index, lap_df in enumerate(race.interp_dict[driver], start=1):

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

#Generates a statistical summary (Min, Max, Avg) for the clustered variable across all drivers and saves it to a CSV.
def export_cluster_summary(race, clusterVariables, filename="cluster_summary.csv"):
    print(f"Generating clustering results summary for {clusterVariables}...")
    
    all_summaries = []

    for driver in race.drivers:
        driver_laps = []
        
        #use enumerate to get lap_idx (starting at 1) without using .index()
        for lap_idx, lap_df in enumerate(race.interp_dict[driver], start=1):
            temp_df = lap_df.copy()
            
            #find the correct 5-lap chunk for denormalization
            lap_inc = lap_idx + (5 - lap_idx % 5)
            min_d, max_d = race._get_min_max_driver_lap(driver, lap_inc - 5, lap_inc)
            
            #denormalize the variables
            for var in clusterVariables:
                temp_df[var] = (temp_df[var] * (max_d[var] - min_d[var])) + min_d[var]
            
            driver_laps.append(temp_df)
        
        if not driver_laps:
            continue
            
        #combine all laps for this driver into one DataFrame for analysis
        df_combined = pd.concat(driver_laps)
        total_points = len(df_combined)

        #calculate stats for every variable in clusterVariables
        for var in clusterVariables:
            #group by the cluster labels and aggregate
            stats = df_combined.groupby('cluster_label')[var].agg(['count', 'min', 'max', 'mean']).reset_index()
            
            stats['Percentage_of_Lap'] = (stats['count'] / total_points) * 100
            stats['race'] = race.race_name
            stats['driver'] = driver
            stats['variable'] = var
            
            #rename for the final output
            stats = stats.rename(columns={
                'cluster_label': 'Cluster',
                'min': 'Min',
                'max': 'Max',
                'mean': 'Avg'
            })
            
            all_summaries.append(stats)

    #finalize and save
    if all_summaries:
        final_df = pd.concat(all_summaries, ignore_index=True)
        #reorder columns
        cols = ['race', 'driver', 'variable', 'Cluster', 'Percentage_of_Lap', 'Min', 'Max', 'Avg']
        final_df = final_df[cols]
        
        final_df.to_csv(filename, index=False)
        print(f"Summary saved to: {filename}")
    else:
        print("No data available to summarize.")


def main():
    race_name = 'Emilia_Romagna_Grand_Prix'

    #All columns that can be fetched from the database:
    #'rel_distance', 'time', 'track_coordinate_x', 'track_coordinate_y', 'track_coordinate_z', 'rpm', 'gear', 'throttle', 'brake', 'drs', 'speed', 'acc_x', 'acc_y', 'acc_z'
    
    #change variables to cluster on here
    #remember to set PCA to false if you are only clustering on one variable
    #clusterVariables = ["rpm", "speed", "throttle", "brake", "gear", "acc_x", "acc_y", "acc_z"]
    clusterVariables = ["acc_x", "acc_y", "acc_z"]

    #create RaceData object with the defined columns
    race = RaceData(race_name, clusterVariables)

    drivers = race.drivers
    data_dict = {}

    if USE_PCA:
        data_dict = race.reduced_dict
    else:
        data_dict = race.interp_dict

    #create kmeans cluster
    #change number of clusters depending on the variable 
    labels, sil, dbi, cah = k_means_cluster(data_dict, drivers, cluster_num=5)

    attach_labels(race.interp_dict, drivers, labels)

    #calculate percentage of laps a driver falls into each cluster
    driver_cluster_distribution(race.interp_dict, drivers)

    #calculate the mean values of each cluster
    cluster_mean_telemetry(race.interp_dict, drivers, race.norm_columns)

    #create visual for cluster results
    #visualize_clusters(race.interp_dict, race.reduced_dict, drivers)

    #print cluster performance metrics
    print(f"\nSilhouette: {sil:.4f}") #Range [-1, 1]. Higher score indicates better clustering
    print(f"Davies Bouldin: {dbi:.4f}") #Lower score indicates better clustering
    print(f"Calinski Harabasz: {cah:.4f}") #Higher score indicates better clustering

    #save cluster results to a .csv
    export_clusters_to_csv(race, clusterVariables)

    #save summary of cluster results to a .csv
    export_cluster_summary(race, clusterVariables)

if __name__ == '__main__':
    main()