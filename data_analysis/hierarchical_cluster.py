import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from race_data import RaceData
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram

# ── Toggle this to switch between raw telemetry and PCA-reduced data ──
# PCA should be set to false for single variable clustering
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


def hierarchical_cluster(data_dict, drivers, cluster_num, linkage_method='ward'):
    print("\nClustering data (Hierarchical)...\n")
    X = _build_matrix(data_dict, drivers)

    hc = AgglomerativeClustering(
        n_clusters=cluster_num,
        linkage=linkage_method
    )

    labels = hc.fit_predict(X)
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


# calculate percentage of laps a driver falls into each cluster
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


# calculate the mean values of each cluster
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
        for lap_df, lap_data in zip(interp_dict[driver], reduced_dict[driver]):

            if isinstance(lap_data, pd.DataFrame):
                coords = lap_data.values
            else:
                coords = lap_data

            pc1_vals.append(coords[:, 0].mean())

            if coords.shape[1] > 1:
                pc2_vals.append(coords[:, 1].mean())
            else:
                pc2_vals.append(0)

            labels.append(lap_df['cluster_label'].iloc[0])

    plt.figure(figsize=(10, 6))

    has_second_dim = any(v != 0 for v in pc2_vals)

    if has_second_dim:
        scatter = plt.scatter(pc1_vals, pc2_vals, c=labels, cmap='tab10', alpha=0.6, edgecolors='w')
        plt.ylabel('PC2 / Variable 2')
    else:
        y_jitter = np.random.normal(0, 0.01, size=len(pc1_vals))
        scatter = plt.scatter(pc1_vals, y_jitter, c=labels, cmap='tab10', alpha=0.6, edgecolors='w')
        plt.yticks([])
        plt.ylim(-0.1, 0.1)

    plt.colorbar(scatter, label='Cluster Label')
    plt.xlabel('PC1 / Variable 1')

    title_suffix = "(PCA-Reduced)" if use_pca else "(Raw Features)"
    plt.title(f'F1 Lap Clustering {title_suffix} - Hierarchical')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('hierarchical_clusters.png')
    plt.show()


def plot_dendrogram(data_dict, drivers, method='ward'):
    print(f"\nGenerating dendrogram using {method} linkage...\n")
    X = _build_matrix(data_dict, drivers)

    Z = linkage(X, method=method)

    plt.figure(figsize=(14, 7))
    dendrogram(Z)
    plt.title(f'Hierarchical Clustering Dendrogram ({method.capitalize()} Linkage)')
    plt.xlabel('Lap Index')
    plt.ylabel('Distance')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('hierarchical_dendrogram.png')
    plt.show()


# denormalizes the clustering data and saves it to an external csv file
def export_clusters_to_csv(race, clusterVariables, filename="hierarchical_clustering_results.csv"):
    print(f"\nExporting denormalized clustering results to {filename}...")

    all_laps_data = []

    for driver in race.drivers:
        prev_lap_increment = 0
        min_dict, max_dict = {}, {}

        for lap_index, lap_df in enumerate(race.interp_dict[driver], start=1):

            lap_increment = lap_index + (5 - lap_index % 5)
            if lap_increment != prev_lap_increment:
                min_dict, max_dict = race._get_min_max_driver_lap(driver, lap_increment - 5, lap_increment)
                prev_lap_increment = lap_increment

            export_df = lap_df.copy()

            for col in clusterVariables:
                export_df[col] = (export_df[col] * (max_dict[col] - min_dict[col])) + min_dict[col]

            columns_to_keep = ['Driver', 'Lap_Index'] + clusterVariables + ['cluster_label']

            export_df.insert(0, 'Driver', driver)
            export_df.insert(1, 'Lap_Index', lap_index)

            export_df = export_df[columns_to_keep]

            all_laps_data.append(export_df)

    if all_laps_data:
        final_csv_df = pd.concat(all_laps_data, ignore_index=True)
        final_csv_df.to_csv(filename, index=False)
        print(f"Successfully saved {len(final_csv_df)} rows to {filename}\n")
    else:
        print("Error. No data found to export.")


# Generates a statistical summary (Min, Max, Avg) for the clustered variable across all drivers and saves it to a CSV.
def export_cluster_summary(race, clusterVariables, filename="hierarchical_cluster_summary.csv"):
    print(f"Generating clustering results summary for {clusterVariables}...")

    all_summaries = []

    for driver in race.drivers:
        driver_laps = []

        for lap_idx, lap_df in enumerate(race.interp_dict[driver], start=1):
            temp_df = lap_df.copy()

            lap_inc = lap_idx + (5 - lap_idx % 5)
            min_d, max_d = race._get_min_max_driver_lap(driver, lap_inc - 5, lap_inc)

            for var in clusterVariables:
                temp_df[var] = (temp_df[var] * (max_d[var] - min_d[var])) + min_d[var]

            driver_laps.append(temp_df)

        if not driver_laps:
            continue

        df_combined = pd.concat(driver_laps)
        total_points = len(df_combined)

        for var in clusterVariables:
            stats = df_combined.groupby('cluster_label')[var].agg(['count', 'min', 'max', 'mean']).reset_index()

            stats['Percentage_of_Lap'] = (stats['count'] / total_points) * 100
            stats['race'] = race.race_name
            stats['driver'] = driver
            stats['variable'] = var

            stats = stats.rename(columns={
                'cluster_label': 'Cluster',
                'min': 'Min',
                'max': 'Max',
                'mean': 'Avg'
            })

            all_summaries.append(stats)

    if all_summaries:
        final_df = pd.concat(all_summaries, ignore_index=True)
        cols = ['race', 'driver', 'variable', 'Cluster', 'Percentage_of_Lap', 'Min', 'Max', 'Avg']
        final_df = final_df[cols]

        final_df.to_csv(filename, index=False)
        print(f"Summary saved to: {filename}")
    else:
        print("No data available to summarize.")


def main():
    race_name = 'Canadian_Grand_Prix'

    # All columns that can be fetched from the database:
    # 'rel_distance', 'time', 'track_coordinate_x', 'track_coordinate_y', 'track_coordinate_z',
    # 'rpm', 'gear', 'throttle', 'brake', 'drs', 'speed', 'acc_x', 'acc_y', 'acc_z'

    # change variables to cluster on here
    # remember to set PCA to false if you are only clustering on one variable
    clusterVariables = ["rpm", "speed"]

    # create RaceData object with the defined columns
    race = RaceData(race_name, clusterVariables)

    drivers = race.drivers

    if USE_PCA:
        data_dict = race.reduced_dict
    else:
        data_dict = race.interp_dict

    # optional: create dendrogram first
    plot_dendrogram(data_dict, drivers, method='ward')

    # create hierarchical cluster
    labels, sil, dbi, cah = hierarchical_cluster(
        data_dict,
        drivers,
        cluster_num=3,
        linkage_method='ward'
    )

    attach_labels(race.interp_dict, drivers, labels)

    # calculate percentage of laps a driver falls into each cluster
    driver_cluster_distribution(race.interp_dict, drivers)

    # calculate the mean values of each cluster
    cluster_mean_telemetry(race.interp_dict, drivers, race.norm_columns)

    # create visual for cluster results
    # visuialize_clusters(race.interp_dict, race.reduced_dict, drivers, use_pca=USE_PCA)

    # print cluster performance metrics
    print(f"\nSilhouette: {sil:.4f}")
    print(f"Davies Bouldin: {dbi:.4f}")
    print(f"Calinski Harabasz: {cah:.4f}")

    # save cluster results to a .csv
    export_clusters_to_csv(race, clusterVariables)

    # save summary of cluster results to a .csv
    export_cluster_summary(race, clusterVariables)


if __name__ == '__main__':
    main()
