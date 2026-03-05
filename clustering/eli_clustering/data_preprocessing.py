# from data_ingestion.query_db import query_db
import pandas as pd  
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

## Variable declaration. Can be done with fastf1, but this is far easier at the moment
telemetry_columns = ['rel_distance', 'time', 'track_coordinate_x', 'track_coordinate_y', 'track_coordinate_z', 'rpm', 'gear', 'throttle', 'brake', 'speed', 'acc_x', 'acc_y', 'acc_z']
normalized_telemetry_columns = ['rpm', 'gear', 'throttle', 'speed', 'acc_x', 'acc_y', 'acc_z']
canadian_gp_drivers = ['RUS', 'VER', 'ANT', 'PIA', 'LEC', 'HAM', 'ALO', 'HUL', 'OCO', 'SAI', 'BEA', 'TSU', 'COL', 'BOR', 'GAS', 'HAD', 'STR', 'NOR', 'LAW', 'ALB']
canadian_gp_driver_laps = {'RUS': 70, 'VER': 70, 'ANT': 70, 'PIA': 70, 'LEC': 70, 'HAM': 70, 'ALO': 70, 'HUL': 70, 'OCO': 69, 'SAI': 69, 'BEA': 69, 'TSU': 69, 'COL': 69, 'BOR': 69, 'GAS': 69, 'HAD': 69, 'STR': 69, 'NOR': 66, 'LAW': 53, 'ALB': 46}
canadian_gp_drivers_df_dict = {'RUS': [], 'VER': [], 'ANT': [], 'PIA': [], 'LEC': [], 'HAM': [], 'ALO': [], 'HUL': [], 'OCO': [], 'SAI': [], 'BEA': [], 'TSU': [], 'COL': [], 'BOR': [], 'GAS': [], 'HAD': [], 'STR': [], 'NOR': [], 'LAW': [], 'ALB': []}
interpolated_laps_df_dict = {'RUS': [], 'VER': [], 'ANT': [], 'PIA': [], 'LEC': [], 'HAM': [], 'ALO': [], 'HUL': [], 'OCO': [], 'SAI': [], 'BEA': [], 'TSU': [], 'COL': [], 'BOR': [], 'GAS': [], 'HAD': [], 'STR': [], 'NOR': [], 'LAW': [], 'ALB': []}
labels = []

datapoints_per_lap = 500
# db = query_db()
max_dict = {}
min_dict = {}


## Function used for querying the database and storing the dataframes locally. EXTREMELY SLOW AT THE MOMENT
def query_and_df_creation():

    for driver in canadian_gp_drivers:
        lap_count = canadian_gp_driver_laps[driver]
        for lap in range(1, lap_count + 1):
            driver_df = db.fetch_driver_telemetry_by_lap('CAN', driver, telemetry_columns, lap_num=lap)
            driver_df.set_index('rel_distance', inplace=True)
            driver_df['brake'] = driver_df['brake'].astype(int)

            canadian_gp_drivers_df_dict[driver].append(driver_df)
            driver_df.to_pickle(f'pandas_df/{driver}{lap}')

## Finds the maximum / minimum values of all metrics for normalization purposes. Looks at every driver, every lap
def get_max_values():
    for driver in canadian_gp_drivers_df_dict.keys():

        for i in range(canadian_gp_driver_laps[driver]):
            driver_df = canadian_gp_drivers_df_dict[driver][i]
            
            for column in normalized_telemetry_columns:
                col_max = driver_df[column].max()
                col_min = driver_df[column].min()
                max_dict[column] = max(max_dict.get(column, -np.inf), col_max)
                min_dict[column] = min(min_dict.get(column, np.inf), col_min)

## Repopulates the dataframes dict if they are stored locally
def pickle_df_repop():
    print("Repopulating dataframes from pickled files...")
    for driver in canadian_gp_drivers:
        for i in range(1, canadian_gp_driver_laps[driver] + 1):
            temp_driver_df = pd.read_pickle(f'pandas_df/{driver}{i}')
            canadian_gp_drivers_df_dict[driver].append(temp_driver_df)

## Normalization of the data / interpolation is performed
def reindex():
    print("Reindexing and interpolating dataframes...")

    for driver in interpolated_laps_df_dict.keys():
        for i in range(canadian_gp_driver_laps[driver]):
            driver_df = canadian_gp_drivers_df_dict[driver][i]

            uniform_index = np.linspace(0, 1, datapoints_per_lap)
            driver_df = canadian_gp_drivers_df_dict[driver][i]

            driver_df = driver_df[~driver_df.index.duplicated(keep='first')]

            # Combine original index with uniform grid
            combined_index = driver_df.index.union(uniform_index)

            # Reindex to combined index (inserts NaNs at new grid points)
            driver_df = driver_df.reindex(combined_index)

            # Interpolate to fill in the NaNs
            driver_df = driver_df.interpolate(method='index')
            driver_df = driver_df.ffill()
            driver_df = driver_df.bfill()

            # Reindex down to only the uniform grid points
            driver_df = driver_df.reindex(uniform_index)

            # Round gear to nearest integer
            driver_df['gear'] = driver_df['gear'].round()
            driver_df['gear'] = driver_df['gear'].astype(int)

            interpolated_laps_df_dict[driver].append(driver_df)

def normalize():
    print("Normalizing dataframes...")
    for driver in interpolated_laps_df_dict.keys():
        for i in range(canadian_gp_driver_laps[driver]):
            driver_df = interpolated_laps_df_dict[driver][i]

            for column in normalized_telemetry_columns:
                col_max = max_dict[column]
                col_min = min_dict[column]
                driver_df[column] = (driver_df[column] - col_min) / (col_max - col_min)

def cluster():    
    print("Clustering data...")
    # Stack all laps from all drivers into one matrix
    all_laps = []
    for driver in interpolated_laps_df_dict.keys():
        for lap_df in interpolated_laps_df_dict[driver]:
            all_laps.append(lap_df[normalized_telemetry_columns].values.flatten())
    
    X = np.array(all_laps)  # shape: (total_laps, 500 * 7)
    
    # Fit KMeans
    km = KMeans(n_clusters=4, random_state=42)
    labels = km.fit_predict(X)
    
    return labels

def attach_labels(labels):
    print("Attaching cluster labels to dataframes...")
    print(labels)
    # This function will attach the cluster labels back to the original dataframes for analysis    
    lap_index = 0
    for driver in interpolated_laps_df_dict.keys():
        for lap_df in interpolated_laps_df_dict[driver]:
            lap_df['cluster_label'] = labels[lap_index]
            lap_index += 1

def elbow_plot():
    print("Generating elbow plot...")
    all_laps = []
    for driver in interpolated_laps_df_dict.keys():
        for lap_df in interpolated_laps_df_dict[driver]:
            all_laps.append(lap_df[normalized_telemetry_columns].values.flatten())
    
    X = np.array(all_laps)
    
    wcss = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        wcss.append(km.inertia_)
    
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.savefig('elbow_plot.png')

def visualize_clusters():
    features = normalized_telemetry_columns
    
    # mean telemetry curve per cluster, per feature
    cluster_data = {}
    for driver in interpolated_laps_df_dict.keys():
        for lap_df in interpolated_laps_df_dict[driver]:
            label = lap_df['cluster_label'].iloc[0]
            if label not in cluster_data:
                cluster_data[label] = []
            cluster_data[label].append(lap_df[features].values)
    
    fig, axes = plt.subplots(len(features), 1, figsize=(12, 20))
    for i, feature in enumerate(features):
        for label, laps in cluster_data.items():
            mean_curve = np.mean([lap[:, i] for lap in laps], axis=0)
            axes[i].plot(mean_curve, label=f'Cluster {label}')
        axes[i].set_title(feature)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('cluster_visualization.png')

def cluster_statistics():
    print("\n========== CLUSTER STATISTICS ==========")
    
    # Build a flat list of (driver, lap_num, cluster_label)
    lap_records = []
    for driver in interpolated_laps_df_dict.keys():
        for lap_num, lap_df in enumerate(interpolated_laps_df_dict[driver]):
            label = lap_df['cluster_label'].iloc[0]
            lap_records.append({'driver': driver, 'lap': lap_num + 1, 'cluster': label})
    
    records_df = pd.DataFrame(lap_records)

    # Per cluster: count + mean telemetry stats
    for cluster_id in sorted(records_df['cluster'].unique()):
        cluster_laps = [
            interpolated_laps_df_dict[row['driver']][row['lap'] - 1]
            for _, row in records_df[records_df['cluster'] == cluster_id].iterrows()
        ]
        
        count = len(cluster_laps)
        total = len(records_df)
        all_data = pd.concat(cluster_laps)[normalized_telemetry_columns]
        
        print(f"\n--- Cluster {cluster_id} ---")
        print(f"  Lap count : {count} ({100 * count / total:.1f}% of all laps)")
        print(f"  Mean telemetry:")
        for col in normalized_telemetry_columns:
            print(f"    {col:<22} mean={all_data[col].mean():.3f}  std={all_data[col].std():.3f}")

    # Per driver: % of laps in each cluster
    print("\n========== DRIVER CLUSTER BREAKDOWN ==========")
    cluster_ids = sorted(records_df['cluster'].unique())
    header = f"{'DRIVER':<8}" + "".join(f"  C{c}%" for c in cluster_ids)
    print(header)
    
    for driver in canadian_gp_drivers:
        driver_laps = records_df[records_df['driver'] == driver]
        total = len(driver_laps)
        row = f"{driver:<8} ({total} laps)"
        for c in cluster_ids:
            pct = 100 * len(driver_laps[driver_laps['cluster'] == c]) / total
            row += f"  {pct:>4.1f}"
        print(row)

def main():

    # query_and_df_creation()
    pickle_df_repop()
    get_max_values()
    reindex()
    normalize()
    labels = cluster()
    attach_labels(labels)
    # elbow_plot()
    visualize_clusters()
    cluster_statistics()

if __name__=="__main__":
    main()
