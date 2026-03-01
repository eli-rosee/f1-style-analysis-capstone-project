from clustering.nick_clustering.kmeans_clustering import raceName, driverName, dataPoint

import pandas as pd
import os

#insert filepath of csv you want to read
#raceName, driverName, and dataPoint are imported from kmeans_clustering.py so you don't have to manually change the filepath
input_csv = f"clustering/nick_clustering/{raceName}_{driverName}_{dataPoint}_cluster_results.csv"

#file path of csv which will be output by the program
output_csv = f"clustering/nick_clustering/{raceName}_{driverName}_{dataPoint}_cluster_summary.csv"

#make sure file exists before reading it
if not os.path.exists(input_csv):
    raise Exception(f"Error: Could not find the file '{input_csv}'.")
    
#load data into a pandas DataFrame
df = pd.read_csv(input_csv)

#calculate total number of data points
total_datapoints = len(df)

#group by cluster and calculate multiple stats at once
summary_stats = df.groupby('cluster_label')[f'{dataPoint}'].agg(['count', 'min', 'max', 'mean']).reset_index()

#calculate the percentage of the lap spent in each cluster
summary_stats['percentage'] = (summary_stats['count'] / total_datapoints) * 100

#add metadata columns
summary_stats['race'] = raceName
summary_stats['driver'] = driverName

#rename the columns to be descriptive
summary_stats = summary_stats.rename(columns={
    'cluster_label': 'Cluster',
    'min': 'Min',
    'max': 'Max',
    'mean': 'Avg',
    'percentage': 'Percentage_of_Lap'
})

#reorder the columns to match exactly what you requested
final_summary_df = summary_stats[['race', 'driver', 'Cluster', 'Percentage_of_Lap', 'Min', 'Max', 'Avg']]

#print results to the screen
print("\n")
print("--- K-Means Clustering Summary ---")
print(f"Race: {raceName} | Driver: {driverName}")
print(f"Total Datapoints: {total_datapoints:,}") #the :, adds comma for the number
print("-" * 60)


#iterate through the summary dataframe to print the stats
for index, row in final_summary_df.iterrows():
    print(f"Cluster {int(row['Cluster'])}: {row['Percentage_of_Lap']:.1f}% of data")
    print(f"  -> Range: {row['Min']:.0f} to {row['Max']:.0f} (Avg: {row['Avg']:.0f})")

print("-" * 60)


#----------------------------------------------------------
#Save analysis to a .csv file

# We use index=False so we don't save the arbitrary row numbers (0, 1, 2)
final_summary_df.to_csv(output_csv, index=False)
print("\n")
print(f"Summary saved to: {output_csv}")