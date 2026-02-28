from data_ingestion.query_db import query_db

import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


#used to determine the best k value
def performElbowMethod():
    plt.figure() # Create fresh canvas
    sse = []

    #run kmean multiple times, record the sse (error) value for each iteration
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)

    #visualization, plot K Value vs SEE
    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")

    #save as a png
    plt.savefig('elbow_plot.png', dpi=300, bbox_inches='tight')
    
    #use knee locator to determine the best k value
    kl = KneeLocator(
        range(1, 11), sse, curve="convex", direction="decreasing"
    )

    #print knee locator result to the screen
    print(kl.elbow)

#used to determine the best k value
def performSilhouette():
    plt.figure() # Create fresh canvas

    #List holds the silhouette coefficients for each k
    silhouette_coefficients = []

    #Must start at 2 clusters for silhouette coefficient or it will throw an error
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        score = silhouette_score(scaled_features, kmeans.labels_, sample_size=2000)
        silhouette_coefficients.append(score)
    
    #Visualization, Plot number of clusters vs silhouette coefficient
    plt.style.use("fivethirtyeight")
    plt.plot(range(2, 11), silhouette_coefficients)
    plt.xticks(range(2, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")

    #save as a png
    plt.savefig('silhouette_coefficient.png', dpi=300, bbox_inches='tight')



#Get data from the database
telColumns = ["rpm"]
queryDB = query_db()

#get a list of pandas dataframes
telemetryData = queryDB.fetch_driver_telemetry("CAN", "HAM", telColumns)

#extract features from giant dataframe and drop missing values
featuresRaw = telemetryData[telColumns].dropna()

#Feature scaling using standardization
scaler = StandardScaler()
scaled_features = scaler.fit_transform(featuresRaw)

#KMEANS Clustering

#Instantiate the Kmeans class
kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10, #number of kmeans runs to preform. Run with lowest SSE value is returned
    max_iter=300, #max number of iterations per run
    random_state=42
)

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

#analysis to determine the best k value
#performElbowMethod()
#performSilhouette()

#run the kmeans clustering
kmeans.fit(scaled_features)

#The lowest SSE value
kmeans.inertia_

#Final locations of the centroid
kmeans.cluster_centers_

#The number of iterations required to converge
kmeans.n_iter_

#data visualization

# 1. Create a larger, cleaner figure
plt.figure(figsize=(12, 7))
plt.style.use("fivethirtyeight")

# 2. Scatter plot optimized for 41,000 points
# We use s=1 and alpha=0.1 so that darker areas represent more frequent RPM levels
plt.scatter(
    range(len(scaled_features)), 
    scaled_features, 
    c=kmeans.labels_, 
    cmap='viridis', 
    s=1,            # Very small points to prevent a "blurry blob"
    alpha=0.1,      # High transparency to show data density
    rasterized=True # Helps keep the file size manageable
)

# 3. Better Centroid Visualization
# Since this is 1D data (RPM only), centroids are "levels." 
# Drawing them as horizontal lines shows the "RPM thresholds" found by K-means.
for i, center in enumerate(kmeans.cluster_centers_):
    plt.axhline(
        y=center, 
        color='red', 
        linestyle='--', 
        linewidth=2, 
        alpha=0.8,
        label=f'Centroid {i}' if i == 0 else "" # Only label the first for the legend
    )

# 4. Labels and Formatting
plt.title("K-Means Clustering: Hamilton RPM Distribution (Canada GP)")
plt.xlabel("Time Sequence (Data Point Index)")
plt.ylabel("Standardized RPM")
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.5)

# 5. Save with high resolution
plt.savefig('clustering/nick_clustering/telemetry_clusters.png', dpi=300, bbox_inches='tight')
print("Plot saved to 'clustering/nick_clustering/telemetry_clusters.png'")