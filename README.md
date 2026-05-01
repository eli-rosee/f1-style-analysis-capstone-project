# F1-Style-Analysis-Capstone-Project

## Overview
Formula 1 is one of the most data-driven sports in the world, yet one key factor remains difficult to measure: the driver. This project aims to determine if there is a certain way of driving the vehicle (or “driving style”) that produces better results. In this project, we analyze high-frequency telemetry data to identify and classify different driving styles by aligning laps, normalizing per driver, and clustering lap-level behavior. The results are then presented through an interactive web dashboard that allows users to explore and compare driving style patterns across drivers and tracks.

## Architecture
1) Data Processing Layer

    A Python scraper is used to ingest F1 telemetry data from TracingInsights.com. The data is downloaded as a JSON file and stored into a PostgreSQL database.

2) Analytics and Clustering Layer

    Principal Component Analysis (PCA) via scikit-learn reduces the dimensionality of the data and feeds it into a K-means clustering algorithm. The clustering algorithm will attempt to seperate drivers into different clusters or "driving styles." Cluster quality is evaluated using silhouette scores and the Davies-Bouldin Index. The results from the K-means clustering are stored as JSON files to be used by the web dashboard.

3) Visualization and Dashboard Layer

    A JavaScript single-page application served via nginx on an Amazon EC2 instance. The data is visualized with Chart.js and Leaflet.

## Installation
This project utilizes Python and several Python libraries. In the same folder as requirements.txt, run the following command to ensure that all the required libraries are installed:
###
    pip install -r requirements.txt

## Features
The codebase is divided into three main folders, each containing code for a specific stage of the data pipeline.
- data_ingestion
  - Data Scraping (from TracingInsights.com)
  - Storing data in a PostgreSQL database
- data_analysis
  - Normalization
    - Outlier Detection (IQR)
    - Principal Component Analysis (PCA)
  - KMeans Clustering
  - Saving Clustering results as JSON files
- frontend
    - Web Front End
    - Data Visualization
