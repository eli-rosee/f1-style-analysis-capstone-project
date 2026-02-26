from data_ingestion.query_db import query_db
import pandas as pd  
import matplotlib.pyplot as plt  

def main():

    telemetry_columns = ['time', 'distance', 'rel_distance', 'track_coordinate_x', 'track_coordinate_y', 'track_coordinate_z', 'rpm', 'gear', 'throttle', 'brake', 'drs', 'speed', 'acc_x', 'acc_y', 'acc_z']
    db = query_db()
    df = db.fetch_driver_telemetry('CAN', 'RUS', telemetry_columns)

    print(df.columns)


if __name__=="__main__":
    main()