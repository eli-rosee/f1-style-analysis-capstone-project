import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt
from data_ingestion.query_db import query_db
import pandas as pd

def main():
    telemetry_columns = ['time', 'distance', 'rel_distance', 'track_coordinate_x', 'track_coordinate_y', 'track_coordinate_z', 'rpm', 'gear', 'throttle', 'brake', 'drs', 'speed', 'acc_x', 'acc_y', 'acc_z']
    race_code_map = {"Belgian_Grand_Prix" : "BEL", "Chinese_Grand_Prix" : "CHN", "Hungarian_Grand_Prix" : "HUN", "Japanese_Grand_Prix" : "JPN", "Dutch_Grand_Prix" : "NED",
                    "Bahrain_Grand_Prix" : "BAH", "Italian_Grand_Prix" : "ITA", "Saudi_Arabian_Grand_Prix" : "SAU", "Azerbaijan_Grand_Prix" : "AZE", "Miami_Grand_Prix" : "MIA",
                    "Singapore_Grand_Prix" : "SIN", "Emilia_Romagna_Grand_Prix" : "EMI", "United_States_Grand_Prix" : "USA", "Monaco_Grand_Prix" : "MON", "Mexico_City_Grand_Prix" : "MEX",
                    "Spanish_Grand_Prix" : "ESP", "São_Paulo_Grand_Prix" : "SAO", "Canadian_Grand_Prix" : "CAN", "Las_Vegas_Grand_Prix" : "LAS", "Australian_Grand_Prix" : "AUS",
                    "Qatar_Grand_Prix" : "QAT", "British_Grand_Prix" : "GBR", "Abu_Dhabi_Grand_Prix" : "ABU",
                    }

    canadian_gp_drivers = [
        "RUS", "VER", "ANT", "PIA", "LEC", "HAM", "ALO", "HUL",
        "OCO", "SAI", "BEA", "TSU", "COL", "BOR", "GAS", "HAD",
        "STR", "NOR", "LAW", "ALB"
    ]

    ## 2500 - 2900

    db = query_db()
    tel_select_cols = [telemetry_columns[i] for i in [1, 3, 4]]

    df = db.fetch_driver_telemetry_by_lap(race_code_map["Canadian_Grand_Prix"], "RUS", tel_select_cols, 70)

    print(df)

    plt.scatter(df['track_coordinate_x'], df['track_coordinate_y'], c=df['distance'], cmap='viridis')
    plt.colorbar(label='Distance')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Distance Metric Over Space')
    plt.savefig("canadian_gp_rus_lap70.png", dpi=300)
    
if __name__=="__main__":
    main()