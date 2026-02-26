from data_ingestion.postgresql_db import telemetry_database
import numpy as np
import pandas as pd
import ast

class query_db():

    #constructor
    def __init__(self):
        self.db = telemetry_database()
    
    def fetch_driver_laps(self, race_name, driver_name):
        self.db.cursor.execute(f"SELECT lap FROM race_lap_data WHERE driver_id = '{driver_name}' AND race_name = '{race_name}';")

        records = self.db.cursor.fetchall()
        records_refined = set()

        for record in records:
            records_refined.add(record[0])

        print()
        print(f"Race: {race_name}")
        print(f"Driver: {driver_name}")     
        print("Laps driven:")   
        print(records_refined)
        print()

    #HELPER FUNCTION, DO NOT CALL
    #returns a string containing the tel_index value for the specified driver on the specified race
    def _fetch_driver_metadata(self, race_name, driver_name):
        self.db.cursor.execute(f"SELECT tel_index FROM race_lap_data WHERE driver_id = '{driver_name}' AND race_name = '{race_name}';")

        # Fetch all the results
        records = self.db.cursor.fetchall()
        records_refined = []

        for record in records:
            records_refined.append(record[0])
            
        output_string = ', '.join(map(str, records_refined))
        return output_string
    
    #Returns a pandas dataframes given a race_name, driver_name, specified telemetry_columns
    #Returns data for an ENTIRE race
    def fetch_driver_telemetry(self, race_name, driver_name, telemetry_column):
        tel_index_list = self._fetch_driver_metadata(race_name, driver_name)
        telemetry_column_string = ', '.join(map(str, telemetry_column))
        self.db.cursor.execute(f"SELECT {telemetry_column_string} FROM telemetry_data WHERE tel_index IN ({tel_index_list});")

        records = self.db.cursor.fetchall()

        if not records:
            print(f"No telemetry data found for {driver_name} at {race_name}")
            return pd.DataFrame(columns=telemetry_column)

        df = pd.DataFrame(records, columns=telemetry_column)
        print("\nReturning Pandas DataFrame of requested data")
        return df
    
    #GOOD for Clustering
    #Returns a pandas dataframes given a race_name, driver_name, specified telemetry_columns and lap number
    #telemetry_column must be a list of strings - see main func for acceptable column entries
    def fetch_driver_telemetry_by_lap(self, race_name, driver_name, telemetry_column, lap_num):
        self.db.cursor.execute(f"SELECT tel_index FROM race_lap_data WHERE driver_id = '{driver_name}' AND race_name = '{race_name}' AND lap = '{lap_num}';")

        # Fetch all the results
        records = self.db.cursor.fetchall()
        records_refined = []
        
        for record in records:
            records_refined.append(record[0])
            
        tel_index_list = ', '.join(map(str, records_refined))

        telemetry_column_string = ', '.join(map(str, telemetry_column))
        self.db.cursor.execute(f"SELECT {telemetry_column_string} FROM telemetry_data WHERE tel_index IN ({tel_index_list});")

        records = self.db.cursor.fetchall()

        if not records:
            print(f"No telemetry data found for {driver_name}, lap {lap_num} at {race_name}")
            return pd.DataFrame(columns=telemetry_column)

        df = pd.DataFrame(records, columns=telemetry_column)
        print("\nReturning Pandas DataFrame of requested data")
        return df

    
def main():
    db = query_db()

    #acceptable columns for queries on telemetry data
    telemetry_columns = ['time', 'distance', 'rel_distance', 'track_coordinate_x', 'track_coordinate_y', 'track_coordinate_z', 'rpm', 'gear', 'throttle', 'brake', 'drs', 'speed', 'acc_x', 'acc_y', 'acc_z']

    race_code_map = {"Belgian_Grand_Prix" : "BEL", "Chinese_Grand_Prix" : "CHN", "Hungarian_Grand_Prix" : "HUN", "Japanese_Grand_Prix" : "JPN", "Dutch_Grand_Prix" : "NED",
                    "Bahrain_Grand_Prix" : "BAH", "Italian_Grand_Prix" : "ITA", "Saudi_Arabian_Grand_Prix" : "SAU", "Azerbaijan_Grand_Prix" : "AZE", "Miami_Grand_Prix" : "MIA",
                    "Singapore_Grand_Prix" : "SIN", "Emilia_Romagna_Grand_Prix" : "EMI", "United_States_Grand_Prix" : "USA", "Monaco_Grand_Prix" : "MON", "Mexico_City_Grand_Prix" : "MEX",
                    "Spanish_Grand_Prix" : "ESP", "São_Paulo_Grand_Prix" : "SAO", "Canadian_Grand_Prix" : "CAN", "Las_Vegas_Grand_Prix" : "LAS", "Australian_Grand_Prix" : "AUS",
                    "Qatar_Grand_Prix" : "QAT", "British_Grand_Prix" : "GBR", "Abu_Dhabi_Grand_Prix" : "ABU",
                    }

    tel_select_cols = [telemetry_columns[i] for i in [1]]

    for race in race_code_map.keys():
        print("RACE: " + race)
        df = db.fetch_driver_telemetry(race_code_map[race], "RUS", tel_select_cols)
        print(df)

    
if __name__ == "__main__":
    main()
