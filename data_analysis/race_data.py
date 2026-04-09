import json
import numpy as np
import pandas as pd
from query_db import TelemetryDatabase
from export_library import ExportLibrary
from sklearn.decomposition import PCA

# Have to run build_metadata_cache.py
CACHE_FILE = 'cache/race_metadata_cache.json'

# Load race schedule and driver lap counts from the JSON cache
# Cache information stored in _cache variable
with open(CACHE_FILE) as f:
    _cache = json.load(f)

# Extracts the valuable race metadata from _cache
RACES = _cache['races']

# Retrieves the list of the racenames in the 2025 F1 Season (e.g., ['Australian_Grand_Prix', 'Chinese_Grand_Prix', ...])
RACE_NAMES = list(RACES.keys())

# DRIVER_LAPS['Australian_Grand_Prix']['HAM'] = Lewis Hamiltons laps raced in the Australian Grand Prix
DRIVER_LAPS = {}
for race in RACE_NAMES:
    DRIVER_LAPS[race] = RACES[race]['driver_laps']

# DRIVERS['Australian_Grand_Prix'] returns the list of drivers that raced in the Australian_Grand_Prix
DRIVERS = {}
for race in RACE_NAMES:
    DRIVERS[race] = list(RACES[race]['driver_laps'].keys())

# All columns that can be fetched from the database
TEL_COLUMNS = ['rel_distance', 'time', 'track_coordinate_x', 'track_coordinate_y', 'track_coordinate_z', 'rpm', 'gear', 'throttle', 'brake', 'drs', 'speed', 'acc_x', 'acc_y', 'acc_z']

# Default columns to normalize. Can be overridden by the user at init
NORM_TEL_COLUMNS = ['rpm', 'gear', 'throttle', 'speed', 'acc_x', 'acc_y', 'acc_z']

# Number of evenly spaced points each lap is interpolated to
DATAPOINTS_PER_LAP = 500


class RaceData:

    # Pass a race name to load, process, and normalize all telemetry for that race.
    # Optionally pass norm_columns to normalize a custom set of columns.
    def __init__(self, race_name, norm_columns=None):
        self.race_name = race_name
        self.drivers = DRIVERS[race_name]
        self.driver_laps = DRIVER_LAPS[race_name]

        # Initializes postgresdb connection (see query_db.py)
        self.db = TelemetryDatabase()

        self.exporter = ExportLibrary(race_name)

        self.max_dict = {}
        self.min_dict = {}

        # Checks to see if the user passed a list of columns they want normalized. If not, use default
        self.norm_columns = NORM_TEL_COLUMNS
        if norm_columns:
            self.norm_columns = norm_columns

        # Stores the raw telemetry from the database, stored in order of laps.
        # e.g., df_dict['HAM'] retrieves all telemetry dataframes in a list for laps 1, 2, 3, ...
        self.df_dict = {}
        for driver in self.drivers:
            self.df_dict[driver] = []

        # Stores the interpolated telemetry, after it has been translated to a grid for euclidean distance comparison (see _reindex_df_operations)
        self.interp_dict = {}
        for driver in self.drivers:
            self.interp_dict[driver] = []

        # Stores the telemetry dataframes AFTER PCA transformation has been applied to it
        self.reduced_dict = {}
        for driver in self.drivers:
            self.reduced_dict[driver] = []

        # Loads the data from the database, stores it in the df_dict, in order of lap
        self._load()

        self.exporter.record_load(self.driver_laps)

        # Gets the GLOBAL min max (depreciated, we now use local min maxes through _get_min_max)
        self._get_min_max()

        # Reindexes all the telemetry onto a grid of DATAPOINTS_PER_LAP values
        # e.g., if DATAPOINTS_PER_LAP = 500, the index will be interpolated to 500 evenly spaced points by relative distance
        self._reindex()

        # Normalizes the telemetry to a 0-1 scale. This is done with _get_min_max_driver_lap integrated into it.
        self._normalize()

        # Checks for laps that are too slow, and drops them from the dict.
        self._average_speed_check()

        # Applies PCA algorithm to the interpolated and normalized DFs
        self.pca()

    # Returns normalized laps for a single driver, or all drivers if none specified
    def get(self, driver=None):
        if driver:
            return self.interp_dict[driver]
        return self.interp_dict

    # Fetches all laps for all drivers from the database and stores them in df_dict
    def _load(self):
        print(f"Loading data for {self.race_name}...")
        for driver in self.drivers:
            print(f"  Loading {driver}...")
            for lap in range(1, self.driver_laps[driver] + 1):
                df = self.db.fetch_driver_telemetry_by_lap(self.race_name, driver, TEL_COLUMNS, lap_num=lap)
                df.set_index('rel_distance', inplace=True)
                df['brake'] = df['brake'].astype(int)
                df['drs'] = df['drs'].astype(int)
                self.df_dict[driver].append(df)

    # CHUNKS OF 5 TO 10, 0-5, 0-10
    # Finds the global min/max for each column across all drivers and laps, used for normalization
    def _get_min_max(self):
        for driver in self.drivers:
            for df in self.df_dict[driver]:
                for col in self.norm_columns:
                    if df[col].count() == 0:
                        continue
                    self.max_dict[col] = max(self.max_dict.get(col, -np.inf), np.percentile(df[col], 98))
                    self.min_dict[col] = min(self.min_dict.get(col, np.inf), np.percentile(df[col], 2))

    def _get_min_max_driver_lap(self, driver, laps_min, laps_max):
        min_dict, max_dict = {}, {}

        if(len(self.df_dict[driver]) < laps_max):
            laps_max = len(self.df_dict[driver])
            laps_min = laps_max - 5

        for df in self.df_dict[driver][laps_min:laps_max]:
            for col in self.norm_columns:
                if df[col].count() == 0:
                    continue
                max_dict[col] = max(max_dict.get(col, -np.inf), np.nanpercentile(df[col], 98))
                min_dict[col] = min(min_dict.get(col, np.inf), np.nanpercentile(df[col], 2))

        return min_dict, max_dict

    def _reindex_df_operations(self, df):
        uniform_index = np.linspace(0, 1, DATAPOINTS_PER_LAP)
        df = df[~df.index.duplicated(keep='first')]
        df = df.reindex(df.index.union(uniform_index))
        df = df.infer_objects(copy=False)
        df = df[df.index.notna()]
        df = df.interpolate(method='index')
        df = df.ffill()
        df = df.bfill()
        df = df.reindex(np.linspace(0, 1, DATAPOINTS_PER_LAP))

        return df

    # Interpolates each lap onto a uniform grid of DATAPOINTS_PER_LAP points between 0 and 1
    def _reindex(self):
        print("Reindexing...")
        for driver in self.drivers:
            for lap_num, df in enumerate(self.df_dict[driver], start=1):
                df = self._reindex_df_operations(df)

                if df.isna().sum().sum() > 0:
                    print(f"  Dropping {driver} lap {lap_num} — NaN values detected")
                    self.exporter.record_nan_drop(driver, lap_num)
                    continue

                df['gear'] = df['gear'].fillna(0).round().astype(int)
                self.interp_dict[driver].append(df)

    # Applies min-max normalization to each column in norm_columns
    def _normalize(self):
        print("Normalizing...")
        for driver in self.drivers:
            print(f"  Normalizing {driver}...")
            prev_lap_increment = 0
            min_dict, max_dict = {}, {}
            for lap_num, df in enumerate(self.interp_dict[driver], start=1):
                lap_increment = lap_num + (5 - lap_num % 5)

                if lap_increment != prev_lap_increment:
                    min_dict, max_dict = self._get_min_max_driver_lap(driver, lap_increment - 5, lap_increment)
                    prev_lap_increment = lap_increment

                for col in self.norm_columns:
                    denom = max_dict[col] - min_dict[col]
                    if denom == 0 or np.isnan(denom):
                        df[col] = 0
                    else:
                        df[col] = (df[col] - min_dict[col]) / denom
                    df[col] = np.clip(df[col], 0, 1)

    def _average_speed_check(self, iqr_multiplier=1.5):
        print("Checking speed thresholds (lower tail IQR)...")

        for driver in self.drivers:
            filtered_laps = []
            lap_speeds = [np.mean(df['speed']) for df in self.interp_dict[driver]]

            if len(lap_speeds) < 4:
                continue

            q1 = np.percentile(lap_speeds, 25)
            q3 = np.percentile(lap_speeds, 75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr

            filtered_laps = []

            for lap_num, df in enumerate(self.interp_dict[driver], start=1):
                avg_speed = np.mean(df['speed'])

                if lap_num == 1:
                    print(f"  Dropping {driver} lap {lap_num} — outlier")
                    self.exporter.record_outlier_drop(driver, lap_num, reason="first_lap")

                elif avg_speed < lower_bound:
                    print(f"  Dropping {driver} lap {lap_num} — outlier")
                    self.exporter.record_outlier_drop(driver, lap_num, reason="iqr_outlier")

                else:
                    filtered_laps.append(df)

            self.interp_dict[driver] = filtered_laps

    def pca(self, n_components=0.95):
        all_points = []

        for driver in self.drivers:
            for lap_df in self.interp_dict[driver]:
                all_points.append(lap_df[self.norm_columns].values)

        X = np.vstack(all_points)

        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X)

        print("\nApplying PCA...")
        print(f"  Components created: {pca.n_components_}")
        print(f"  Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

        for driver in self.drivers:
            for lap_df in self.interp_dict[driver]:
                X_lap = lap_df[self.norm_columns].values
                X_reduced = pca.transform(X_lap)
                self.reduced_dict[driver].append(X_reduced)

if __name__ == '__main__':
    race = RaceData('Canadian_Grand_Prix')

    pca, X_reduced = race.pca()
    
