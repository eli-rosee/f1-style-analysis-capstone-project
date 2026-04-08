import json
import os
import numpy as np


class ExportLibrary:
    """
    Collects metadata throughout the RaceData + KMeans pipeline and
    writes JSON exports to an organised folder structure:

        exports/
          <RaceName>/
            Track_Summary.json
            Cluster_Overview_Summary.json   (Export 2 - future)
            Cluster_Profiles.json           (Export 3 - future)
            Driver_Distribution/
              <DRIVER>.json                 (Export 4 - future)
    """

    def __init__(self, race_name: str, output_root: str = "../frontend/clustering_results"):
        self.race_name = race_name
        self.output_root = output_root

        # ── drop tracking ──────────────────────────────────────────────
        # Each entry: {"driver": str, "lap": int, "reason": str}
        self._dropped: list[dict] = []

        # Total laps loaded before any dropping (set in record_load)
        self._total_ingested: int = 0

    # ──────────────────────────────────────────────────────────────────
    # Public recording helpers  (called from race_data.py / kmeans)
    # ──────────────────────────────────────────────────────────────────

    def record_load(self, driver_laps: dict[str, int]) -> None:
        """
        Call once after _load() completes.
        driver_laps = race.driver_laps  →  {"HAM": 63, "VER": 57, ...}
        """
        self._total_ingested = sum(driver_laps.values())

    def record_nan_drop(self, driver: str, lap: int) -> None:
        """Call from _reindex() when a lap is skipped due to NaN values."""
        self._dropped.append({
            "driver": driver,
            "lap": lap,
            "reason": "nan_detection",
        })

    def record_outlier_drop(self, driver: str, lap: int, reason: str = "iqr_outlier") -> None:
        """
        Call from _average_speed_check() for each dropped lap.
        reason can be:
          "first_lap"  - always-drop lap 1
          "iqr_outlier" - failed lower-bound speed check
        """
        self._dropped.append({
            "driver": driver,
            "lap": lap,
            "reason": reason,
        })

    # ──────────────────────────────────────────────────────────────────
    # Export 1 – Track Summary
    # ──────────────────────────────────────────────────────────────────

    def export_track_summary(self) -> dict:
        """
        Builds and writes Track_Summary.json.

        Returns the dict so the caller can inspect / log it if needed.
        """
        reasons = {"first_lap": 0, "iqr_outlier": 0, "nan_detection": 0}
        for entry in self._dropped:
            r = entry["reason"]
            if r in reasons:
                reasons[r] += 1
            else:
                # Catch-all bucket for any future reason strings
                reasons[r] = reasons.get(r, 0) + 1

        total_dropped = sum(reasons.values())
        usable = self._total_ingested - total_dropped

        summary = {
            "race": self.race_name,
            "total_laps_ingested": self._total_ingested,
            "usable_laps": usable,
            "dropped_laps": total_dropped,
            "drop_reasons": reasons,
        }

        self._write_json(summary, "Track_Summary.json")
        return summary

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _race_dir(self) -> str:
        """Returns (and creates) the per-race output directory."""
        # Strip spaces so folder names are clean
        clean = self.race_name.replace(" ", "_")
        path = os.path.join(self.output_root, clean)
        os.makedirs(path, exist_ok=True)
        return path

    def _write_json(self, data: dict | list, filename: str) -> None:
        path = os.path.join(self._race_dir(), filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_json_serializer)
        print(f"  Exported → {path}")


# ── JSON serialiser that handles numpy types ──────────────────────────
def _json_serializer(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
