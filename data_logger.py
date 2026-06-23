"""
data_logger.py
================
A small, generic utility that sits between the radar simulator and the
feature extractor: it buffers raw measurement rows into a rolling window
and (optionally) appends them to a CSV file. It does NOT compute features
itself -- that is `feature_extraction.py`'s job, and only its job.

Why this file exists
---------------------
V1's `radar_logger.py` mixed THREE responsibilities into one class:
generating measurements, writing CSV rows, AND running feature
extraction + model inference. `behavior_dataset_generator.py` then
duplicated the feature-extraction piece independently. In V2, those
responsibilities are split into three single-purpose files
(`radar_simulator.py` generates, `data_logger.py` buffers/persists,
`feature_extraction.py` extracts), and every script -- training data
generation, test data generation, and the live dashboard -- uses the
exact same `DataLogger` class.

Inputs
------
- Raw measurement dicts (one per tick), each with the keys in
  `config.RADAR_CHANNELS`, via `DataLogger.add(row)`.

Outputs
-------
- `DataLogger.window` -- the most recent `window_size` rows (a rolling
  buffer).
- `DataLogger.window_ready()` -- True once enough rows have accumulated
  for a full feature window.
- If constructed with a `csv_path`, every added row is also appended to
  that CSV file on disk.
"""

import csv
import os

import config


class DataLogger:
    def __init__(self, csv_path=None, window_size: int = config.WINDOW_SIZE):
        self.window_size = window_size
        self.window: list[dict] = []
        self.csv_path = csv_path
        self._csv_file = None
        self._csv_writer = None

        if self.csv_path is not None:
            write_header = not os.path.exists(self.csv_path)
            self._csv_file = open(self.csv_path, "a", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=["timestamp"] + config.RADAR_CHANNELS
            )
            if write_header:
                self._csv_writer.writeheader()

    def add(self, row: dict) -> None:
        """Append one raw measurement row. `row` must include 'timestamp'
        plus every key in config.RADAR_CHANNELS."""
        self.window.append(row)
        if len(self.window) > self.window_size:
            self.window.pop(0)

        if self._csv_writer is not None:
            self._csv_writer.writerow(row)

    def window_ready(self) -> bool:
        return len(self.window) == self.window_size

    def reset_window(self) -> None:
        self.window = []

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
