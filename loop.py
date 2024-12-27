import pandas as pd
import numpy as np
from datetime import timedelta
from glob import glob
import os

# Define the folder path
folder_path = "/Users/amyxqc/Desktop/amy_bikenycfall2024/Data/CitiBike"

# Define bounding boxes for each borough
borough_bounds = {
    'Manhattan': {'lat_min': 40.70, 'lat_max': 40.88, 'lng_min': -74.02, 'lng_max': -73.90},
    'Brooklyn': {'lat_min': 40.57, 'lat_max': 40.73, 'lng_min': -74.04, 'lng_max': -73.85},
    'Queens': {'lat_min': 40.54, 'lat_max': 40.80, 'lng_min': -73.95, 'lng_max': -73.70},
    'Bronx': {'lat_min': 40.79, 'lat_max': 40.91, 'lng_min': -73.93, 'lng_max': -73.80},
    'Staten Island': {'lat_min': 40.49, 'lat_max': 40.65, 'lng_min': -74.25, 'lng_max': -74.05}
}

def get_borough(lat, lng):
    for borough, bounds in borough_bounds.items():
        if bounds['lat_min'] <= lat <= bounds['lat_max'] and bounds['lng_min'] <= lng <= bounds['lng_max']:
            return borough
    return 'Other'

# Initialize an empty list to store all DataFrames
all_dataframes = []

# Loop through years and months
for year in range(2021, 2025):  # From 2021 to 2024
    for month in range(1, 13):  # From January to December
        if year == 2024 and month > 11:  # Stop at November 2024
            break

        # Construct the file pattern to match multiple parts
        file_pattern = os.path.join(folder_path, f"{year}{month:02d}-citibike-tripdata_*.csv")
        file_list = glob(file_pattern)

        if not file_list:
            print(f"No files found for {year}-{month:02d}. Skipping.")
            continue

        for file_path in file_list:
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Parse date columns explicitly
                df['started_at'] = pd.to_datetime(df['started_at'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
                df['ended_at'] = pd.to_datetime(df['ended_at'], format='%m/%d/%Y %H:%M:%S', errors='coerce')

                # Drop rows with invalid dates
                df = df.dropna(subset=['started_at', 'ended_at'])

                # Ensure all dates are valid
                df = df[df['started_at'].dt.year == year]

                # Extract temporal features
                df['year'] = df['started_at'].dt.year
                df['month'] = df['started_at'].dt.month
                df['week_number'] = df['started_at'].dt.isocalendar().week
                df['day_of_week'] = df['started_at'].dt.weekday

                # Recalculate trip duration
                df['trip_duration'] = (df['ended_at'] - df['started_at']).dt.total_seconds()
                df['trip_duration_min'] = df['trip_duration'] / 60

                # Apply borough mapping
                df['borough'] = df.apply(lambda row: get_borough(row['start_lat'], row['start_lng']), axis=1)

                # Aggregate weekly data
                weekly_aggregated = df.groupby(['year', 'week_number', 'month', 'borough']).agg(
                    total_trips=('ride_id', 'count'),
                    weekday_trips=('day_of_week', lambda x: (x < 5).sum()),
                    weekend_trips=('day_of_week', lambda x: (x >= 5).sum()),
                    electric_bike_rides=('rideable_type', lambda x: (x == 'electric_bike').sum()),
                    classic_bike_rides=('rideable_type', lambda x: (x == 'classic_bike').sum()),
                    member_rides=('member_casual', lambda x: (x == 'member').sum()),
                    casual_rides=('member_casual', lambda x: (x == 'casual').sum()),
                    avg_trip_duration=('trip_duration_min', 'mean'),
                    unique_start_stations=('start_station_id', 'nunique'),
                    unique_end_stations=('end_station_id', 'nunique')
                ).reset_index()

                # Calculate proportions and averages
                weekly_aggregated['electric_bike_proportion'] = weekly_aggregated['electric_bike_rides'] / weekly_aggregated['total_trips']
                weekly_aggregated['classic_bike_proportion'] = weekly_aggregated['classic_bike_rides'] / weekly_aggregated['total_trips']
                weekly_aggregated['member_proportion'] = weekly_aggregated['member_rides'] / weekly_aggregated['total_trips']
                weekly_aggregated['casual_proportion'] = weekly_aggregated['casual_rides'] / weekly_aggregated['total_trips']
                weekly_aggregated['avg_daily_trips'] = weekly_aggregated['total_trips'] / 7
                weekly_aggregated['avg_weekday_trips'] = weekly_aggregated['weekday_trips'] / 5
                weekly_aggregated['avg_weekend_trips'] = weekly_aggregated['weekend_trips'] / 2

                # Add start and end dates for each week
                weekly_aggregated['start_date'] = pd.to_datetime(
                    weekly_aggregated['year'].astype(str) + '-W' + weekly_aggregated['week_number'].astype(str) + '-1',
                    format='%Y-W%U-%w'
                )
                weekly_aggregated['end_date'] = weekly_aggregated['start_date'] + timedelta(days=6)

                # Append to the list of all DataFrames
                all_dataframes.append(weekly_aggregated)

            except FileNotFoundError:
                print(f"File not found: {file_path}. Skipping.")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}. Skipping.")

# Combine all weekly aggregated DataFrames into a single DataFrame
if all_dataframes:
    combined_data = pd.concat(all_dataframes, ignore_index=True)

    # Display the combined DataFrame
    print(combined_data)
else:
    print("No data to combine.")
