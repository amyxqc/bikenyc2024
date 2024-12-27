# Data Processing for One File

import pandas as pd
import numpy as np
from datetime import timedelta

# File path to the data
file_path = "/Users/amyxqc/Desktop/amy_bikenycfall2024/Data/CitiBike/202101-citibike-tripdata_1.csv"

# Define column types explicitly (adjust types as needed)
column_types = {
    'ride_id': str,
    'start_lat': float,
    'start_lng': float,
    'end_lat': float,
    'end_lng': float,
    'rideable_type': str,
    'member_casual': str,
    'start_station_id': str,
    'end_station_id': str
}

# Read the CSV file with explicit types and suppress memory warnings
df = pd.read_csv(file_path, dtype=column_types, low_memory=False)

# Parse date columns explicitly
df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce')

# Drop rows with invalid or missing dates
df = df.dropna(subset=['started_at', 'ended_at'])

# Ensure all dates are in 2021
df = df[df['started_at'].dt.year == 2021]

# Recalculate trip duration
df['trip_duration'] = (df['ended_at'] - df['started_at']).dt.total_seconds()
df['trip_duration_min'] = df['trip_duration'] / 60

# Extract temporal features
df['year'] = df['started_at'].dt.year
df['month'] = df['started_at'].dt.month
df['day_of_week'] = df['started_at'].dt.weekday  # 0=Monday, 6=Sunday

# Calculate week numbers
df['week_number'] = ((df['started_at'] - pd.Timestamp(f'{df["year"].iloc[0]}-01-01')).dt.days // 7 + 1)

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

# Apply borough mapping
df['borough'] = df.apply(lambda row: get_borough(row['start_lat'], row['start_lng']), axis=1)

# Aggregate weekly data
weekly_aggregated = df.groupby(['year', 'week_number', 'month', 'borough']).agg(
    total_trips=('ride_id', 'count'),  # Total trips
    weekday_trips=('day_of_week', lambda x: (x < 5).sum()),  # Weekdays (Mon-Fri)
    weekend_trips=('day_of_week', lambda x: (x >= 5).sum()),  # Weekends (Sat-Sun)
    electric_bike_rides=('rideable_type', lambda x: (x == 'electric_bike').sum()),  # Electric bike trips
    classic_bike_rides=('rideable_type', lambda x: (x == 'classic_bike').sum()),  # Classic bike trips
    member_rides=('member_casual', lambda x: (x == 'member').sum()),  # Member trips
    casual_rides=('member_casual', lambda x: (x == 'casual').sum()),  # Casual trips
    avg_trip_duration=('trip_duration_min', 'mean'),  # Average trip duration (minutes)
    unique_start_stations=('start_station_id', 'nunique'),  # Unique start stations
    unique_end_stations=('end_station_id', 'nunique')  # Unique end stations
).reset_index()

# Calculate proportions and averages
weekly_aggregated['electric_bike_proportion'] = weekly_aggregated['electric_bike_rides'] / weekly_aggregated['total_trips']
weekly_aggregated['classic_bike_proportion'] = weekly_aggregated['classic_bike_rides'] / weekly_aggregated['total_trips']
weekly_aggregated['member_proportion'] = weekly_aggregated['member_rides'] / weekly_aggregated['total_trips']
weekly_aggregated['casual_proportion'] = weekly_aggregated['casual_rides'] / weekly_aggregated['total_trips']
weekly_aggregated['avg_daily_trips'] = weekly_aggregated['total_trips'] / 7
weekly_aggregated['avg_weekday_trips'] = weekly_aggregated['weekday_trips'] / 5  # Average weekday trips
weekly_aggregated['avg_weekend_trips'] = weekly_aggregated['weekend_trips'] / 2  # Average weekend trips

# Add start and end dates for each week
weekly_aggregated['start_date'] = weekly_aggregated.apply(
    lambda row: pd.Timestamp(f"{row['year']}-01-01") + timedelta(days=(row['week_number'] - 1) * 7),
    axis=1
)
weekly_aggregated['end_date'] = weekly_aggregated['start_date'] + timedelta(days=6)

# Display all columns of the final aggregated DataFrame
print(weekly_aggregated[['start_date', 'end_date', 'year', 'month', 'week_number', 'borough',
                         'total_trips', 'weekday_trips', 'weekend_trips', 
                         'electric_bike_rides', 'classic_bike_rides', 'member_rides', 
                         'casual_rides', 'avg_trip_duration', 'unique_start_stations', 
                         'unique_end_stations', 'electric_bike_proportion', 
                         'classic_bike_proportion', 'member_proportion', 
                         'casual_proportion', 'avg_daily_trips', 
                         'avg_weekday_trips', 'avg_weekend_trips']])

