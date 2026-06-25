import pandas as pd
import glob
import os
import numpy as np
import pyart

# path = "/Users/ethan1/Desktop/vs_code/Rainmaker/AnalysisData/"

# filenames = sorted(glob.glob(os.path.join(path, f"*trackID*")))
# for f in filenames:
#     df = pd.read_csv(f)

#     print(f"file: {f}")
#     for n in df['cell'].unique():
#         max_ref = df[df['cell'] == n]['reflectivity'].max()
#         max_ref_loc = df[(df['cell'] == n) & (df['reflectivity'] == max_ref)].iloc[0]
#         min_ref = df[df['cell'] == n]['reflectivity'].min()
#         min_ref_loc = df[(df['cell'] == n) & (df['reflectivity'] == min_ref)].iloc[0]
#         print(f"cell: {n}, max ref: {max_ref}, pixels: {max_ref_loc['num']}, min ref: {min_ref}, pixels: {min_ref_loc['num']}")
    
#     print()

# file = "/Users/ethan1/Desktop/vs_code/Rainmaker/Seeding_Flight_Conditions_MASTER - PDT_Seeding_Flights.csv"
# df = pd.read_csv(file)

# # for PDT025 = 13, 14, PDT032 SM1-3 = 21, 22, 23
# # cols 17, 18 are closest sounding wspd and wdir, use these if they exist, otherwise use drone data 24, 25
# df = df.iloc[[13, 14, 21, 22, 23], [0, 1, 5, 17, 18, 24, 25]]
# #print(df)

# for index, row in df.iterrows():

#     if pd.isna(row.iloc[3]) or pd.isna(row.iloc[4]):
#         wspd = row.iloc[5]  # Original 24
#         wdir = row.iloc[6]  # Original 25
#     else:
#         wspd = row.iloc[3]  # Original 17
#         wdir = row.iloc[4]  # Original 18

#     print(f"wspd: {wspd}, wdir: {wdir}")
# Color codes
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

# Print examples
print(f"{RED}This text is red!{RESET}")
print(f"{GREEN}This text is green!{RESET}")
print(f"Normal text, {YELLOW}yellow text{RESET}, back to normal.")


# we will lock on to the site butter or cabin, easier when there's multiple drones right next to each other all launched from the same site
# butter_latlon = (45.50800, -119.01300)
# cabin_latlon = (45.76400, -118.28100)

# file = "/Users/ethan1/Desktop/vs_code/Rainmaker/RadarFiles/KPDT20260124/KPDT20260124_014146_V06"

# radar = pyart.io.read_nexrad_archive(file)

# def get_site_location(site):
#     # convert from lat lon to (x, y) in meters relative to the KPDT radar location

#     a = 6378137.0 # WGS-84 equatorial radius in meters
#     e2 = 0.00669437999014 # WGS-84 eccentricity

#     lat = np.asarray(site[0])
#     lon = np.asarray(site[1])

#     radar_lat = np.radians(radar.latitude['data'][0])
#     radar_lon = np.radians(radar.longitude['data'][0])

#     lat_rad = np.radians(lat)
#     lon_rad = np.radians(lon)

#     m_per_rad_lat = a * (1 - e2) / (1 - e2 * np.sin(radar_lat)**2)**(3/2)
#     m_per_rad_lon = (a * np.cos(radar_lat)) / np.sqrt(1 - e2 * np.sin(radar_lat)**2)

#     dlat = lat_rad - radar_lat
#     dlon = lon_rad - radar_lon

#     x = dlon * m_per_rad_lon
#     y = dlat * m_per_rad_lat

#     return x, y

# x, y = get_site_location(butter_latlon)

# print(f"Butter site location relative to radar: x = {x:.2f} m, y = {y:.2f} m")
# print(f"Straight-line distance: {np.sqrt(x**2 + y**2):.2f} m")