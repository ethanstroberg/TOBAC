'''
SeedSig - Seeding Signature Automatic Identification and Tracking
@author: ethan stroberg
@date: 6/22/26

@version: 1.2
- thresholds now based on positive dBZ anomalies
- several new requirements added for feature tracking
- drone seeding location and wind vector information now added
'''
#%%
#imports, make sure everything we need is here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import pyart
import tobac 
import glob
import os
from matplotlib.animation import FuncAnimation, PillowWriter
from default_radar_filter import add_default_filtered_fields
from scipy.ndimage import gaussian_filter
import trackpy as tp
tp.quiet() # turn off trackpy warnings/messages

#############################################################################################################################################################
# read in the data 
# NOTE: YOU NEED THESE DOWNLOADED ON YOUR MACHINE WITH THE CORRECT FILEPATH BELOW FOR THIS TO WORK
#############################################################################################################################################################
cases = ["KPDT20260124", "KPDT20260220_SM1_3", "KPDT20260220_SM4_13", "KPDT20260220_SM16_18"]
case_str = "\n".join([f"{i+1}. {case}" for i, case in enumerate(cases)])
choice = int(input(f"choose a case to run: \n{case_str}\n"))-1

data_directory = f"/Users/ethan1/Desktop/vs_code/Rainmaker/RadarFiles/{cases[choice]}/"
filenames = sorted(glob.glob(os.path.join(data_directory, f"*_V06")))

for f in filenames:
    print(os.path.basename(f)) # make sure they're sorted correctly and we have the right files

print(f"number of files: {len(filenames)}") # check the number of files

# read first radar to gather common metadata (location, times format, etc.)
radar = pyart.io.read_nexrad_archive(filenames[0])

#############################################################################################################################################################
# set custom settings per case for now as we're testing params --> eventually this should be automated based on the data itself
#############################################################################################################################################################
#%%
gridded_refl = []
scans = []

# set the grid based on the case once (same for all files in a case, let's not loop this 20 times)
# we want to try to avoid hardcoding more things and if else statements later, add any other variables that vary by case
if cases[choice] == "KPDT20260124":
    zmin, zmax = 0, 2500
    ymin, ymax = -70000, 10000
    xmin, xmax = -70000, 10000
    zshape, yshape, xshape = 5, 250, 250
    zslice_min, zslice_max = 0, 3
    max_velocity= 15.0
    
    # feature params
    min_distance = 5000.0
    n_min_threshold = 0
    n_erosion_threshold = 0
    sigma_threshold = 1.0

    # tracking params
    memory = 0

elif cases[choice] == "KPDT20260220_SM1_3":
    zmin, zmax = 1300, 1700
    ymin, ymax = -10000, 15000
    xmin, xmax = 20000, 100000
    zshape, yshape, xshape = 4, 63, 200
    zslice_min, zslice_max = 0, 4
    max_velocity= 15.0

    # feature params
    min_distance = 5000.0
    n_min_threshold = 0
    n_erosion_threshold = 0
    sigma_threshold = 1.0

    # tracking params
    memory = 0

elif cases[choice] == "KPDT20260220_SM4_13":
    zmin, zmax = 1300, 1700
    ymin, ymax = 2500, 27500
    xmin, xmax = 20000, 100000
    zshape, yshape, xshape = 4, 63, 200
    zslice_min, zslice_max = 0, 4
    max_velocity= 15.0

    # feature params
    min_distance = 5000.0
    n_min_threshold = 0
    n_erosion_threshold = 0
    sigma_threshold = 1.0

    # tracking params
    memory = 0

elif cases[choice] == "KPDT20260220_SM16_18":
    zmin, zmax = 1300, 1700
    ymin, ymax = 2500, 27500
    xmin, xmax = 20000, 100000
    zshape, yshape, xshape = 4, 63, 200
    zslice_min, zslice_max = 0, 4
    max_velocity= 15.0

    # feature params
    min_distance = 5000.0
    n_min_threshold = 0
    n_erosion_threshold = 0
    sigma_threshold = 1.0

    # tracking params
    memory = 0

else:
    raise ValueError("Case not recognized... input a valid case")

#############################################################################################################################################################
# convert NEXRAD input files to gridded data
#############################################################################################################################################################
#%%
for i, f in enumerate(filenames, start=1):

    print(f"\riteration: {i} / {len(filenames)}", end="", flush=True) # a progress bar for the user

    radar = pyart.io.read_nexrad_archive(f) # read in the radar file once

    # apply the default filter in-place to avoid re-reading the file
    radar = add_default_filtered_fields(radar)

    ref_filtered = radar.fields["reflectivity_filtered"]["data"]
    rho_filtered = radar.fields["cross_correlation_ratio_filtered"]["data"]
    zdr_filtered = radar.fields["differential_reflectivity_filtered"]["data"]

    # additional mask: remove very low reflectivity values
    mask = ref_filtered < -5
    ref_filtered = np.ma.masked_where(mask, ref_filtered)

    # ensure radar field reflects the additional mask
    radar.add_field_like(
        "reflectivity",
        "reflectivity_filtered",
        ref_filtered,
        replace_existing=True,
    )

    grid = pyart.map.grid_from_radars(
            radar, 
            grid_shape=(zshape, yshape, xshape), 
            grid_limits=((zmin, zmax), # these are the bounds of the grid, (z, y, x)
                        (ymin, ymax), 
                        (xmin, xmax)),
            fields = ['reflectivity_filtered'])

    gridded_refl.append(grid.fields["reflectivity_filtered"]["data"].filled(np.nan)) # append the gridded reflectivity data to the list, filling masked values with NaN

    scantime = pd.to_datetime(radar.time["units"].split("since ")[1])
    scans.append(scantime)

print() # newline after progress tracker
print() # newline

all_refl = np.stack(gridded_refl, axis=0) # stack the gridded reflectivity data into a single array
print(f"shape of full reflectivity array: {all_refl.shape}") # check the shape of the full reflectivity array
# all_refl has dimensions (time, z, y, x)

dxy =grid.x["data"][1] - grid.x["data"][0] # calculate grid spacing in meters
print(f"grid spacing = {dxy}m")
print()

# build the xarray DataArray
da = xr.DataArray(
    all_refl,
    dims=("time", "z", "y", "x"),
    coords={
        "time": scans,
        "z": grid.z["data"],
        "y": grid.y["data"],
        "x": grid.x["data"],
    },
    name="reflectivity"
)                                            

da_track = da.isel(z=slice(zslice_min, zslice_max)).max(dim="z")

#############################################################################################################################################################
# compute relevant statistics for thresholding and testing
#############################################################################################################################################################
#%%
# calculate the background noise level --> do this using the median reflectivity
median_background_noise = float(da_track.median(dim=("time", "y", "x")).values)
mean_background_noise = float(da_track.mean(dim=("time", "y", "x")).values)
print(f"median background noise level = {median_background_noise:.2f} dBZ")
print(f"mean background noise level = {mean_background_noise:.2f} dBZ")

# calculate a bunch of statistics to try to find something we can use to set a branch for thresholds that handles high noise cases better
# for each threshold, add a certain std dev above the background noise level to add layers
std_dev = float(da_track.std(dim=("time", "y", "x")).values)
print(f"standard deviation = {std_dev:.2f} dBZ")

# see if IQR is meaningully different between the two
iqr = float(da_track.quantile(0.75, dim=("time", "y", "x")).values - da_track.quantile(0.25, dim=("time", "y", "x")).values)
print(f"interquartile range = {iqr:.2f} dBZ")

# calculate the percent of the grid that has any ref signal at all
percent_covered = float((da_track > 0).sum(dim=("time", "y", "x")).values) / float(da_track.size) * 100.0
print(f"percent nonzero = {percent_covered:.2f}%")

# set thresholds with some low enough to maintain detection of a decaying plume, set a minimum peak later in the code
ref_thresholds = [median_background_noise + 3,
                  median_background_noise + 8,
                  median_background_noise + 13]

print(f"reflectivity thresholds set to: {', '.join(f'{x:.2f}' for x in ref_thresholds)} dBZ")
print()
#############################################################################################################################################################
# set up params for feature detection and tracking
#############################################################################################################################################################
#%%
# set up the parameters for feature detection
parameters_features = {
    "position_threshold": "weighted_diff", # four options here, center, extreme, weighted_diff, and abs_diff. tobac recommends weighted or abs
    "min_distance": min_distance, # meters, min required difference between features. if two features are closer than this, the one with the more extreme value is kept
    "sigma_threshold": sigma_threshold, # gaussian smoothing parameter (scipy.ndimage.gaussian_filter)
    "n_erosion_threshold": n_erosion_threshold, # reduces the size of a feature in all direcitons (skimage.morphology.binary_erosion)
    "n_min_threshold": n_min_threshold, # min number of pixels required for a feature to be detected
    "threshold": ref_thresholds, # lower dbz thresholds for seeding signatures
    "target": "maximum", # looking for local maxima to be marked as features
}

# feature_detection_multithreshold outputs a pandas dataframe
features = tobac.feature_detection_multithreshold(
    da_track,
    dxy=float(dxy), # grid spacing in meters
    **parameters_features
)
# output the pandas dataframe to a file
ymdt = pd.to_datetime(da_track.time.values[0]).strftime("%Y%m%d_%H%M%SZ")

# calculate the median time step... they are all nearly the same so this is ok
# we just need to be able to find velocity so tobac can track speed
times = pd.to_datetime(da_track.time.values)

# robustly compute median time step in seconds; handle cases where times may be object-dtype
try:
    # preferred fast path using numpy datetime64 array
    dt = float(np.median(np.diff(times.values) / np.timedelta64(1, "s")))
except Exception:
    # fallback to pandas Timedelta median
    dt = float(pd.Series(times).diff().median() / np.timedelta64(1, "s"))

# set up the parameters for tracking in time and space
parameters_tracking = {
    "method_linking": "predict",
    "adaptive_stop": 0.2,
    "adaptive_step": 0.95,
    "extrapolate": 0,
    "order": 1,
    "subnetwork_size": 30,
    "memory": memory, # how long a feature can disappear for before we stop trying to link it to a track
}

# tracks is also a pandas dataframe
tracks = tobac.linking_trackpy(
    features,
    None,
    dt=dt,
    dxy=dxy,
    v_max= max_velocity, # maximum velocity in m/s
    **parameters_tracking,
)

#############################################################################################################################################################
# POST PROCESSING AFTER CREATING TRACKS AND FEATURES DATAFRAMES
#############################################################################################################################################################
#%%
# create a filter for a required minimum number of frames, minimum displacement, minimum peak reflectivity, and minimum number of pixels during peak reflectivity

# tracks does not natively include the reflectivity value of the feature, so we need to grab that and add it into the tracks dataframe
features["reflectivity"] = [
    da_track.isel(
        time = int(row["frame"]),
        y = int(row["hdim_1"]),
        x = int(row["hdim_2"])
    ).item()
    for _, row in features.iterrows()
]
# save features to a csv
features.to_csv(f"/Users/ethan1/Desktop/vs_code/Rainmaker/AnalysisData/{ymdt}_featureID.csv", index=False)
# merge these values into the tracks df
tracks = tracks.merge(
    features[["feature", "reflectivity"]],
    on = "feature",
    how = "left"
)

# calculate the number of frames each track lasts and filter out tracks that are too short to get rid of noise and nonmet features
track_lengths = tracks.groupby("cell").size()
#print(track_lengths)
min_frames = 3 # a feature needs to survive at least n frames to be considered
good_frames= track_lengths[track_lengths >= min_frames].index

# we also want a minimum track length to filter out stationary or short tracks that are less than the min_length
min_length = 5000 # meters
track_displacements = tracks.groupby("cell").apply(lambda x: np.sqrt((x["x"].iloc[-1] - x["x"].iloc[0])**2 + (x["y"].iloc[-1] - x["y"].iloc[0])**2))
good_displacement = track_displacements[track_displacements >= min_length].index

# third, we want to make it so that a feature must pass a certain threshold to be detected, but can decay and still be tracked as it decays
# set lower threshold levels above, but set a minimum peak ref value that must be passed to count the track as a good cell
required_ref = median_background_noise + 15 # plume must at some point exceed this threshold
peak_ref = tracks.groupby("cell")["reflectivity"].max()
good_ref = peak_ref[peak_ref >= required_ref].index

# at its peak reflectivity, the plume must have at least 10 pixels above required_ref
required_min_peak_pixels = 0
idx_peak = tracks.groupby("cell")["reflectivity"].idxmax()
peaks = tracks.loc[idx_peak]
good_peak_pixels = peaks.loc[peaks["num"] >= required_min_peak_pixels, "cell"]

# combine the above requirements to get a filtered tracks dataframe
# NOTE: moved down to after the cone creation, could be moved later
# good_cells = good_frames.intersection(good_displacement).intersection(good_ref).intersection(good_peak_pixels)
# tracks_filtered = tracks[tracks["cell"].isin(good_cells)]

# # output the filtered tracks to a csv so we can look at exact points and track them better
# tracks_filtered = tracks_filtered[['frame', 'idx', 'reflectivity', 'num', 'cell', 'y', 'x', 'timestr', 'hdim_1', 'hdim_2', 'threshold_value', 'feature', 'time', 'time_cell']]
# tracks_filtered.to_csv(f"/Users/ethan1/Desktop/vs_code/Rainmaker/AnalysisData/{ymdt}_trackID.csv", index=False)

# print(f"features detected: {len(features)}")
# print(f"tracks drawn: {len(tracks_filtered)}")
# print()

#############################################################################################################################################################
# use the drone seeding location to obtain the mean wind vector and radar relative location of the site
#############################################################################################################################################################
#%%
# we will need the drone location, mean wind vector, and the time of seeding
flight_data = "/Users/ethan1/Desktop/vs_code/Rainmaker/Seeding_Flight_Conditions_MASTER - PDT_Seeding_Flights.csv"

df = pd.read_csv(flight_data)

# cols we're interested in: 0 (date), 1(seeding start time), 5 (drone #), 24 (median drone wspd), 25 (median drone wdir), 26/27 (stdev wspd / wdir)
cols = [0, 1, 5, 17, 18, 24, 25] # 17, 18 are closest sounding wspd and wdir, use these if they exist, otherwise use drone data 24, 25

drone_df = df.iloc[:, cols] # keep all rows for now, keep only relevant cols

# save the cleaned up drone data to a csv to make our life easier if we want to look at it
drone_df.to_csv("/Users/ethan1/Desktop/vs_code/Rainmaker/AnalysisData/drone_data_master.csv", index=False)

# until automation is added, branching statements to choose the right rows
# we will lock on to the site butter or cabin, easier when there's multiple drones right next to each other all launched from the same site
butter_latlon = (45.50800, -119.01300)
cabin_latlon = (45.76400, -118.28100)

# func to convert from lat lon to (x, y) in meters relative to the radar location
def get_site_location(site): 

    a = 6378137.0 # WGS-84 equatorial radius in meters
    e2 = 0.00669437999014 # WGS-84 eccentricity

    lat = np.asarray(site[0])
    lon = np.asarray(site[1])

    radar_lat = np.radians(radar.latitude['data'][0])
    radar_lon = np.radians(radar.longitude['data'][0])

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    m_per_rad_lat = a * (1 - e2) / (1 - e2 * np.sin(radar_lat)**2)**(3/2)
    m_per_rad_lon = (a * np.cos(radar_lat)) / np.sqrt(1 - e2 * np.sin(radar_lat)**2)

    dlat = lat_rad - radar_lat
    dlon = lon_rad - radar_lon

    x = dlon * m_per_rad_lon
    y = dlat * m_per_rad_lat

    return x, y

def get_wind(df):
    wspd_list = []
    wdir_list = []

    for index, row in df.iterrows():
        
        if pd.isna(row.iloc[3]) or pd.isna(row.iloc[4]):
            wspd = row.iloc[5]  # Original 24
            wdir = row.iloc[6]  # Original 25
        else:
            wspd = row.iloc[3]  # Original 17
            wdir = row.iloc[4]  # Original 18

        wspd_list.append(wspd)
        wdir_list.append(wdir)

    return wspd_list, wdir_list

wspd_list = []
wdir_list = []

if cases[choice] == "KPDT20260124":
    x_site, y_site = get_site_location(butter_latlon)
    drone_df = drone_df.iloc[[13, 14], :]
    wspd_list, wdir_list = get_wind(drone_df)
    
elif cases[choice] == "KPDT20260220_SM1_3":
    x_site, y_site = get_site_location(cabin_latlon)
    drone_df = drone_df.iloc[[21, 22, 23], :]
    wspd_list, wdir_list = get_wind(drone_df)

elif cases[choice] == "KPDT20260220_SM4_13":
    x_site, y_site = get_site_location(cabin_latlon)
    drone_df = drone_df.iloc[[26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38], :]
    wspd_list, wdir_list = get_wind(drone_df)

elif cases[choice] == "KPDT20260220_SM16_18":
    x_site, y_site = get_site_location(cabin_latlon)
    drone_df = drone_df.iloc[[38, 39, 40, 41], :]
    wspd_list, wdir_list = get_wind(drone_df)

else:
    raise ValueError("Case not recognized... input a valid case")

print(f"radar relative site location: x = {x_site:.2f} m, y = {y_site:.2f} m")

# for many of these cases, there are multiple drones seeding at various heights all right next to each other
# they recorded statistically indifferent wind speeds and directions most of the time, so we will take the average of the list as our wspd and wdir
# in case there is an outlier, we will still print the list of wspds and wdirs used in the average so the user can tell if there is a bad value that needs to be removed
wspd_avg = np.mean(wspd_list)

# wdir is in degrees, so we need to make sure the 0/360 wrap around is handled correctly 
def mean_wind_direction(wdir_list):
    wdir_rad = np.radians(wdir_list)

    mean_sin = np.mean(np.sin(wdir_rad))
    mean_cos = np.mean(np.cos(wdir_rad))

    mean_dir = np.degrees(np.arctan2(mean_sin, mean_cos))

    return mean_dir % 360

wdir_avg = mean_wind_direction(wdir_list)

for i in range(len(wspd_list)):
    print(f"wspd: {wspd_list[i]} m/s, wdir: {wdir_list[i]}º")
print(f"avg wspd: {wspd_avg:.2f} m/s, avg wdir: {wdir_avg:.2f}º")

#############################################################################################################################################################
# create the cone of allowance
#############################################################################################################################################################
#%%
# a plume will show up within 45 mins usually, set a max distance
max_dist = wspd_avg * 2400 # max allowed dist of cone in meters --> speed (m/s) * time (s) = distance (m)
half_angle = 30 # deg to each side of the mean wdir
downwind_angle = (wdir_avg + 180) % 360 # we need downwind so we can use our side angle
theta = np.radians(90 - downwind_angle) # this is the mathematical angle, not meteorological

print(f"max distance of cone = {max_dist:.2f} m")
print(f"half angle set to {half_angle}º, downwind angle = {downwind_angle:.2f}º")
print(f"mathematical theta = {np.degrees(theta):.2f}º")
print()

left_theta = np.radians(90 - (downwind_angle - half_angle))
right_theta = np.radians(90 - (downwind_angle + half_angle))

x_left = x_site + max_dist * np.cos(left_theta)
y_left = y_site + max_dist * np.sin(left_theta)
x_right = x_site + max_dist * np.cos(right_theta)
y_right = y_site + max_dist * np.sin(right_theta)

# unit vector pointing downwind
u = np.cos(theta)
v = np.sin(theta)
wind_vec = np.array([u, v])

good_cells_cone = []

for cell, tr in tracks.groupby("cell"):
    # get the first point of the track
    first_point = tr.iloc[0]

    dx = first_point["x"] - x_site
    dy = first_point["y"] - y_site

    point_vec = np.array([dx, dy])
    dist = np.linalg.norm(point_vec)

    # reject a track that starts beyond the max distance
    if dist > max_dist: 
        continue

    # angle between the starting point and downwind axis
    cosang = np.dot(point_vec, wind_vec) / dist

    angle = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

    if angle <= half_angle:
        good_cells_cone.append(cell)

good_cells_cone = pd.Index(good_cells_cone)

print(f"tracks within cone: {len(good_cells_cone)}")

good_cells = good_frames.intersection(good_displacement).intersection(good_ref).intersection(good_peak_pixels).intersection(good_cells_cone)
tracks_filtered = tracks[tracks["cell"].isin(good_cells)]

# output the filtered tracks to a csv so we can look at exact points and track them better
tracks_filtered = tracks_filtered[['frame', 'idx', 'reflectivity', 'num', 'cell', 'y', 'x', 'timestr', 'hdim_1', 'hdim_2', 'threshold_value', 'feature', 'time', 'time_cell']]
tracks_filtered.to_csv(f"/Users/ethan1/Desktop/vs_code/Rainmaker/AnalysisData/{ymdt}_trackID.csv", index=False)

print(f"features detected: {len(features)}")
print(f"tracks drawn: {len(tracks_filtered)}")
print()

#############################################################################################################################################################
# plot the animations
#############################################################################################################################################################
#%%
# create the figure
fig, ax = plt.subplots(figsize=(8, 6))

# use imshow for faster frame updates; set extent from grid coords (convert to km)
xmin_km = da_track.x.values.min() / 1000.0
xmax_km = da_track.x.values.max() / 1000.0
ymin_km = da_track.y.values.min() / 1000.0
ymax_km = da_track.y.values.max() / 1000.0

im = ax.imshow(
    da_track.isel(time=0),
    origin="lower",
    extent=(xmin_km, xmax_km, ymin_km, ymax_km),
    cmap="NWSRef",
    vmin=-20,
    vmax=70,
    aspect="auto",
)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("dBZ")

ax.set_xlabel("Distance from Radar (km)")
ax.set_ylabel("Distance from Radar (km)")
ax.set_xlim(xmin_km, xmax_km)
ax.set_ylim(ymin_km, ymax_km)
ax.autoscale(False) # so the cone can't mess with it

# pre-create line artists for each track to avoid re-plotting every frame
track_ids = sorted(tracks_filtered["cell"].unique())
line_artists = {}
for track_id in track_ids:
    ln, = ax.plot([], [], "-o", label=f"Track {track_id}")
    line_artists[track_id] = ln

# scatter artist for tracked features at the current frame
scatter_artist = ax.scatter([], [], c="black", marker="x", s=100, label="Tracked Features")

# plot radar location if inside bounds
radar_marker = None
if (0 >= xmin) and (0 <= xmax) and (0 >= ymin) and (0 <= ymax):
    radar_marker, = ax.plot(0, 0, marker="o", color="k", markersize=12, label="Radar")

# plot drone seeding location
drone_marker, = ax.plot(x_site / 1000.0, y_site / 1000.0, marker="s", color="k", markersize=8, label="Drone Seeding Location")

# plot the cone of allowance
ax.plot(
    [x_site / 1000.0, x_left / 1000.0],
    [y_site / 1000.0, y_left / 1000.0],
    "k--",
    lw=2
)
ax.plot(
    [x_site / 1000.0, x_right / 1000.0],
    [y_site / 1000.0, y_right / 1000.0],
    "k--",
    lw=2
)

ax.fill(
    [x_site / 1000.0, x_left / 1000.0, x_right / 1000.0],
    [y_site / 1000.0, y_left / 1000.0, y_right / 1000.0],
    alpha=0.15,
    clip_on=True
)


ax.legend(loc="upper left")

def update(frame):
    # update the image
    im.set_data(da_track.isel(time=frame))

    # update each track line to only include points up to the current frame
    for track_id, ln in line_artists.items():
        tr = tracks_filtered[(tracks_filtered["cell"] == track_id) & (tracks_filtered["frame"] <= frame)]
        if not tr.empty:
            ln.set_data(tr["x"].values / 1000.0, tr["y"].values / 1000.0)
        else:
            ln.set_data([], [])

    # update scatter for tracked features at this frame
    tracked_features = tracks_filtered[tracks_filtered["frame"] == frame]
    if not tracked_features.empty:
        offsets = np.c_[tracked_features["x"].values / 1000.0, tracked_features["y"].values / 1000.0]
        scatter_artist.set_offsets(offsets)
    else:
        scatter_artist.set_offsets(np.empty((0, 2)))

    ax.set_title(f"Reflectivity Seeding Signature Tracks\n{pd.to_datetime(da_track.time.values[frame]).strftime('%Y-%m-%dT%H:%M:%SZ')}")

ani = FuncAnimation(fig, update, frames=len(da_track.time), repeat=False)
ani.save(f"/Users/ethan1/Desktop/vs_code/Rainmaker/Animations/{ymdt}_AnomRef.gif", writer=PillowWriter(fps=2), dpi=300)

plt.close()
