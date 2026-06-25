'''
radar reflectivity percentile testing
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
from Rainmaker.Scripts.default_radar_filter import add_default_filtered_fields

########################################################################################################
# read in the data 
# NOTE: YOU NEED THESE DOWNLOADED ON YOUR MACHINE WITH THE CORRECT FILEPATH BELOW FOR THIS TO WORK
cases = ["KPDT20260124", "KPDT20260220"]
case_str = "\n".join([f"{i+1}. {case}" for i, case in enumerate(cases)])
choice = int(input(f"choose a case to run: \n{case_str}\n"))-1

data_directory = f"/Users/ethan1/Desktop/vs_code/Rainmaker/RadarFiles/{cases[choice]}/"
filenames = sorted(glob.glob(os.path.join(data_directory, f"{cases[choice]}*_V06")))

for f in filenames:
    print(os.path.basename(f)) # make sure they're sorted correctly and we have the right files

print(f"number of files: {len(filenames)}") # check the number of files

# read first radar to gather common metadata (location, times format, etc.)
radar = pyart.io.read_nexrad_archive(filenames[0])

########################################################################################################
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
    #ref_thresholds = [3, 8, 13]
    max_velocity= 15.0
    
    # feature params
    min_distance = 5000.0
    n_min_threshold = 0
    n_erosion_threshold = 0
    sigma_threshold = 1.0

    # tracking params
    memory = 0

elif cases[choice] == "KPDT20260220":
    zmin, zmax = 1300, 1700
    ymin, ymax = -10000, 15000
    xmin, xmax = 20000, 100000
    zshape, yshape, xshape = 4, 63, 200
    zslice_min, zslice_max = 0, 4
    #ref_thresholds = [3, 8]
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

# convert NEXRAD input files to gridded data
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
########################################################################################################

all_refl = np.stack(gridded_refl, axis=0) # stack the gridded reflectivity data into a single array
print(f"shape of full reflectivity array: {all_refl.shape}") # check the shape of the full reflectivity array
# all_refl has dimensions (time, z, y, x)


#%%
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

# use these statements for debugging if you need to check the DataArray
# print(da)
# print(da.shape)
# print(da.time.values)
# print(da.z.values)


da_track = da.isel(z=slice(zslice_min, zslice_max)).max(dim="z")
ymd = pd.to_datetime(da_track.time.values[0]).strftime("%Y%m%d") # make an easy to use ymd variable for file naming and titles later

dxy =grid.x["data"][1] - grid.x["data"][0] # calculate grid spacing in meters
print(f"grid spacing = {dxy}m")

# using our filtered reflectivity data, let's calculate ref_thresholds based on percentiles to get a more generic threshold
# that way, we can apply this to any case instead of having to hardcode values each time we test a new case

# first, I want to actually know what the distribution is here, so let's calculate the statistics of the filtered reflectivity data
print(f"reflectivity stats (filtered): min={ref_filtered.min()} \n25th percentile={np.percentile(ref_filtered.compressed(), 25)}\nmedian={np.percentile(ref_filtered.compressed(), 50)} \n75th percentile={np.percentile(ref_filtered.compressed(), 75)} \nmax={ref_filtered.max()}")
# plot the histogram of this so we can visualize the distribution and mark percentiles
plt.figure(figsize=(8,6))
data = ref_filtered.compressed()
plt.hist(data, bins=50, color='blue', alpha=0.7)
plt.title(f"Histogram of Filtered Reflectivity Values {ymd}")
plt.xlabel("Reflectivity (dBZ)")
plt.ylabel("Frequency")
plt.grid()
# show specific percentiles on the histogram
pct_list = [50, 60, 70, 80, 90]
pct_vals = np.percentile(data, pct_list)
colors = ['k','r','g','m','c']
for p, v, c in zip(pct_list, pct_vals, colors):
    plt.axvline(v, color=c, linestyle='--', linewidth=1)
    plt.text(v, plt.gca().get_ylim()[1]*0.9, f"{p}th: {v:.1f}", rotation=90, color=c, va='top', ha='right')
plt.savefig(f"/Users/ethan1/Desktop/vs_code/Rainmaker/AnalysisData/{ymd}_percentile_histogram.png", dpi=300)


ref_thresholds = np.percentile(ref_filtered.compressed(), [72.5, 82.5, 92.5])

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
#features.to_csv(f"/Users/ethan1/Desktop/vs_code/Rainmaker/AnalysisData/{ymd}_featureID.csv", index=False)


# calculate the median time step... they are all nearly the same so this is ok
# we just need to be able to find velocity so tobac can track speed
times = pd.to_datetime(da_track.time.values)
dt = float(np.median(np.diff(times) / np.timedelta64(1, "s")))

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

tracks = tobac.linking_trackpy(
    features,
    None,
    dt=dt,
    dxy=dxy,
    v_max= max_velocity, # maximum velocity in m/s
    **parameters_tracking,
)

# calculate each track length and filter out tracks that are too short to get rid of noise and nonmet features
track_lengths = tracks.groupby("cell").size()

print(track_lengths)

min_frames = 6 # a feature needs to survive at least n frames to be considered
good_frames= track_lengths[track_lengths >= min_frames].index

# we also want a minimum track length to filter out stationary or short tracks that are more than the min_length
min_length = 5000 # meters
track_displacements = tracks.groupby("cell").apply( # displacement --> final - initial position
    lambda x: np.sqrt(
        (x["x"].iloc[-1] - x["x"].iloc[0])**2 + 
        (x["y"].iloc[-1] - x["y"].iloc[0])**2
    )
)

good_displacement = track_displacements[track_displacements >= min_length].index

good_cells = good_frames.intersection(good_displacement)

tracks_filtered = tracks[tracks["cell"].isin(good_cells)]

# output the filtered tracks to a csv so we can look at exact points and track them better
#tracks_filtered.to_csv(f"/Users/ethan1/Desktop/vs_code/Rainmaker/AnalysisData/{ymd}_trackID.csv", index=False)

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

ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")

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

    ax.set_title(pd.to_datetime(da_track.time.values[frame]).strftime("%Y-%m-%dT%H:%M:%SZ"))

# extract the date for animation title
date = pd.to_datetime(da_track.time.values[0]).strftime("%Y-%m-%d")

ani = FuncAnimation(fig, update, frames=len(da_track.time), repeat=False)
ani.save(f"/Users/ethan1/Desktop/vs_code/Rainmaker/Animations/Reflectivity_test_{date}.gif", writer=PillowWriter(fps=2), dpi=300)

plt.close()

# %%
