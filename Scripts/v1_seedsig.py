'''
@author: ethan stroberg
@date: 6/12/26
@version: 1.0
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
from Rainmaker.Scripts.default_radar_filter import filter_radar_volume

########################################################################################################
# read in the data ** uncomment the case you want to run **
cases = ["KPDT20260124", "KPDT20260220"]
case_str = "\n".join([f"{i+1}. {case}" for i, case in enumerate(cases)])
choice = int(input(f"choose a case to run: \n{case_str}\n"))-1

data_directory = f"/Users/ethan1/Desktop/vs_code/Rainmaker/RadarFiles/{cases[choice]}/" # directory where the NEXRAD files are stored
filenames = sorted(glob.glob(os.path.join(data_directory, f"{cases[choice]}*_V06"))) # list of NEXRAD files that make up the dataset 

for f in filenames:
    print(os.path.basename(f)) # make sure they're sorted correctly and we have the right files

# print the number of files
print(f"number of files: {len(filenames)}")

radar = pyart.io.read_nexrad_archive(filenames[0]) # assign the radar using the first file in the filenames list, they all have the same radar location
#print(radar.fields.keys())

########################################################################################################
gridded_refl = []
scans = []

# convert NEXRAD input files to gridded data
for i, f in enumerate(filenames, start=1):

    print(f"\riteration: {i} / {len(filenames)}", end="", flush=True) # a progress bar for the user

    radar = pyart.io.read_nexrad_archive(f) # read in the radar file

    # do QC filtering here
    fields = filter_radar_volume(f)

    ref_filtered = fields["reflectivity_filtered"] 
    rho_filtered = fields["cross_correlation_ratio_filtered"]
    zdr_filtered = fields["differential_reflectivity_filtered"]

    #mask, this is what we want to filter out and hide
    mask = ref_filtered < -5
    ref_filtered = np.ma.masked_where(mask, ref_filtered)

    # Add the filtered field to the radar object
    radar.add_field_like(
        "reflectivity",
        "reflectivity_filtered",
        ref_filtered,
        replace_existing=True,
    )

    # set the grid based on the case, for now at least this is gonna be somewhat hardcoded for each case
    if cases[choice] == "KPDT20260124":
        zmin, zmax = 0, 2500
        ymin, ymax = -70000, 10000
        xmin, xmax = -70000, 10000
        zshape, yshape, xshape = 5, 250, 250
    
    elif cases[choice] == "KPDT20260220":
        zmin, zmax = 500, 500
        ymin, ymax = -5000, 15000
        xmin, xmax = 20000, 80000
        zshape, yshape, xshape = 1, 80, 80

    else:
        raise ValueError("Case not recognized... input a valid case")
        
    
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

#%%

da_track = da.isel(z=slice(0,3)).max(dim="z")

dxy =grid.x["data"][1] - grid.x["data"][0] # calculate grid spacing in meters
print(f"grid spacing = {dxy}m")

# set up the parameters for feature detection
parameters_features = {
    "position_threshold": "weighted_diff", # four options here, center, extreme, weighted_diff, and abs_diff. tobac recommends weighted or abs
    "min_distance": 5000.0, # meters, min required difference between features. if two features are closer than this, the one with the more extreme value is kept
    "sigma_threshold": 1.0, # gaussian smoothing parameter (scipy.ndimage.gaussian_filter)
    "n_erosion_threshold": 0, # reduces the size of a feature in all direcitons (skimage.morphology.binary_erosion)
    "n_min_threshold": 0, # min number of pixels required for a feature to be detected
    "threshold": [3, 8, 13], # lower dbz thresholds for seeding signatures
    "target": "maximum", # looking for local maxima to be marked as features
}

# feature_detection_multithreshold outputs a pandas dataframe
features = tobac.feature_detection_multithreshold(
    da_track,
    dxy=float(dxy), # grid spacing in meters
    **parameters_features
)
# output the pandas dataframe to a file
ymd = pd.to_datetime(da_track.time.values[0]).strftime("%Y%m%d")
features.to_csv(f"/Users/ethan1/Desktop/vs_code/Rainmaker/{ymd}_featureID.csv", index=False)


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
    "memory": 2, # how long a feature can disappear for before we stop trying to link it to a track
}

tracks = tobac.linking_trackpy(
    features,
    None,
    dt=dt,
    dxy=dxy,
    v_max= 15.0, # maximum velocity in m/s
    **parameters_tracking,
)

# calculate each track length and filter out tracks that are too short to get rid of noise and nonmet features
track_lengths = tracks.groupby("cell").size()

print(track_lengths)

min_frames = 6 # a feature needs to survive at least n frames to be considered
good_cells = track_lengths[track_lengths >= min_frames].index

# we also want a minimum track length to filter out stationary or short tracks that are more than the min_length
# min_length = 5000 # meters
# track_displacements = tracks.groupby("cell").apply( # displacement --> final - initial position
#     lambda x: np.sqrt(
#         (x["x"].iloc[-1] - x["x"].iloc[0])**2 + 
#         (x["y"].iloc[-1] - x["y"].iloc[0])**2
#     )
# )

# good_displacement = track_displacements[track_displacements >= min_length].index

# good_cells = good_frames.intersection(good_displacement)

tracks_filtered = tracks[tracks["cell"].isin(good_cells)]

# create the figure
fig, ax = plt.subplots(figsize=(8, 6))

# track and plot seeding signatures and radar data
pcm = ax.pcolormesh(
    da_track.x / 1000,
    da_track.y / 1000,
    da_track.isel(time=0),
    shading="auto",
    cmap="NWSRef",
    vmin=-20,
    vmax=70,
)

cbar = fig.colorbar(
    plt.cm.ScalarMappable(
        cmap="NWSRef",
        norm=plt.Normalize(-20,70)
    ),
    ax=ax
)
cbar.set_label("dBZ")

ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
ax.legend(loc="upper right")

# define a function that will update the plot for each frame of the animation
def update(frame):
    ax.clear() # clear any old display
    # color information
    pcm = ax.pcolormesh(
        da_track.x / 1000,
        da_track.y / 1000,
        da_track.isel(time=frame),
        shading="auto",
        cmap="NWSRef",
        vmin=-20,
        vmax=70,
    )

    # plot the tracks identfied by tobac (lines and dots)
    for track_id in tracks_filtered["cell"].unique():
        tr = tracks_filtered[tracks_filtered["cell"] == track_id]
        ax.plot(
            tr["x"] / 1000,
            tr["y"] / 1000,
            "-o",
            label=f"Track {track_id}",
        )

    # plot the markers for identified features (X)
    tracked_features = tracks_filtered[
    tracks_filtered["frame"] == frame
]

    ax.scatter(
        tracked_features["x"] / 1000,
        tracked_features["y"] / 1000,
        c="black",
        marker="x",
        s=100,
        label="Tracked Features"
    )

    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    # only plot the radar location if the radar is within the grid bounds, otherwise the plot gets all stretchy
    if (0 >= xmin) and (0 <= xmax) and (0 >= ymin) and (0 <= ymax):
        ax.plot(0, 0, marker="o", color="k", markersize=12, label="Radar")
    ax.legend(loc="upper left")
    ax.set_title(pd.to_datetime(da_track.time.values[frame]).strftime("%Y-%m-%dT%H:%M:%SZ"))

# extract the date for animation title
date = pd.to_datetime(da_track.time.values[0]).strftime("%Y-%m-%d")

ani = FuncAnimation(fig, update, frames=len(da_track.time), repeat=False)
ani.save(f"/Users/ethan1/Desktop/vs_code/Rainmaker/Reflectivity_{date}.gif", writer=PillowWriter(fps=2), dpi=300)

plt.close()
