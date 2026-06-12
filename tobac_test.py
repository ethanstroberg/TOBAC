'''
this file is to get to know tobac and make sure everything is working fine before I start to use it
@author: ethan stroberg
@date: 6/10/26
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

########################################################################################################
# read in the data
data_directory = "/Users/ethan1/Desktop/vs_code/Rainmaker/RadarFiles/" # directory where the NEXRAD files are stored
filenames = sorted(glob.glob(os.path.join(data_directory, "KPDT20260124*_V06"))) # list of NEXRAD files that make up the dataset 

for f in filenames:
    print(os.path.basename(f)) # make sure they're sorted correctly and we have the right files

radar = pyart.io.read_nexrad_archive(filenames[0]) # assign the radar using the first file in the filenames list, they all have the same radar location
#print(radar.fields.keys())

########################################################################################################
gridded_refl = []
scans = []

iteration = 1
# convert NEXRAD input files to gridded data
for f in filenames:

    print(f"iteration: {iteration} / {len(filenames)}") # a progress bar for the user
    iteration += 1

    radar = pyart.io.read_nexrad_archive(f) # read in the radar file

    # do QC filtering here
    rhohv = radar.fields["cross_correlation_ratio"]["data"]
    refl = radar.fields["reflectivity"]["data"]
    diff_refl = radar.fields["differential_reflectivity"]["data"]
    phidp = radar.fields["differential_phase"]["data"]

    #mask, this is what we want to filter out and hide (low CC, large ZDR values, low reflectivity)
    mask = (rhohv < 0.7) | (np.abs(diff_refl) > 4) | (refl < 0)
    refl_filtered = np.ma.masked_where(mask, refl)

    # Add the filtered field to the radar object
    radar.add_field_like(
        "reflectivity",
        "reflectivity_filtered",
        refl_filtered,
        replace_existing=True,
    )

    grid = pyart.map.grid_from_radars(
        radar, 
        grid_shape=(5, 250, 250), 
        grid_limits=((0, 2500), # these are the bounds of the grid, (z, y, x)
                     (-70000, 10000), 
                     (-70000, 10000)),
        fields = ['reflectivity_filtered'])
    
    gridded_refl.append(grid.fields["reflectivity_filtered"]["data"].filled(np.nan)) # append the gridded reflectivity data to the list, filling masked values with NaN

    scantime = pd.to_datetime(radar.time["units"].split("since ")[1])
    scans.append(scantime)

########################################################################################################

all_refl = np.stack(gridded_refl, axis=0) # stack the gridded reflectivity data into a single array
print(all_refl.shape) # check the shape of the full reflectivity array
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

print(da)
print(da.shape)
print(da.time.values)
print(da.z.values)

#%%

da_track = da.isel(z=slice(0,3)).max(dim="z")

dxy =grid.x["data"][1] - grid.x["data"][0] # calculate grid spacing in meters
print(f"grid spacing = {dxy}m")

# run tobac feature detection on the data
parameters_features = {
    "position_threshold": "center",
    "min_distance": 1000.0, # meters
    "sigma_threshold": 1.5,
    "n_erosion_threshold": 1,
    "threshold": [5, 10, 15], # lower dbz thresholds for seeding signatures
    "target": "maximum",
}

features = tobac.feature_detection_multithreshold(
    da_track,
    dxy=float(dxy), # grid spacing in meters
    **parameters_features
)

print(features)

# calculate the median time step... they are all nearly the same so this is ok
# we just need to be able to find velocity so tobac can track speed
times = pd.to_datetime(da_track.time.values)
dt = float(np.median(np.diff(times) / np.timedelta64(1, "s")))

parameters_tracking = {
    "method_linking": "predict",
    "adaptive_stop": 0.2,
    "adaptive_step": 0.95,
    "extrapolate": 0,
    "order": 1,
    "subnetwork_size": 30,
    "memory": 1,
}

tracks = tobac.linking_trackpy(
    features,
    None,
    dt=dt,
    dxy=dxy,
    v_max= 30.0, # maximum velocity in m/s
    **parameters_tracking,
)

# create the figure
fig, ax = plt.subplots(figsize=(8, 6))

# track and plot seeding signatures and radar data
plt.figure(figsize=(8, 6))
pcm = plt.pcolormesh(
    da_track.x / 1000,
    da_track.y / 1000,
    da_track.isel(time=0),
    shading="auto",
    cmap="NWSRef",
    vmin=-20,
    vmax=70,
)

plt.colorbar(pcm, label = "dBZ")

ax.set_label("x (km)")
ax.set_ylabel("y (km)")
ax.plot(0, 0, marker="o", color="k", markersize=12, label="Radar")
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
    for track_id in tracks["cell"].unique():
        tr = tracks[tracks["cell"] == track_id]
        ax.plot(
            tr["x"] / 1000,
            tr["y"] / 1000,
            "-o",
            label=f"Track {track_id}",
        )
    # plot the markers for identified features (X)
    ax.scatter(
        features["x"] / 1000,
        features["y"] / 1000,
        c="black",
        marker="x",
        s=100,
        label="Seeding Signatures"
    )

    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.plot(0, 0, marker="o", color="k", markersize=12, label="Radar")
    ax.legend(loc="upper left")
    ax.set_title(pd.to_datetime(da_track.time.values[frame]).strftime("%Y-%m-%dT%H:%M:%SZ"))

# extract the date for animation title
date = pd.to_datetime(da_track.time.values[0]).strftime("%Y-%m-%d")

ani = FuncAnimation(fig, update, frames=len(da_track.time), repeat=False)
ani.save(f"/Users/ethan1/Desktop/vs_code/Rainmaker/Reflectivity_{date}.gif", writer=PillowWriter(fps=2), dpi=300)

plt.close()
