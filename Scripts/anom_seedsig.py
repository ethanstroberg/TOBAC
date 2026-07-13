'''
SeedSig - Seeding Signature Automatic Identification and Tracking
@author: ethan stroberg
@date: 7/7/26

@version: 1.3.1
1. code is now compartmentalized into functions for easier testing and debugging going forward
2. case listing is now adaptive to what is in the RadarFiles folder
    a. subsequent branching is still hardcoded for now
3. added parallel processing - gridding is now FAR faster
4. horizontal gridding and limits is now automated
5. Segmentation added, needs work but it IS encompassing seeding signatures
6. partially automated parameter selection - horizontal is adaptive, vertical is still manual
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import tobac 
import glob
import os
import pyart
from matplotlib.animation import FuncAnimation, PillowWriter
from default_radar_filter import add_default_filtered_fields
from scipy.ndimage import gaussian_filter
import trackpy as tp
tp.quiet() # turn off trackpy warnings/messages
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from skimage.measure import find_contours
import warnings
warnings.filterwarnings("ignore")


def selectCase():
    # read in case options directly from RadarFiles
    cases = sorted([d for d in os.listdir("/Users/ethan1/Desktop/vs_code/Rainmaker/RadarFiles/") if os.path.isdir(os.path.join("/Users/ethan1/Desktop/vs_code/Rainmaker/RadarFiles/", d))])
    case_str = "\n".join([f"{i+1}. {case}" for i, case in enumerate(cases)])
    choice = int(input(f"choose a case to run: \n{case_str}\n"))-1

    data_directory = f"/Users/ethan1/Desktop/vs_code/Rainmaker/RadarFiles/{cases[choice]}/"
    filenames = sorted(glob.glob(os.path.join(data_directory, f"*_V06")))

    for f in filenames:
        print(os.path.basename(f)) # make sure they're sorted correctly and we have the right files

    print(f"number of files: {len(filenames)}") # check the number of files
    print()

    # read first radar to gather common metadata (location, times format, etc.)
    radar = pyart.io.read_nexrad_archive(filenames[0])


    return filenames, cases[choice], radar



def loadCaseConfig(caseName): 
    # this will need to change eventually when we want to automate things, but for now we will put a dictionary of our cases in here
    case_configs = {
        "KPDT20260124": {
            "zgrid_limits": (0, 2500),
            "zgrid_shape": 6,
            "zslice": (0, 6),
            "min_distance": 5000.0,
            "sigma_threshold": 0.5,
            "grid_weighting_function": "Barnes2",
            "grid_roi_func": "constant",
            "grid_constant_roi": 400,
            "segmentation_threshold": -5,
            "memory": 0,
            "site": "butter",
            "rows": [13, 14]
        },

        "KPDT20260220_SM1_3": {
            "zgrid_limits": (1300, 1700),
            "zgrid_shape": 4,
            "zslice": (0, 4),
            "min_distance": 5000.0,
            "sigma_threshold": 0.25,
            "grid_weighting_function": "Barnes2",
            "grid_roi_func": "constant",
            "grid_constant_roi": 400,
            "segmentation_threshold": 2,
            "memory": 0,
            "site": "cabin",
            "rows": [21, 22, 23]
        },

        "KPDT20260220_SM4_13": {
            "zgrid_limits": (1300, 1700),
            "zgrid_shape": 4,
            "zslice": (0, 4),
            "min_distance": 5000.0,
            "sigma_threshold": 0.25,
            "grid_weighting_function": "Barnes2",
            "grid_roi_func": "constant",
            "grid_constant_roi": 600,
            "segmentation_threshold": 8,
            "memory": 1,
            "site": "cabin",
            "rows": [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
        },

        "KPDT20260220_SM16_18": {
            "zgrid_limits": (1300, 1700),
            "zgrid_shape": 5,
            "zslice": (0, 5),
            "min_distance": 5000.0,
            "sigma_threshold": 0.25,
            "grid_weighting_function": "Barnes2",
            "grid_roi_func": "constant",
            "grid_constant_roi": 400,
            "segmentation_threshold": 10,
            "memory": 0,
            "site": "cabin",
            "rows": [38, 39, 40, 41]
        }
    }

    return case_configs.get(caseName, None)



def autoGridAxes(filenames, wind_location): 
    
    wspd_avg = wind_location["wspd_avg"]
    wdir_avg = wind_location["wdir_avg"]
    x0 = wind_location["x_site"]
    y0 = wind_location["y_site"]

    # calculate a mathematical maxumum extent of the plume xf, yf
    # extract the length of time from the radar files, given they have the same format and are sorted
    first = os.path.basename(filenames[0]).split("_")
    last = os.path.basename(filenames[-1]).split("_")

    start_time = pd.to_datetime(first[0][4:] + first[1], format="%Y%m%d%H%M%S")
    end_time = pd.to_datetime(last[0][4:] + last[1], format="%Y%m%d%H%M%S")

    # splice the times together
    dt = (end_time - start_time).total_seconds() # total time in seconds

    # theoretical cap at 2hrs of plume lifetime = 7200s, at 10m/s with some grace is 75000m
    # take the smaller of the two, either the real distance or the cap
    dist = min(50000, wspd_avg * dt) # m/s * s = meters

    downwind_angle = (wdir_avg + 180) % 360 # we need downwind so we can use our side angle
    theta = np.radians(90 - downwind_angle) # this is the mathematical angle, not meteorological
    
    xf = x0 + (dist * np.cos(theta))
    yf = y0 + (dist * np.sin(theta))

    # x0, y0 and xf, yf form two corners of a box (in METERS), add a pad to each edge of this box to ensure we capture the plumes
    pad = 5000 # meters

    xlim = (min(x0, xf) - pad, max(x0, xf) + pad)
    ylim = (min(y0, yf) - pad, max(y0, yf) + pad)
    
    # if the x and y axes are too small, expand them to be at least 30km each ( at least 15km from the drone in all directions)
    if (xlim[1] - xlim[0]) < 30000: 
        extra = (30000 - (xlim[1] - xlim[0])) / 2
        xlim = (xlim[0] - extra, xlim[1] + extra)

    if (ylim[1] - ylim[0]) < 30000:
        extra = (30000 - (ylim[1] - ylim[0])) / 2
        ylim = (ylim[0] - extra, ylim[1] + extra)

    zlim = (0, 2500) # most things shouldn't go over 2500m?

    # Print with 2 decimal places using :.2f formatting
    print(
        f"grid limits set to:\n"
        f"  xlim = ({xlim[0]:.2f}, {xlim[1]:.2f})\n"
        f"  ylim = ({ylim[0]:.2f}, {ylim[1]:.2f})\n"
        f"  zlim = ({zlim[0]:.2f}, {zlim[1]:.2f})"
    )

    # compute the diagonal of the grid
    diagonal = np.sqrt((xlim[1] - xlim[0])**2 + (ylim[1] - ylim[0])**2)
    print(f"grid diagonal = {diagonal:.2f} m")


    grid_limits = (zlim, ylim, xlim)

    dz=100

    center = ((xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2) # coordinates of the center of the plot, base grid spacing on this
    dist_to_center = np.sqrt(center[0]**2 + center[1]**2)
    beamwidth = np.deg2rad(0.5)
    beam_size = dist_to_center * beamwidth
    xy_spacing = beam_size * 0.75
    xy_spacing = np.clip(xy_spacing, 150, 600)

    # set the horizontal desired grid spacing (meters)
    dx, dy = xy_spacing, xy_spacing 

    # compute the shape of the grid required to sastisfy the spacing requirement in each dimension
    xshape = int(np.round((max(xlim) - min(xlim)) / dx)) + 1
    yshape = int(np.round((max(ylim) - min(ylim)) / dy)) + 1
    zshape = int(np.round((max(zlim) - min(zlim)) / dz)) + 1

    grid_shape = (zshape, yshape, xshape)

    actual_dx = (max(xlim) - min(xlim)) / (xshape - 1)
    actual_dy = (max(ylim) - min(ylim)) / (yshape - 1)
    actual_dz = (max(zlim) - min(zlim)) / (zshape - 1)

    print(f"xshape = {xshape}, yshape = {yshape}, zshape = {zshape}")
    print(f"target grid spacings = dx = {dx:.2f} m, dy = {dy:.2f} m, dz = {dz:.2f} m")
    print(f"actual grid spacings = dx = {actual_dx:.2f} m, dy = {actual_dy:.2f} m, dz = {actual_dz:.2f} m")


    box = {
        "grid_limits": grid_limits,
        "grid_shape": grid_shape,
        "diagonal": diagonal
    }

    return box



def processOneTilt(filename, grid_shape, grid_limits, config):
    radar = pyart.io.read_nexrad_archive(filename) # read in the radar file once

    # apply the default filter in-place to avoid re-reading the file
    radar = add_default_filtered_fields(radar)

    ref_filtered = radar.fields["reflectivity_filtered"]["data"]

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

    sweep_results = []

    unique_angles, unique_indices = np.unique(radar.fixed_angle['data'], return_index=True)
    sweeps = unique_indices[:6]

    for sweep in sweeps: # could do range(radar.nsweeps) if you want all of them
        sweep_radar = radar.extract_sweeps([sweep])

        grid = pyart.map.grid_from_radars(
                sweep_radar, 
                grid_shape=grid_shape, 
                grid_limits=grid_limits,
                fields = ['reflectivity_filtered'],
                weighting_function=config["grid_weighting_function"], # Barnes2 is the one that makes paintballs, Nearest looks like high res grid radar
                roi_func = config["grid_roi_func"],
                constant_roi = config["grid_constant_roi"], # meters
                gridding_algo='map_gates_to_grid')
    
        #refl = grid.fields["reflectivity_filtered"]["data"][0].filled(np.nan) # fill masked values with NaN
        #sweep_results.append(refl)
        grid_data = grid.fields["reflectivity_filtered"]["data"].filled(np.nan)
        refl_2d = np.ma.getdata(np.nanmax(grid_data, axis=0))
        refl_2d = np.where(np.isnan(refl_2d), np.nan, refl_2d)
        sweep_results.append(refl_2d)


    scantime = pd.to_datetime(radar.time["units"].split("since ")[1])
    selected_elevations = radar.fixed_angle["data"][sweeps]

    return (sweep_results, scantime, grid.x["data"], grid.y["data"], selected_elevations)



def loadGridRadar(filenames, config, box):

    # these come from autoGridAxes data
    grid_limits = box["grid_limits"]
    grid_shape = box["grid_shape"]

    worker = partial(
        processOneTilt,
        grid_shape=grid_shape,
        grid_limits=grid_limits,
        config=config
    )

    with ProcessPoolExecutor(max_workers=8) as executor: # max workers depends on cpu's available on your machine
        results = list(
            tqdm(
                executor.map(worker, filenames),
                total=len(filenames),
                desc="Gridding radar files"
                )
        )

    scans = [r[1] for r in results] # list of scan times

    x = results[0][2] # x coordinates of the grid
    y = results[0][3] # y coordinates of the grid
    elevations = results[0][4] # list of elevations for each sweep

    num_sweeps = len(results[0][0]) # number of sweeps per radar file
    sweep_data = {}

    # Convert timezone-aware pandas Timestamps to timezone-naive numpy.datetime64
    scans = (
        pd.to_datetime(scans)
        .tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )

    for sweep in range(num_sweeps):
        sweep_stack = np.stack(
            [result[0][sweep] for result in results],
            axis=0
        )

        elevation = float(elevations[sweep])
        sweep_data[elevation] = xr.DataArray(
            sweep_stack,
            dims=("time", "y", "x"),
            coords={
                "time": scans,
                "y": y,
                "x": x
            },
            name=f"sweep_{elevation:.2f}º",
        )


    dxy = x[1] - x[0] # calculate grid spacing in meters
    print(f"grid spacing = {dxy}m")
    print()

    return sweep_data, dxy, elevations



def computeThresholds(da_track):
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

    # return a dictionary of the stats instead of 6 variables
    stats = {
        "median_background_noise": median_background_noise,
        "mean_background_noise": mean_background_noise,
        "std_dev": std_dev,
        "iqr": iqr,
        "percent_covered": percent_covered,
        "ref_thresholds": ref_thresholds
    }

    return stats



def detectFeatures(da_track, dxy, stats, config, ymdt):
    min_distance = config["min_distance"]
    sigma_threshold = config["sigma_threshold"]
    ref_thresholds = stats["ref_thresholds"]

    # set up the parameters for feature detection
    parameters_features = {
        "position_threshold": "weighted_diff", # four options here, center, extreme, weighted_diff, and abs_diff. tobac recommends weighted or abs
        "min_distance": min_distance, # meters, min required difference between features. if two features are closer than this, the one with the more extreme value is kept
        "sigma_threshold": sigma_threshold, # gaussian smoothing parameter (scipy.ndimage.gaussian_filter)
        "n_erosion_threshold": 0, # reduces the size of a feature in all direcitons (skimage.morphology.binary_erosion)
        "n_min_threshold": 0, # min number of pixels required for a feature to be detected
        "threshold": ref_thresholds, # lower dbz thresholds for seeding signatures
        "target": "maximum", # looking for local maxima to be marked as features
    }

    # feature_detection_multithreshold outputs a pandas dataframe
    features = tobac.feature_detection_multithreshold(
        da_track,
        dxy=float(dxy), # grid spacing in meters
        **parameters_features
    )

    features["reflectivity"] = [
        da_track.isel(
            time = int(row["frame"]),
            y = int(row["hdim_1"]),
            x = int(row["hdim_2"])
        ).item()
        for _, row in features.iterrows()
    ]

    print(f"features detected: {len(features)}")

    features = features[["frame", "idx", "reflectivity", "num", "y", "x", "hdim_1", "hdim_2", "threshold_value", "feature", "time", "timestr"]]

    saveOutput(features, "featureID", ymdt)

    return features



def segmentFeatures(da_track, features, dxy, config):
    mask, features_mask = tobac.segmentation_2D(
        features,
        da_track,
        dxy,
        threshold=config["segmentation_threshold"],
        max_distance = 5000,
        seed_3D_flag = "box",
    )

    return mask, features_mask



def trackFeatures(da_track, features, dxy, config):
    memory = config["memory"]

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
        v_max= 15.0, # maximum velocity in m/s
        **parameters_tracking,
    )

    return tracks



def filterTracks(stats, tracks, config, radar, ymdt, wind_location):

    median_background_noise = stats["median_background_noise"]
    # calculate the number of frames each track lasts and filter out tracks that are too short to get rid of noise and nonmet features
    track_lengths = tracks.groupby("cell").size()
    #print(track_lengths)
    min_frames = 5 # a feature needs to survive at least n frames to be considered
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

    # fourth, we only want to allow tracks that start within the cone of allowance
    good_cells_cone, cartesian_drone_coords = buildCone(tracks, wind_location)
    
    # merge all the filtering criteria together to get the final list of good cells
    good_cells = good_frames.intersection(good_displacement).intersection(good_ref).intersection(good_cells_cone)
    tracks_filtered = tracks[tracks["cell"].isin(good_cells)]

    # output the filtered tracks to a csv so we can look at exact points and track them better
    tracks_filtered = tracks_filtered[['frame', 'idx', 'reflectivity', 'num', 'cell', 'y', 'x', 'timestr', 'hdim_1', 'hdim_2', 'threshold_value', 'feature', 'time', 'time_cell']] 

    print(f"tracks drawn: {len(tracks_filtered)}")
    print()

    saveOutput(tracks_filtered, "trackID", ymdt)

    return tracks_filtered, cartesian_drone_coords



def mean_wind_direction(wdir_list):
    wdir_rad = np.radians(wdir_list)

    mean_sin = np.mean(np.sin(wdir_rad))
    mean_cos = np.mean(np.cos(wdir_rad))

    mean_dir = np.degrees(np.arctan2(mean_sin, mean_cos))

    return mean_dir % 360



def get_wind(df):
    wspd_list = []
    wdir_list = []

    for index, row in df.iterrows():
        
        if pd.isna(row.iloc[5]) or pd.isna(row.iloc[6]):
            wspd = row.iloc[7]  # Original 24
            wdir = row.iloc[8]  # Original 25
        else:
            wspd = row.iloc[5]  # Original 17
            wdir = row.iloc[6]  # Original 18

        wspd_list.append(wspd)
        wdir_list.append(wdir)

    return wspd_list, wdir_list



def get_site_location(site, radar): 

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



def droneLocation_windVector(config, radar):
    # we will need the drone location, mean wind vector, and the time of seeding
    flight_data = "/Users/ethan1/Desktop/vs_code/Rainmaker/Seeding_Flight_Conditions_MASTER - PDT_Seeding_Flights.csv"

    df = pd.read_csv(flight_data)

    # cols we're interested in: 0 (date), 1(seeding start time), 5 (drone #), 24 (median drone wspd), 25 (median drone wdir), 26/27 (stdev wspd / wdir)
    cols = [0, 1, 5, 15, 16, 17, 18, 24, 25] # 17, 18 are closest sounding wspd and wdir, use these if they exist, otherwise use drone data 24, 25

    drone_df = df.iloc[:, cols] # keep all rows for now, keep only relevant cols

    # save the cleaned up drone data to a csv to make our life easier if we want to look at it
    df.to_csv(f"/Users/ethan1/Desktop/vs_code/Rainmaker/AnalysisData/drone_data_master.csv", index=False)

    # until automation is added, branching statements to choose the right rows
    # we will lock on to the site butter or cabin, easier when there's multiple drones right next to each other all launched from the same site
    sites = {
        "butter": (45.50800, -119.01300),
        "cabin": (45.76400, -118.28100)
    }

    wspd_list = []
    wdir_list = []

    x_site, y_site = get_site_location(sites[config["site"]], radar)
    drone_df = drone_df.iloc[config["rows"], :]
    wspd_list, wdir_list = get_wind(drone_df)

    print(f"radar relative site location: x = {x_site:.2f} m, y = {y_site:.2f} m")

    # for many of these cases, there are multiple drones seeding at various heights all right next to each other
    # they recorded statistically indifferent wind speeds and directions most of the time, so we will take the average of the list as our wspd and wdir
    # in case there is an outlier, we will still print the list of wspds and wdirs used in the average so the user can tell if there is a bad value that needs to be removed
    wspd_avg = np.mean(wspd_list)

    wdir_avg = mean_wind_direction(wdir_list)

    for i in range(len(wspd_list)):
        print(f"wspd: {wspd_list[i]} m/s, wdir: {wdir_list[i]}º")
    print(f"avg wspd: {wspd_avg:.2f} m/s, avg wdir: {wdir_avg:.2f}º")

    wind_location = {
        "wspd_avg": wspd_avg,
        "wdir_avg": wdir_avg,
        "x_site": x_site,
        "y_site": y_site
    }

    return wind_location



def buildCone(tracks, wind_location):
    
    wspd_avg = wind_location["wspd_avg"]
    wdir_avg = wind_location["wdir_avg"]
    x_site = wind_location["x_site"]
    y_site = wind_location["y_site"]

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

    cartesian_drone_coords = {
        "x_site": x_site,
        "y_site": y_site,
        "x_left": x_left,
        "y_left": y_left,
        "x_right": x_right,
        "y_right": y_right
    }

    return good_cells_cone, cartesian_drone_coords



def animateTracks(results, cartesian_drone_coords, ymdt):
    # drone and cone coordinates
    x_site = cartesian_drone_coords["x_site"]
    y_site = cartesian_drone_coords["y_site"]
    x_left = cartesian_drone_coords["x_left"]
    y_left = cartesian_drone_coords["y_left"]
    x_right = cartesian_drone_coords["x_right"]
    y_right = cartesian_drone_coords["y_right"]

    # set the common grid for all plots using the first sweep
    first_result = next(iter(results.values()))
    first_da = first_result["data"]

    xmin_km = first_da.x.values.min() / 1000.0
    xmax_km = first_da.x.values.max() / 1000.0
    ymin_km = first_da.y.values.min() / 1000.0
    ymax_km = first_da.y.values.max() / 1000.0

    nframes = len(first_da.time)

    # create the 2x3 subplot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True, constrained_layout=True)
    axes = axes.flatten()

    # create a dict to store everything associated with one subplot
    plots = {}

    # create shared_image and initialize to none
    shared_image = None

    # create each subplot

    for ax, (elev, result) in zip(axes, results.items()):
        da_track = result["data"]
        tracks_filtered = result["tracks_filtered"]

        image = ax.imshow(
            da_track.isel(time=0),
            origin="lower",
            extent=(xmin_km, xmax_km, ymin_km, ymax_km),
            cmap="NWSRef",
            vmin=-20,
            vmax=70,
            aspect="auto",
        )

        # save the first image to make a shared colorbar after the loop
        if shared_image is None:
            shared_image = image
        
        ax.set_xlim(xmin_km, xmax_km)
        ax.set_ylim(ymin_km, ymax_km)
        ax.set_xlabel("Distance from Radar (km)")
        ax.set_ylabel("Distance from Radar (km)")
        ax.autoscale(False)  # so the cone can't mess with it

        # plot radar location if inside bounds
        radar_marker = None
        if (0 >= xmin_km) and (0 <= xmax_km) and (0 >= ymin_km) and (0 <= ymax_km):
            radar_marker = ax.plot(0, 0, marker="o", color="k", markersize=12, label="Radar")

        # plot drone seeding location
        drone_marker = ax.plot(x_site / 1000.0, y_site / 1000.0, marker="s", color="k", markersize=8, label="Drone")

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

        # scatter artist for tracked features at the current frame
        scatter = ax.scatter([], [], c="black", marker="x", s=100, label="Feature")

        # line artists for plotting tracks
        line_artists = {}

        for track_id in sorted(tracks_filtered["cell"].unique()):
            line, = ax.plot([], [], "-o", label=f"Track {track_id}")
            line_artists[track_id] = line

        # store all of these things in plots for animation
        plots[elev] = {
            "ax": ax,
            "image": image,
            "scatter": scatter,
            "line_artists": line_artists,
            "segment_lines": [],
            "da_track": da_track,
            "tracks_filtered": tracks_filtered,
            "mask_filtered": result["mask"]
        }

        ax.set_title(f"{elev:.2f}º")

        ax.legend(fontsize=7, loc="upper left")

    # create colorbar for all subplots using the shared image
    cbar = fig.colorbar(shared_image, ax=axes, location="right", shrink=0.95, pad=0.02)
    cbar.set_label("Reflectivity (dBZ)")

    # get rid of any plots with nothing in case there are less than 6 tilts
    for ax in axes[len(results):]:
        ax.axis("off")

    def update(frame):
        for elev, plot in plots.items():

            ax = plot["ax"]
            da_track = plot["da_track"]
            tracks_filtered = plot["tracks_filtered"]
            mask_filtered = plot["mask_filtered"]
            image = plot["image"]
            scatter = plot["scatter"]
            track_lines = plot["line_artists"]
            segment_lines = plot["segment_lines"]

            # update the reflectivity image
            image.set_data(da_track.isel(time=frame))

            # remove old segments
            for line in segment_lines:
                line.remove()
            segment_lines.clear()

            # draw new segment contours
            mask = mask_filtered.isel(time=frame).values
            feature_ids = np.unique(mask)
            
            for feature_id in feature_ids:
                if feature_id == 0:
                    continue  # skip background

                contours = find_contours(mask == feature_id, 0.5)

                for contour in contours:
                    y = np.interp(
                        contour[:, 0],
                        np.arange(mask.shape[0]),
                        da_track.y.values / 1000,
                    )
                    x = np.interp(
                        contour[:, 1],
                        np.arange(mask.shape[1]),
                        da_track.x.values / 1000,
                    )

                    line, = ax.plot(
                        x,
                        y,
                        color="black",
                        linewidth=2,
                        zorder=20,
                    )
                    segment_lines.append(line)

            # update each track line to only include points up to the current frame
            for track_id, line in track_lines.items():
                tr = tracks_filtered[
                    (tracks_filtered["cell"] == track_id) &
                    (tracks_filtered["frame"] <= frame)
                ]

                if not tr.empty:
                    line.set_data(
                        tr["x"].values / 1000.0,
                        tr["y"].values / 1000.0
                    )

                else:
                    line.set_data([], [])
            
            # update feature markers
            current = tracks_filtered[tracks_filtered["frame"] == frame]
            if not current.empty:
                offsets = np.column_stack((current["x"].values / 1000.0, current["y"].values / 1000.0))
                scatter.set_offsets(offsets)
            else:
                scatter.set_offsets(np.empty((0, 2)))

            # update title
            timestamp = pd.to_datetime(first_da.time.values[frame]).strftime('%Y-%m-%d %H:%M:%SZ')
            fig.suptitle(f"Reflectivity Seeding Signatures {timestamp}", fontsize=16)
            ax.set_title(f"{elev:.2f}º")

        # return artists for blitting NOTE figure out what blitting is, if not done get rid of this block
        artists = []
        for plot in plots.values():
            artists.append(plot["image"])
            artists.append(plot["scatter"])
            artists.extend(plot["line_artists"].values())
            artists.extend(plot["segment_lines"])

            return artists
        
    # create animation
    ani = FuncAnimation(fig, update, frames=nframes, repeat=False)
    ani.save(f"/Users/ethan1/Desktop/vs_code/Rainmaker/Animations/{ymdt}_AnomRef.gif", writer=PillowWriter(fps=2), dpi=300)

    plt.close(fig)



def saveOutput(df, title, ymdt): #NOTE returns for now, will fix once sweeps are working
    # save features to a csv 
    #df.to_csv(f"/Users/ethan1/Desktop/vs_code/Rainmaker/AnalysisData/{ymdt}_{title}.csv", index=False)
    return



def saveTuningData(stats, box, wind_location, config, ymdt):
    # take all background/env stats and put them in a df to save so we can analyze the differences more easily
    row = {
        "datetime": str(ymdt),
        "median_background_noise": stats["median_background_noise"],
        "mean_background_noise": stats["mean_background_noise"],
        "std_dev": stats["std_dev"],
        "iqr": stats["iqr"],
        "percent_covered": stats["percent_covered"],
        "ref_thresholds": str(stats["ref_thresholds"]),
        "x_min": box["grid_limits"][2][0],
        "x_max": box["grid_limits"][2][1],
        "y_min": box["grid_limits"][1][0],
        "y_max": box["grid_limits"][1][1],
        "z_min": box["grid_limits"][0][0],
        "z_max": box["grid_limits"][0][1],
        "grid_diagonal": box["diagonal"],
        "xshape": box["grid_shape"][2],
        "yshape": box["grid_shape"][1],
        "zshape": box["grid_shape"][0],
        "wspd_avg": wind_location["wspd_avg"],
        "wdir_avg": wind_location["wdir_avg"],
        "segmentation_threshold": config["segmentation_threshold"],
        "sigma_threshold": config["sigma_threshold"],
        "min_distance": config["min_distance"],
    }

    csv_path = f"/Users/ethan1/Desktop/vs_code/Rainmaker/AnalysisData/tuning_data.csv"
    
    row_df = pd.DataFrame([row])

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, dtype={"datetime": str})
    else:
        df = pd.DataFrame()

    # Remove any previous row for this datetime
    df = df[df["datetime"] != row["datetime"]]

    # Add the new row
    df = pd.concat([df, row_df], ignore_index=True)

    # Sort and save
    df.sort_values("datetime", inplace=True)
    df.to_csv(csv_path, index=False)



def main():
    
    filenames, case, radar = selectCase()

    config = loadCaseConfig(case)

    wind_location = droneLocation_windVector(config, radar)
    
    box = autoGridAxes(filenames, wind_location)

    sweep_data, dxy, elevations = loadGridRadar(filenames, config, box)

    results = {} # dict to hold our analysis below

    for elevation, da in sweep_data.items():
        print(f"Processing {elevation:.2f}º sweep")

        stats = computeThresholds(da) 

        ymdt = pd.to_datetime(da.time.values[0]).strftime("%Y%m%d_%H%M%SZ")

        #saveTuningData(stats, box, wind_location, config, ymdt)

        features = detectFeatures(da, dxy, stats, config, ymdt)

        if features.empty:
            print(f"Skipping {elevation:.2f}º due to no features detected\n")

            throwaway, cartesian_drone_coords = buildCone(pd.DataFrame(columns=["cell", "frame", "x", "y"]), wind_location)

            results[elevation] = {
                "data": da,
                "stats": stats,
                "features": features,
                "tracks": pd.DataFrame(columns=["cell", "frame", "x", "y"]),
                "tracks_filtered": pd.DataFrame(columns=["cell", "frame", "x", "y"]),
                "mask": xr.zeros_like(da),
                "features_mask": None
            }

            continue

        tracks = trackFeatures(da, features, dxy, config)

        tracks_filtered, cartesian_drone_coords = filterTracks(stats, tracks, config, radar, ymdt, wind_location)

        mask, features_mask = segmentFeatures(da, tracks_filtered, dxy, config)

        results[elevation] = {
            "data": da,
            "stats": stats,
            "features": features,
            "tracks": tracks,
            "tracks_filtered": tracks_filtered,
            "mask": mask,
            "features_mask": features_mask
        }


    animateTracks(results, cartesian_drone_coords, ymdt)



if __name__ == "__main__":
    main()
