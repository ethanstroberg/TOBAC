"""
all TOBAC related processing plus initial statistics/threshold computing
"""

import numpy as np
import pandas as pd
from scipy import stats
import tobac 
import trackpy as tp
tp.quiet() # turn off trackpy warnings/messages
from tobac.utils import get_statistics_from_mask
import warnings
warnings.filterwarnings("ignore")


def compute_thresholds(da_track):

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

    # calculate the percent of the grid that has any ref signal at all (ref has a floor of -5dBZ)
    percent_covered = float((da_track > -5).sum(dim=("time", "y", "x")).values) / float(da_track.size) * 100.0
    print(f"percent covered = {percent_covered:.2f}%")

    # set thresholds with some low enough to maintain detection of a decaying plume, set a minimum peak later in the code
    ref_thresholds = [median_background_noise +3,
                    median_background_noise + 8,
                    median_background_noise + 13]
    
    if percent_covered >= 35.0: # if super noisy, increase the thresholds to avoid giant features
        ref_thresholds = [rt * 1.5 for rt in ref_thresholds]


    print(f"reflectivity thresholds set to: {', '.join(f'{x:.2f}' for x in ref_thresholds)} dBZ")

    # segmentation threshold
    if percent_covered < 5.0:
        segmentation_threshold = -5
    elif percent_covered < 10.0:
        segmentation_threshold = 2
    elif percent_covered < 15.0:
        segmentation_threshold = 5
    elif percent_covered < 20.0:
        segmentation_threshold = 8
    elif percent_covered < 35.0:
        segmentation_threshold = 10
    elif percent_covered < 50.0:
        segmentation_threshold = 15
    else: # if larger than 50%
        segmentation_threshold = 20

    print(f"segmentation threshold set to: {segmentation_threshold:.2f} dBZ")

    # memory -- allow a missed frame if the background noise is minimal enough... this is too risky with a lot of noise
    if percent_covered <= 10.0:
        memory = 1
    else:
        memory = 0

    # return a dictionary of the stats instead of 6 variables
    stats = {
        "median_background_noise": median_background_noise,
        "mean_background_noise": mean_background_noise,
        "std_dev": std_dev,
        "iqr": iqr,
        "percent_covered": percent_covered,
        "ref_thresholds": ref_thresholds,
        "segmentation_threshold": segmentation_threshold,
        "sigma_threshold": 0.3,
        "memory": memory
    }

    return stats



def detect_features(da_track, dxy, stats):
    ref_thresholds = stats["ref_thresholds"]
    sigma_threshold = stats["sigma_threshold"]

    # set up the parameters for feature detection
    parameters_features = {
        "position_threshold": "weighted_diff", # four options here, center, extreme, weighted_diff, and weighted_abs. tobac recommends weighted or abs
        "min_distance": 10000, # meters, min required difference between features. if two features are closer than this, the one with the more extreme value is kept
        "sigma_threshold": sigma_threshold, # gaussian smoothing parameter (scipy.ndimage.gaussian_filter)
        "n_erosion_threshold": 1, # reduces the size of a feature in all direcitons (skimage.morphology.binary_erosion)
        "n_min_threshold": 3, # min number of pixels required for a feature to be detected
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
    print()

    features = features[["frame", "idx", "reflectivity", "num", "y", "x", "hdim_1", "hdim_2", "threshold_value", "feature", "time", "timestr"]]

    return features



def segment_features(da_track, features, dxy, stats):

    mask, features_mask = tobac.segmentation_2D(
        features,
        da_track,
        dxy,
        threshold=stats["segmentation_threshold"],
        max_distance = 5000,
        seed_3D_flag = "box",
    )

    return mask, features_mask



def track_features(da_track, features, dxy, stats, wind_location_time):

    # calculate the median time step... they are all nearly the same so this is ok
    # we just need to be able to find velocity so tobac can track speed
    times = pd.to_datetime(da_track.time.values)

    # robustly compute median time step in seconds; handle cases where times may be object-dtype
    try:
        # preferred fast path using numpy datetime64 array
        dt = float(np.median(np.diff(times.values) / np.timedelta64(1, "s")))
    except Exception:
        # fallback to pandas timedelta median
        dt = float(pd.Series(times).diff().median() / np.timedelta64(1, "s"))

    # set up the parameters for tracking in time and space
    parameters_tracking = {
        "method_linking": "predict",
        "adaptive_stop": 0.2,
        "adaptive_step": 0.95,
        "extrapolate": 0, # this will extrapolate the initiation and decay of a track at levels below the minimum detection threshold (this apparently is not yet implemented in tobac, don't use)
        "order": 1,
        "subnetwork_size": 30,
        "memory": stats["memory"], # how long a feature can disappear for before we stop trying to link it to a track
        "stubs": 3 # minimum timesteps for a tracked cell to be reported
    }

    
    # tracks is also a pandas dataframe
    tracks = tobac.linking_trackpy(
        features,
        None,
        dt=dt,
        dxy=dxy,
        v_max= 11.0, # maximum velocity in m/s
        **parameters_tracking,
    )

    return tracks



def filter_tracks(stats, tracks, wind_location_time):

    median_background_noise = stats["median_background_noise"]
    # we want a minimum track length to filter out stationary or short tracks that are less than the min_length
    min_length = 5000 # meters
    track_displacements = tracks.groupby("cell").apply(lambda x: np.sqrt((x["x"].iloc[-1] - x["x"].iloc[0])**2 + (x["y"].iloc[-1] - x["y"].iloc[0])**2))
    good_displacement = track_displacements[track_displacements >= min_length].index

    # second, we want to make it so that a feature must pass a certain threshold to be detected, but can decay and still be tracked as it decays
    # set lower threshold levels above, but set a minimum peak ref value that must be passed to count the track as a good cell
    required_ref = median_background_noise + 15 # plume must at some point exceed this threshold
    peak_ref = tracks.groupby("cell")["reflectivity"].max()
    good_ref = peak_ref[peak_ref >= required_ref].index

    # third, we only want to allow tracks that start within the cone of allowance
    good_cells_cone, cartesian_drone_coords = build_cone(tracks, wind_location_time)

    # merge all the filtering criteria together to get the final list of good cells
    good_cells = good_displacement.intersection(good_ref).intersection(good_cells_cone)
    tracks_filtered = tracks[tracks["cell"].isin(good_cells)]

    # output the filtered tracks to a csv so we can look at exact points and track them better
    tracks_filtered = tracks_filtered[['frame', 'idx', 'reflectivity', 'num', 'cell', 'y', 'x', 'timestr', 'hdim_1', 'hdim_2', 'threshold_value', 'feature', 'time', 'time_cell']] 

    # remove any track cells with cell == -1 from trakcs_filtered
    tracks_filtered = tracks_filtered[tracks_filtered['cell'] != -1]
    
    print(f"tracks drawn: {len(tracks_filtered)}")
    print()

    return tracks_filtered, cartesian_drone_coords




def build_cone(tracks, wind_location):
    
    wspd_avg = wind_location["wspd_avg"]
    wdir_avg = wind_location["wdir_avg"]
    x_site = wind_location["x_site"]
    y_site = wind_location["y_site"]

    # a plume will show up within 45 mins usually, set a max distance
    max_dist = wspd_avg * 2400 # max allowed dist of cone in meters --> speed (m/s) * time (s) = distance (m)
    half_angle = 25 # deg to each side of the mean wdir
    downwind_angle = (wdir_avg + 180) % 360 # we need downwind so we can use our side angle
    theta = np.radians(90 - downwind_angle) # this is the mathematical angle, not meteorological

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



def save_output(df, title, ymdt):
    # save features to a csv 
    df.to_csv(f"/Users/ethan1/Desktop/vs_code/Rainmaker/AnalysisData/{ymdt}_{title}.csv", index=False)
    return



def get_bulk_statistics(df, results, dxy):
    '''compute various statistics on the tracked segments'''
    # we need the mask from results for QPE: apply the Z-R relationship to each pixel in the
    # full segmented plume footprint for each feature (instead of just the pixel used to ID it),
    # then aggregate the per-pixel rain rates directly, so the nonlinear Z-R transform is never
    # applied to an already-averaged Z (which would bias the result, per Jensen's inequality)
    df_with_plume_r = []
    for elevation, result in results.items():
        elevation_df = df[df["height"] == elevation]
        if elevation_df.empty:
            continue

        df_with_plume_r.append(
            get_statistics_from_mask(
                elevation_df,
                result["mask"],
                result["data"], # this is the reflectivity array
                statistic={
                    "plume_mean_r": lambda vals: np.mean((10 ** (vals / 10) / 200) ** (1 / 1.6)),
                    "plume_sum_r": lambda vals: np.sum((10 ** (vals / 10) / 200) ** (1 / 1.6)),
                },
                id_column="feature",
            )
        )

    df = pd.concat(df_with_plume_r, ignore_index=True)

    stats = df.groupby(["height", "cell"]).apply(lambda group: _group_stats(group, dxy)).reset_index()

    return stats



def _group_stats(group, dxy):
    motion_speed_ms, motion_dir_deg = _compute_motion_vector(group)

    # every statistic that we want to compute
    return pd.Series({
        "mean_dbz": group["reflectivity"].mean(),
        "std_dbz": group["reflectivity"].std(),
        "median_dbz": group["reflectivity"].median(),
        "max_dbz": group["reflectivity"].max(),
        "min_dbz": group["reflectivity"].min(),
        "max_px": group["ncells"].max(skipna=True),
        "min_px": group["ncells"].min(skipna=True),
        "avg_px": group["ncells"].mean(skipna=True),
        "max_area (km^2)": group["ncells"].max(skipna=True) * (dxy ** 2) / 1e6,
        "min_area (km^2)": group["ncells"].min(skipna=True) * (dxy ** 2) / 1e6,
        "avg_area (km^2)": group["ncells"].mean(skipna=True) * (dxy ** 2) / 1e6,
        "QPE accumulation (mm)": _compute_qpe(group["plume_mean_r"], group["time"]),
        "total vol precip (m^3)": _compute_integrated_volume(group["plume_sum_r"], group["time"], dxy),
        "avg speed (m/s)": motion_speed_ms,
        "avg dir (deg from N)": motion_dir_deg, # cardinal convention, 0/360º = due north, 90º = due east etc
    })



def _compute_motion_vector(group):
    '''average the per-frame velocity vectors (dx/dt, dy/dt) across the track to get a mean
    motion vector, then convert to speed (m/s) and meteorological direction (deg, 0/360 = North,
    increasing clockwise). averaging the u/v components -- rather than averaging speed and
    direction separately -- avoids wraparound issues when a track's heading crosses due north,
    and is more robust to noise than just using the start/end points'''
    order = group["time"].sort_values().index
    x = group["x"].loc[order]
    y = group["y"].loc[order]
    time = group["time"].loc[order]

    dt = time.diff().dt.total_seconds()
    dx = x.diff()
    dy = y.diff()

    valid = dt > 0
    u = dx[valid] / dt[valid]
    v = dy[valid] / dt[valid]

    if u.empty:
        return np.nan, np.nan

    mean_u = u.mean()
    mean_v = v.mean()

    speed = float(np.sqrt(mean_u ** 2 + mean_v ** 2))
    direction = float(np.degrees(np.arctan2(mean_u, mean_v)) % 360)

    return speed, direction



def _compute_qpe(mean_r, time):
    '''take time and the per-pixel-averaged rain rate (Z-R relationship already applied per
    pixel before averaging) and integrate over time to get accumulated depth in mm'''
    order = time.sort_values().index

    R = mean_r.loc[order]  # mm/hr, mean of per-pixel rain rates

    # get the dt in hours
    dt = (
        time.loc[order].diff().dt.total_seconds().fillna(0) / 3600
    )
    # integrate
    return np.sum(R * dt)



def _compute_integrated_volume(sum_r, time, dxy):
    '''take time and the per-pixel-summed rain rate (Z-R relationship already applied per
    pixel before summing) to compute the total precipitation volume swept out by the
    segment over its whole lifetime, accumulated pixel-by-pixel'''
    order = time.sort_values().index
    time_sorted = time.loc[order]

    R_sum = sum_r.loc[order]  # mm/hr, sum of per-pixel rain rates across the footprint

    dt = time_sorted.diff().dt.total_seconds() / 3600
    dt.iloc[0] = 0  # no previous time point for the first entry

    depth_sum_m = (R_sum * dt) / 1000  # sum over pixels of accumulated depth per timestep, m

    return float((depth_sum_m * dxy ** 2).sum())  # each pixel has area dxy^2