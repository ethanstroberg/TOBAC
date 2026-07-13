"""
all radar related processing, gridding the data and processing elevation tilts
"""
import numpy as np
import pandas as pd
import xarray as xr
import pyart
from default_radar_filter import add_default_filtered_fields
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def process_one_tilt(filename, grid_shape, grid_limits):
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
                weighting_function= "Barnes2", # Barnes2 is the one that makes paintballs, Nearest looks like high res grid radar
                roi_func = "constant",
                constant_roi = 400, # meters
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



def load_grid_radar(filenames, box):

    # these come from autoGridAxes data
    grid_limits = box["grid_limits"]
    grid_shape = box["grid_shape"]

    worker = partial(
        process_one_tilt,
        grid_shape=grid_shape,
        grid_limits=grid_limits
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
