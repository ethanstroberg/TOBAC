"""
main driver for automatic radar validation of cloud seeding signatures
@author: Ethan Stroberg
@date: 7/14/26
"""

# import support .py files
from io_utils import select_case, load_case_config, merge_metadata
import tracking as tr
import pandas as pd
import xarray as xr

def main():

    # switch to use either sweeps (True) or CAPPI (False)
    use_sweeps = False
    
    if use_sweeps:
        from geometry_sweep import drone_location_wind_vector, auto_grid_axes
        from animation_sweep import animate_tracks
        from radar_processing_sweep import load_grid_radar

    else:
        from geometry import drone_location_wind_vector, auto_grid_axes, build_cone_only
        from animation import animate_tracks
        from radar_processing import load_grid_radar

    
    filenames, caseName, radar = select_case()

    config = load_case_config(caseName)

    wind_location_time = drone_location_wind_vector(config, radar)

    box = auto_grid_axes(filenames, wind_location_time)

    sweep_data, dxy, elevations = load_grid_radar(filenames, box)

    drone_coords = build_cone_only(wind_location_time)

    results = {}

    for elevation, da in sweep_data.items():
        print(f"Processing {elevation:.2f}m sweep")

        stats = tr.compute_thresholds(da)

        ymdt = pd.to_datetime(da.time.values[0]).strftime("%Y%m%d_%H%M%SZ")

        features = tr.detect_features(da, dxy, stats)

        if features.empty:
            print(f"Skipping {elevation:.2f}m due to no features detected\n")

            throwaway, cartesian_drone_coords = tr.build_cone(pd.DataFrame(columns=["cell", "frame", "x", "y"]), wind_location_time)

            results[elevation] = {
                "data": da,
                "stats": stats,
                "features": features,
                "tracks": pd.DataFrame(columns=["cell", "frame", "x", "y"]),
                "tracks_filtered": pd.DataFrame(columns=["cell", "frame", "x", "y"]),
                "mask": xr.zeros_like(da),
                "features_mask": pd.DataFrame()
            }

            continue
        # NOTE this section is altered currently to test TINT as the tracking algorithm
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        mask, features_mask = tr.segment_features(da, features, dxy, stats) # NOTE changed param 2 from tracks_filtered to features ... also swapped tracking and segmentation
        
        tracks = tr.track_features(da, features_mask, dxy, stats)

        tracks_filtered, cartesian_drone_coords = tr.filter_tracks(stats, tracks, wind_location_time)

        good_cell_ids = tracks_filtered["feature"].unique()

        clean_mask = mask.where(mask.isin(good_cell_ids), 0)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        results[elevation] = {
            "data": da,
            "stats": stats,
            "features": features,
            "tracks": tracks,
            "tracks_filtered": tracks_filtered,
            "mask": clean_mask,
            "features_mask": features_mask
        }
    
    # combine the features dataframes into one, and the tracks_filtered dataframes into one
    # also add a height column to each df so we can identify which elevation each feature/track came from
    features_df = pd.concat([result["features"].assign(height=elevation) for elevation, result in results.items()], ignore_index=True)
    tracks_filtered_df = pd.concat([result["tracks_filtered"].assign(height=elevation) for elevation, result in results.items()], ignore_index=True)
    seg_df = pd.concat([result["features_mask"].assign(height=elevation) for elevation, result in results.items()], ignore_index=True)
    
    features_df.insert(0, "height", features_df.pop("height"))
    tracks_filtered_df.insert(0, "height", tracks_filtered_df.pop("height"))
    seg_df.insert(0, "height", seg_df.pop("height"))

    seg_df = merge_metadata(seg_df, results, config)
    
    tr.save_output(features_df, "featuresID", ymdt)
    tr.save_output(tracks_filtered_df, "tracksID", ymdt)
    tr.save_output(seg_df, "segID", ymdt)

    # animate figures
    animate_tracks(results, cartesian_drone_coords, ymdt)


if __name__ == "__main__":
    main()
