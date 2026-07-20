
import os
import glob
import pyart
from config import CASE_CONFIGS
import numpy as np

def select_case():
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



def load_case_config(caseName): 
    return CASE_CONFIGS.get(caseName, None)



def merge_metadata(df, results, stats):
    # add metadata to the df so all variables live in one place
    df = df.copy()

    # create empty cols for formatting
    df["median_background_noise"] = np.nan
    df["mean_background_noise"] = np.nan
    df["std_dev"] = np.nan
    df["iqr"] = np.nan
    df["percent_covered"] = np.nan

    for elevation, result in results.items():
        stats = result["stats"]

        mask = df["height"] == elevation

        df.loc[mask, "median_background_noise"] = stats["median_background_noise"]
        df.loc[mask, "mean_background_noise"] = stats["mean_background_noise"]
        df.loc[mask, "std_dev"] = stats["std_dev"]
        df.loc[mask, "iqr"] = stats["iqr"]
        df.loc[mask, "percent_covered"] = stats["percent_covered"]

    df["sigma_threshold"] = stats["sigma_threshold"]
    df["segmentation_threshold"] = stats["segmentation_threshold"]

    return df

