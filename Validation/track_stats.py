# test running stats on the csvs
import pandas as pd

def track_stats(df):
    path = "/Users/ethan1/Desktop/vs_code/Rainmaker/AnalysisData/"
    df = pd.read_csv(path + "20260124_014146Z_tracksID.csv")

    # compute track statistics that may be useful for determining segmentation stuff
    # compute the stats by track by height
    #for every unique height in the dataframe, compute the stats
    results = {}

    for height in sorted(df["height"].unique()):

        df_height = df[df["height"] == height]

        stats = (
            df_height.groupby("cell")["reflectivity"]
            .agg(
                mean="mean",
                median="median",
                maximum="max",
                minimum="min",
                std="std",
                lifetime="count",
            )
        )

        results[height] = stats

    return results