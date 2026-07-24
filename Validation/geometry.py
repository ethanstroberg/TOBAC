"""
coordinate grids, dynamic bounting, geometric calculations
"""

import numpy as np
import pandas as pd
import os


def auto_grid_axes(filenames, wind_location): 
    
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

    zlim = (500, 2000) # we want to cover the 500-2000m zone with out CAPPIs

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

    dz=300

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
        
        if pd.isna(row.iloc[6]) or pd.isna(row.iloc[7]):
            wspd = row.iloc[8]  # Original 24
            wdir = row.iloc[9]  # Original 25
        else:
            wspd = row.iloc[6]  # Original 17
            wdir = row.iloc[7]  # Original 18

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



def drone_location_wind_vector(config, radar):
    # we will need the drone location, mean wind vector, and the time of seeding
    flight_data = "/Users/ethan1/Desktop/vs_code/Rainmaker/Seeding_Flight_Conditions_MASTER - PDT_Seeding_Flights.csv"

    df = pd.read_csv(flight_data)

    # cols we're interested in: 0 (date), 1(seeding start time), 3 (site name), 5 (drone #), 24 (median drone wspd), 25 (median drone wdir), 26/27 (stdev wspd / wdir)
    cols = [0, 1, 3, 5, 15, 16, 17, 18, 24, 25] # 17, 18 are closest sounding wspd and wdir, use these if they exist, otherwise use drone data 24, 25

    drone_df = df.iloc[:, cols] # keep all rows for now, keep only relevant cols

    # save the cleaned up drone data to a csv to make our life easier if we want to look at it
    drone_df.to_csv(f"/Users/ethan1/Desktop/vs_code/Rainmaker/AnalysisData/drone_data_master.csv", index=False)

    # for now, we will choose the site based on the rows chosen from the csv that are listed in the config file
    # we will lock on to the site butter or cabin, easier when there's multiple drones right next to each other all launched from the same site
    sites = {
        "Butter": (45.50800, -119.01300),
        "Cabin": (45.76400, -118.28100),
        "Toast": (45.433, -118.834)
    }

    wspd_list = []
    wdir_list = []

    drone_df = drone_df.iloc[config["rows"], :] # narrow down to only the case-specific rows
    
    # obtain the site name from the df, they SHOULD all be the same most of the time, but in case they aren't, take the more common one under the assumption they are close together
    site = drone_df.iloc[:, 2].mode().iloc[0] # get the mode of the site names
    print("site: ", site)

    # get the wind and drone info
    wspd_list, wdir_list = get_wind(drone_df)
    x_site, y_site = get_site_location(sites[site], radar)

    print(f"radar relative site location: x = {x_site:.2f} m, y = {y_site:.2f} m")

    # for many of these cases, there are multiple drones seeding at various heights all right next to each other
    # they recorded statistically indifferent wind speeds and directions most of the time, so we will take the average of the list as our wspd and wdir
    # in case there is an outlier, we will still print the list of wspds and wdirs used in the average so the user can tell if there is a bad value that needs to be removed
    wspd_avg = np.mean(wspd_list)

    wdir_avg = mean_wind_direction(wdir_list)

    for i in range(len(wspd_list)):
        print(f"wspd: {wspd_list[i]} m/s, wdir: {wdir_list[i]}º")
    print(f"avg wspd: {wspd_avg:.2f} m/s, avg wdir: {wdir_avg:.2f}º")

    # also pull the seeding time(s) from the dataframe
    seeding_date_time = drone_df.iloc[:, [0, 1]]
    dtimes = pd.to_datetime(seeding_date_time.iloc[:, 0].astype(str) + " " + seeding_date_time.iloc[:, 1].astype(str))

    wind_location_time = {
        "wspd_avg": wspd_avg,
        "wdir_avg": wdir_avg,
        "x_site": x_site,
        "y_site": y_site,
        "seeding_times": dtimes
    }

    return wind_location_time
