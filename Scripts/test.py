import pyart
import numpy as np

f = "/Users/ethan1/Desktop/vs_code/Rainmaker/RadarFiles/KPDT20260124/KPDT20260124_021642_V06"
radar = pyart.io.read_nexrad_archive(f)

unique_angles, unique_indices = np.unique(radar.fixed_angle['data'], return_index=True)
sweeps = unique_indices[:6]

for sweep in sweeps:
    print(f"Sweep {sweep}: Fixed Angle = {radar.fixed_angle['data'][sweep]} degrees")