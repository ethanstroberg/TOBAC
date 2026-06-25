Here is a list of things in this latest version of the plume tracker:
- cone of allowance added
    - tracks are only allowed to exist if they begin within the cone extending from the drone
    - cone is calculated from sounding wind speed and direction
- thresholds are now set based on positive dBZ anomalies relative to the background reflectivity
- added a minimm peak reflectivity that a track must hit at some point in its lifetime to be considered a track