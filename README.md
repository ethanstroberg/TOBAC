# Radar Auto-Validation of Cloud Seeding

This module takes in radar V06 files and mission-specific seeding data to identify, track, segment, and validate cloud seeding signatures in radar reflectivity. After processing, the program outputs a csv with bulk statistics for each tracked segment. You will need to have the required input drone/mission data in csv form to pass to the program. There is a toggle to choose either sweep-based analysis or constant altitude analysis, and this choice is reflected in the existence of "sweep" verisons of a few files. 

Code is in the Validation directory.

Full list of required data:
- all V06 radar files pertinent to your case
- csv containing the following:
    - IOP date
    - IOP seeding start time
    - Seeding Site ID
    - Closest Sounding Wind Speed (m/s)
    - Closest Sounding Wind Direction (m/s)
    - Median Drone Wind Speed during seeding (m/s)
    - Median Drone Wind Direction during seeding (deg from N)
