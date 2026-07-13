"""
creating the subplots and animating the radar data + tracks + segments
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import trackpy as tp
tp.quiet() # turn off trackpy warnings/messages
from skimage.measure import find_contours
import warnings
warnings.filterwarnings("ignore")

def animate_tracks(results, cartesian_drone_coords, ymdt):
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
            fig.suptitle(f"CAPPI Reflectivity Seeding Signatures {timestamp}", fontsize=16)
            ax.set_title(f"{elev:.2f}m")

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
    ani.save(f"/Users/ethan1/Desktop/vs_code/Rainmaker/Animations/{ymdt}_CAPPI.gif", writer=PillowWriter(fps=2), dpi=300)

    plt.close(fig)


