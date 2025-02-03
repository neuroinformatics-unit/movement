"""Wrappers to plot movement data."""

import matplotlib.pyplot as plt
import xarray as xr


def occupancy_histogram(
    da: xr.DataArray,
    keypoint: int | str = 0,
    individual: int | str = 0,
    title: str | None = None,
) -> plt.Figure:
    """Create a 2D histogram of the occupancy data given.

    Parameters
    ----------
    da : xarray.DataArray
        Spatial data to create histogram for. NaN values are dropped.
    keypoint : int | str
        The keypoint to create a histogram for.
    individual : int | str
        The individual to create a histogram for.
    title : str, optional
        Title to give to the plot. Default will be generated from the
        ``keypoint`` and ``individual``

    Returns
    -------
    matplotlib.pyplot.Figure
        Plot handle containing the rendered 2D histogram.

    """
    data = da.position if isinstance(da, xr.Dataset) else da
    title_components = []

    # Remove additional dimensions before dropping NaN values
    if "individuals" in data.dims:
        if individual not in data["individuals"]:
            individual = data["individuals"].values[individual]
        data = data.sel(individuals=individual).squeeze()
        title_components.append(f"individual {individual}")
    if "keypoints" in data.dims:
        if keypoint not in data["keypoints"]:
            keypoint = data["keypoints"].values[keypoint]
        data = data.sel(keypoints=keypoint).squeeze()
        title_components.append(f"keypoint {keypoint}")
    # We should now have just the relevant time-space data,
    # so we can remove time-points with NaN values.
    data = data.dropna(dim="time", how="any")
    # This makes us agnostic to the planar coordinate system.
    x_coord = data["space"].values[0]
    y_coord = data["space"].values[1]

    # Now it should just be a case of creating the histogram
    fig, ax = plt.subplots()
    _, _, _, hist_image = ax.hist2d(
        data.sel(space=x_coord), data.sel(space=y_coord)
    )  # counts, xedges, yedges, image
    colourbar = fig.colorbar(hist_image, ax=ax)
    colourbar.solids.set(alpha=1.0)

    # Axis labels and title
    if not title and title_components:
        title = "Occupancy of " + ", ".join(title_components)
    if title:
        ax.set_title(title)
    ax.set_xlabel(x_coord)
    ax.set_ylabel(y_coord)

    return fig


if __name__ == "__main__":
    from movement import sample_data
    from movement.io import load_poses

    ds_path = sample_data.fetch_dataset_paths(
        "SLEAP_single-mouse_EPM.analysis.h5"
    )["poses"]
    position = load_poses.from_sleap_file(ds_path, fps=None).position

    f = occupancy_histogram(position)
    plt.show(block=True)
    pass
