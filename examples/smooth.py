"""Smoothing
============

Smoothing pose tracks using median and Savitzky-Golay filters.
"""

# %%
# Imports
# -------

from matplotlib import pyplot as plt
from scipy.signal import welch

from movement import sample_data
from movement.filtering import (
    interpolate_over_time,
    median_filter,
)

# %%
# Load some sample datasets
# -------------------------

ds_wasp = sample_data.fetch_dataset("DLC_single-wasp.predictions.h5")
ds_mouse = sample_data.fetch_dataset("SLEAP_single-mouse_EPM.analysis.h5")

# %%
# Let's inspect the loaded datasets

ds_wasp

# %%
# We see that the wasp dataset has a single individual with two keypoints,
# ``head`` and ``thorax`` tracked in 2D space. The video was recorded at 40 fps
# and lasts for ~27 seconds.
#
# Now let's take a look at the mouse dataset.


ds_mouse

# %%
# The mouse dataset has a single individual with six keypoints tracked in 2D
# space. The video was recorded at 30 fps and lasts for ~616 seconds. We can
# see that the data contains some missing (nan) values.
#
# Note also that the ``time unit``` for both datasets is seconds.


# %%
# Define some plotting functions
# ------------------------------
# Let us define some plotting functions that will help us visualise the
# effects of smoothing.


def plot_raw_and_smooth_timeseries_and_psd(
    ds_raw,
    ds_smooth,
    individual="individual_0",
    keypoint="stinger",
    time_range=None,
):
    """Plot position time series and PSD before and after smoothing.

    Only the x coortinate of the chosen keypoint is plotted.
    """
    # If not time range is provided, plot the entire time series
    if time_range is None:
        time_range = slice(0, ds_raw.time[-1])
    selection = {
        "time": time_range,
        "individuals": individual,
        "keypoints": keypoint,
        "space": "x",
    }

    pos_raw = ds_raw.position.sel(**selection)
    pos_smooth = ds_smooth.position.sel(**selection)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Plot the time series for the x coordinate
    ax[0].plot(
        pos_raw.time,
        pos_raw,
        color="k",
        lw=2,
        alpha=0.7,
        label="raw x",
    )
    ax[0].plot(
        pos_smooth.time,
        pos_smooth,
        color="r",
        lw=2,
        alpha=0.7,
        label="smooth x",
    )
    ax[0].legend()
    ax[0].set_ylabel("x position (px)")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_title("Time Domain")

    # Generate interpolated datasets to avoid NaNs in the PSD calculation
    ds_raw_interp = interpolate_over_time(
        ds_raw, max_gap=None, print_report=False
    )
    ds_smooth_interp = interpolate_over_time(
        ds_smooth, max_gap=None, print_report=False
    )
    pos_raw_interp = ds_raw_interp.position.sel(**selection)
    pos_smooth_interp = ds_smooth_interp.position.sel(**selection)

    # Calculate and plot the PSD for the x coordinate
    f_raw, Pxx_raw = welch(
        pos_raw_interp,
        fs=ds_raw.fps,
        nperseg=256,
    )
    f_smooth, Pxx_smooth = welch(
        pos_smooth_interp,
        fs=ds_smooth.fps,
        nperseg=256,
    )

    ax[1].semilogy(
        f_raw,
        Pxx_raw,
        color="k",
        lw=2,
        alpha=0.7,
        label="raw x",
    )
    ax[1].semilogy(
        f_smooth,
        Pxx_smooth,
        color="r",
        lw=2,
        alpha=0.7,
        label="smooth x",
    )
    ax[1].legend()
    ax[1].set_ylabel("PSD (dB/Hz)")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_title("Frequency Domain")

    plt.tight_layout()
    plt.show()


# %%
# Smooth using median filter
# --------------------------
# Here we apply a rolling window median filter to the wasp dataset.
# The ``window_length`` parameter is defined in seconds (according to the
# ``time_unit``` dataset attribute).

ds_wasp_medfilt = median_filter(ds_wasp, window_length=0.1)

# %%
# Let's visualise the effects of the median filter in the time domain.

plot_raw_and_smooth_timeseries_and_psd(
    ds_wasp, ds_wasp_medfilt, keypoint="stinger"
)

# %%
# We see that the median filter has removed the "spikes" present around the
# 14 second mark in the raw data. However, it has not dealt the big shift
# occurring during the final second. In the frequency domain, we can see that
# the filter has reduced the power in the high frequencies, without
# affecting the very low frequency components.
#
# This illustrated what the median filter is good at: removing brief "spikes"
# (e.g. a keypoint abruptly jumping to a different location for a frame or two)
# and high frequency "jitter" (which is often a consequence of pose estimation
# working on a per-frame basis). In general, using the median filter is a good
# idea, but you should choose ``window_length`` conservatively to avoid
# removing relevant information. Always inspect the results (as we are doing
# here) to ensure that the filter is not removing important features.

# %%
# What happens if the data contains missing values (NaNs). Let's apply the
# same filter to the mouse dataset and see.

ds_mouse_medfilt = median_filter(ds_mouse, window_length=0.1)

# %%
# The report informs that the raw data contains some NaN values, particularly
# for the ``snout`` and ``tail_end``` keypoints. After filtering, the number of
# NaNs has increased. This is because the the default behaviour of the median
# filter is to propagate NaN values, i.e. if any value in the rolling window is
# NaN, the output will also be NaN. To modify this behaviour, you can set the
# value of the ``min_periods`` parameter to an integer value. This parameter
# determines the minimum number of non-NaN values in the window for the output
# to be non-NaN. For example, setting ``min_periods=2`` would mean that two
# non-NaN values in the window are sufficient for the median to be calculated
# (based only on those non-NaN values).

ds_mouse_medfilt = median_filter(ds_mouse, window_length=0.1, min_periods=2)

# %%
# We see that this time the number of NaN values has not increased after
# filtering. Instead, it has even decreased by a bit across keypoints.
# Let's visualise the effects of the median filter in the time and frequency
# domains. Here we focus on a 40 second time range for the `snout` keypoint.
# You can adjust the ``keypoint`` and ``time_range`` arguments to explore other
# parts of the data.

plot_raw_and_smooth_timeseries_and_psd(
    ds_mouse, ds_mouse_medfilt, keypoint="snout", time_range=slice(0, 40)
)
