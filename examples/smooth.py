"""Smooth pose tracks
=====================

Smooth pose tracks using the median and Savitzky-Golay filters.
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
    savgol_filter,
)

# %%
# Load a sample dataset
# ---------------------
# Let's load a sample dataset and print it to inspect its contents.
# Note that if you are running this notebook interactively, you can simply
# type the variable name (here ``ds_wasp``) in a cell to get an interactive
# display of the dataset's contents.

ds_wasp = sample_data.fetch_dataset("DLC_single-wasp.predictions.h5")
print(ds_wasp)

# %%
# We see that the dataset contains a single individual (a wasp) with two
# keypoints tracked in 2D space. The video was recorded at 40 fps and lasts for
# ~27 seconds.

# %%
# Define a plotting function
# --------------------------
# Let's define a plotting function to help us visualise the effects smoothing
# both in the time and frequency domains.
# The function takes as inputs two datasets containing raw and smooth data
# respectively, and plots the position time series and power spectral density
# (PSD) for a given individual and keypoint. The function also allows you to
# specify the spatial coordinate (``x`` or ``y``) and a time range to focus on.


def plot_raw_and_smooth_timeseries_and_psd(
    ds_raw,
    ds_smooth,
    individual="individual_0",
    keypoint="stinger",
    space="x",
    time_range=None,
):
    # If no time range is specified, plot the entire time series
    if time_range is None:
        time_range = slice(0, ds_raw.time[-1])

    selection = {
        "time": time_range,
        "individuals": individual,
        "keypoints": keypoint,
        "space": space,
    }

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    for ds, color, label in zip(
        [ds_raw, ds_smooth], ["k", "r"], ["raw", "smooth"]
    ):
        # plot position time series
        pos = ds.position.sel(**selection)
        ax[0].plot(
            pos.time,
            pos,
            color=color,
            lw=2,
            alpha=0.7,
            label=f"{label} {space}",
        )

        # generate interpolated dataset to avoid NaNs in the PSD calculation
        ds_interp = interpolate_over_time(ds, max_gap=None, print_report=False)
        pos_interp = ds_interp.position.sel(**selection)
        # compute and plot the PSD
        freq, psd = welch(pos_interp, fs=ds.fps, nperseg=256)
        ax[1].semilogy(
            freq,
            psd,
            color=color,
            lw=2,
            alpha=0.7,
            label=f"{label} {space}",
        )

    ax[0].set_ylabel(f"{space} position (px)")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_title("Time Domain")
    ax[0].legend()

    ax[1].set_ylabel("PSD (dB/Hz)")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_title("Frequency Domain")
    ax[1].legend()

    plt.tight_layout()
    fig.show()


# %%
# Smoothing with a median filter
# ------------------------------
# Here we use the :py:func:`movement.filtering.median_filter` function to
# apply a rolling window median filter to the wasp dataset.
# The ``window_length`` parameter is defined in seconds (according to the
# ``time_unit`` dataset attribute).

ds_wasp_medfilt = median_filter(ds_wasp, window_length=0.1)

# %%
# We see from the printed report that the dataset has no missing values
# neither before nor after smoothing. Let's visualise the effects of the
# median filter in the time and frequency domains.

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
# This illustrates what the median filter is good at: removing brief "spikes"
# (e.g. a keypoint abruptly jumping to a different location for a frame or two)
# and high frequency "jitter" (often present due to pose estimation
# working on a per-frame basis).

# %%
# Choosing parameters for the median filter
# -----------------------------------------
# You can control the behaviour of :py:func:`movement.filtering.median_filter`
# via two parameters: ``window_length`` and ``min_periods``.
# To better understand the effect of these parameters, let's use a
# dataset that contains missing values.

ds_mouse = sample_data.fetch_dataset("SLEAP_single-mouse_EPM.analysis.h5")
print(ds_mouse)

# %%
# The dataset contains a single mouse with six keypoints tracked in
# 2D space. The video was recorded at 30 fps and lasts for ~616 seconds. We can
# see that there are some missing values, indicated as "nan" in the
# printed dataset. Let's apply the median filter to this dataset, with
# the ``window_length`` set to 0.1 seconds.

ds_mouse_medfilt = median_filter(ds_mouse, window_length=0.1)

# %%
# The report informs us that the raw data contains NaN values, particularly
# for the ``snout`` and ``tail_end`` keypoints. After filtering, the number of
# NaNs has increased. This is because the default behaviour of the median
# filter is to propagate NaN values, i.e. if any value in the rolling window is
# NaN, the output will also be NaN.
#
# To modify this behaviour, you can set the value of the ``min_periods``
# parameter to an integer value. This parameter determines the minimum number
# of non-NaN values in the window for the output to be non-NaN.
# For example, setting ``min_periods=2`` means that two non-NaN values in the
# window are sufficient for the median to be calculated. Let's try this.

ds_mouse_medfilt = median_filter(ds_mouse, window_length=0.1, min_periods=2)

# %%
# We see that this time the number of NaN values has not increased after
# filtering. Instead, it has even decreased by a bit across keypoints.
# Let's visualise the effects of the median filter in the time and frequency
# domains. Here we focus on a 80 second time range for the ``snout`` keypoint.
# You can adjust the ``keypoint`` and ``time_range`` arguments to explore other
# parts of the data.

plot_raw_and_smooth_timeseries_and_psd(
    ds_mouse, ds_mouse_medfilt, keypoint="snout", time_range=slice(0, 80)
)

# %%
# The smoothing has reduced high-frequency content but the
# resulting time series stays quite close to the raw data.
#
# What happens if we increase the ``window_length`` to 2 seconds?

ds_mouse_medfilt = median_filter(ds_mouse, window_length=1, min_periods=2)

# %%
# The number of NaN values has decreased even further. That's because the
# chance of finding at least 2 valid values within a 2 second window is
# quite high. Let's plot the results for the same keypoint and time range
# as before.

plot_raw_and_smooth_timeseries_and_psd(
    ds_mouse, ds_mouse_medfilt, keypoint="snout", time_range=slice(0, 80)
)
# %%
# We see that the filtered time series is much smoother and it has even
# "bridged" over some small gaps. That said, it often deviates from the raw
# data, in ways that may not be desirable, depending on the application.
# That means that our choice of ``window_length`` may be too large.
# In general, you should choose a ``window_length`` that is small enough to
# preserve the original data structure, but large enough to remove
# "spikes" and high-frequency noise. Always inspect the results to ensure
# that the filter is not removing important features.

# %%
# Smoothing with a Savitzky-Golay filter
# --------------------------------------
# Here we use the :py:func:`movement.filtering.savgol_filter` function,
# which is a wrapper around :py:func:`scipy.signal.savgol_filter`.
# The Savitzky-Golay filter is a polynomial smoothing filter that can be
# applied to time series data on a rolling window basis. A polynomial of
# degree ``polyorder`` is fitted to the data in each window of length
# ``window_length``, and the value of the polynomial at the center of the
# window is used as the output value.
#
# Let's try it on the mouse dataset first.

ds_mouse_savgol = savgol_filter(ds_mouse, window_length=0.2, polyorder=2)


# %%
# We see that the number of NaN values has increased after filtering. This is
# for the same reason as with the median filter (in it's default mode), i.e.
# if there is at least one NaN value in the window, the output will be NaN.
# Unlike the median filter, the Savitzky-Golay filter does not provide a
# ``min_periods`` parameter to control this behaviour. Let's visualise the
# effects in the time and frequency domains.

plot_raw_and_smooth_timeseries_and_psd(
    ds_mouse, ds_mouse_savgol, keypoint="snout", time_range=slice(0, 80)
)
# %%
# We indeed see that high-frequencies have been reduced but the gaps of missing
# values have increase in extent. Now let's take a look at the wasp dataset.

ds_wasp_savgol = savgol_filter(ds_wasp, window_length=0.2, polyorder=2)

# %%
plot_raw_and_smooth_timeseries_and_psd(
    ds_wasp,
    ds_wasp_savgol,
    keypoint="stinger",
)
# %%
# This example shows two important limitations of the Savitzky-Golay filter.
# Firstly, it can introduce "wiggles" around sharp boundaries. For
# example, focus on what happens around the sudden drop in position
# during the final second. Secondly, the PSD appears to have large periodic
# drops in power at certain frequencies. Both of these effects vary with the
# choice of ``window_length`` and ``polyorder``. You can read more about these
# and other limitations of the Savitzky-Golay filter in
# `this paper <https://pubs.acs.org/doi/10.1021/acsmeasuresciau.1c00054>`_.


# %%
# Combining multiple smoothing filters
# ------------------------------------
# You can also combine multiple smoothing filters by applying them
# sequentially. For example, we can first apply the median filter with a small
# ``window_length`` to remove spikes and then apply the Savitzky-Golay filter
# with a larger ``window_length`` to further smooth the data.
# Between the two filters, we can interpolate over small gaps to avoid the
# excessive proliferation of NaN values. Let's try this on the mouse dataset.
# First, let's apply the median filter.

ds_mouse_medfilt = median_filter(ds_mouse, window_length=0.1, min_periods=2)

# %%
# Next, let's linearly interpolate over gaps smaller than 1 second.

ds_mouse_medfilt_interp = interpolate_over_time(ds_mouse_medfilt, max_gap=1)

# %%
# Finally, let's apply the Savitzky-Golay filter.

ds_mouse_medfilt_interp_savgol = savgol_filter(
    ds_mouse_medfilt_interp, window_length=0.4, polyorder=2
)

# %%
# A record of all applied operations is stored in the dataset's ``log``
# attribute. Let's inpsect it to summarise what we've done.

for entry in ds_mouse_medfilt_interp_savgol.log:
    print(entry)

# %%
# Now let's visualise the difference between the raw data and the final
# smoothed result.

plot_raw_and_smooth_timeseries_and_psd(
    ds_mouse,
    ds_mouse_medfilt_interp_savgol,
    keypoint="snout",
    time_range=slice(0, 80),
)

# %%
# Feel free to play around with the parameters of the applied filters and to
# also look at other keypoints and time ranges.
