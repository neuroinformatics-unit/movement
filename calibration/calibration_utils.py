"""Utility functions for confidence score calibration.

This module provides functions for different calibration methods, including:
- Log-Log Regression (Keypoint-Moseq style)
- Binning-based Calibration
- Temperature Scaling
- Computing Expected Calibration Error (ECE)
- Plotting Reliability Diagrams
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic
from sklearn.linear_model import LinearRegression


def log_log_regression_calibration(confidences, labels):
    """Apply log-log regression calibration (Keypoint-Moseq style).

    Parameters
    ----------
    confidences : np.ndarray
        Array of predicted confidence scores.
    labels : np.ndarray
        Array of true labels (approximated accuracy).

    Returns
    -------
    np.ndarray
        Adjusted confidence scores after log-log regression calibration.

    """
    eps = 1e-6
    valid_indices = ~np.isnan(labels)
    confidences = np.clip(confidences[valid_indices], eps, 1)
    labels = labels[valid_indices]

    log_confidences = np.log(confidences)
    log_labels = np.log(np.clip(labels, eps, 1))

    regressor = LinearRegression()
    log_confidences = log_confidences.reshape(-1, 1)
    regressor.fit(log_confidences, log_labels)

    a, b = regressor.coef_[0], regressor.intercept_
    print(f"Log-Log Regression Parameters: a = {a:.4f}, b = {b:.4f}")

    log_calibrated_confidences = a * log_confidences + b
    calibrated_confidences = np.exp(log_calibrated_confidences).flatten()
    return np.clip(calibrated_confidences, 0, 1)


def binning_based_calibration(confidences, labels, n_bins=10):
    """Perform binning-based calibration on confidence scores.

    This method divides the confidence scores into discrete bins and
    replaces each confidence value with the average accuracy of that bin.
    It helps adjust confidence scores to better reflect actual correctness.

    Parameters
    ----------
    confidences : np.ndarray
        Array of predicted confidence scores (values between 0 and 1).
    labels : np.ndarray
        Array of ground truth labels (1 for correct, 0 for incorrect).
    n_bins : int, optional
        Number of bins for calibration (default is 10).

    Returns
    -------
    calibrated_confidences : np.ndarray
        Adjusted confidence scores after binning-based calibration.
    bin_edges : np.ndarray
        Edges of the bins used for calibration.
    bin_accuracies : np.ndarray
        Empirical accuracy in each bin.

    Notes
    -----
    - Each bin represents a confidence range.
    - The mean accuracy of each bin replaces the confidences in that bin.
    - This method is effective for calibration when enough data is available.

    """
    bin_edges = np.linspace(0, 1, n_bins + 1)  # Define bin edges
    bin_indices = (
        np.digitize(confidences, bin_edges, right=True) - 1
    )  # Assign each score to a bin

    bin_accuracies = np.zeros(n_bins)
    calibrated_confidences = np.copy(confidences)

    # Compute bin-wise accuracy
    for i in range(n_bins):
        bin_mask = bin_indices == i  # Find indices of scores in this bin
        if np.sum(bin_mask) > 0:
            bin_accuracies[i] = np.mean(
                labels[bin_mask]
            )  # Compute accuracy in bin
            calibrated_confidences[bin_mask] = bin_accuracies[
                i
            ]  # Assign new confidence

    return calibrated_confidences, bin_edges, bin_accuracies


def compute_ece(confidences, calibrated_confidences, labels, bins=10):
    """Compute the Expected Calibration Error(ECE)
    before and after calibration.

    ECE measures how well a model's confidence scores align with
    actual accuracy.
    It compares the predicted confidence against the observed
    accuracy in predefined bins.

    Parameters
    ----------
    confidences : np.ndarray
        Array of original confidence scores before calibration.
    calibrated_confidences : np.ndarray
        Array of confidence scores after applying calibration.
    labels : np.ndarray
        Array of ground truth labels (1 for correct, 0 for incorrect).
    bins : int, optional
        Number of bins for discretizing confidence scores (default: 10).

    Returns
    -------
    ece_before : float
        Expected Calibration Error before calibration.
    ece_after : float
        Expected Calibration Error after calibration.

    Notes
    -----
    - A well-calibrated model should have a lower ECE.
    - The metric is computed based on bin-wise differences
    between confidence and accuracy.

    """  # noqa: D205
    bin_edges = np.linspace(0, 1, bins + 1)  # Define bin edges

    # Compute accuracy per bin
    bin_acc, _, _ = binned_statistic(
        confidences, labels, statistic="mean", bins=bin_edges
    )
    bin_conf, _, _ = binned_statistic(
        confidences, confidences, statistic="mean", bins=bin_edges
    )
    bin_counts, _, _ = binned_statistic(
        confidences, labels, statistic="count", bins=bin_edges
    )

    # Compute accuracy per bin for calibrated scores
    bin_acc_cal, _, _ = binned_statistic(
        calibrated_confidences, labels, statistic="mean", bins=bin_edges
    )
    bin_conf_cal, _, _ = binned_statistic(
        calibrated_confidences,
        calibrated_confidences,
        statistic="mean",
        bins=bin_edges,
    )

    # Remove NaNs (bins with no data)
    valid_bins = ~np.isnan(bin_acc)
    valid_bins_cal = ~np.isnan(bin_acc_cal)

    # Compute ECE
    ece_before = np.sum(
        (bin_counts[valid_bins] / len(confidences))
        * np.abs(bin_acc[valid_bins] - bin_conf[valid_bins])
    )
    ece_after = np.sum(
        (bin_counts[valid_bins_cal] / len(confidences))
        * np.abs(bin_acc_cal[valid_bins_cal] - bin_conf_cal[valid_bins_cal])
    )

    return ece_before, ece_after


def temperature_scaling(confidences, temperature=1.5):
    """Apply temperature scaling to confidence scores.

    Temperature scaling is a post-processing calibration technique that
    adjusts the confidence scores to improve calibration without affecting
    classification decisions.

    Parameters
    ----------
    confidences : np.ndarray
        Array of predicted confidence scores.
    temperature : float, optional
        Scaling factor to smooth confidence scores (default: 1.5).

    Returns
    -------
    scaled_confidences : np.ndarray
        Adjusted confidence scores after scaling.

    Notes
    -----
    - Higher temperature values make the confidence scores more uniform.
    - A temperature of 1.0 means no scaling is applied.

    """
    eps = 1e-6  # Prevent log(0) errors
    scaled_confidences = np.exp(
        np.log(np.clip(confidences, eps, 1)) / temperature
    )
    return np.clip(scaled_confidences, 0, 1)


def plot_reliability_diagram(
    confidences, calibrated_confidences, labels, bin_edges
):
    """Plot a reliability diagram comparing confidence
    scores before and after calibration.

    Reliability diagrams visualize how well a model's confidence scores
    align with empirical accuracy. A well-calibrated model should have
    points close to the diagonal (perfect calibration line).

    Parameters
    ----------
    confidences : np.ndarray
        Original confidence scores before calibration.
    calibrated_confidences : np.ndarray
        Adjusted confidence scores after calibration.
    labels : np.ndarray
        True labels (1 for correct, 0 for incorrect).
    bin_edges : np.ndarray
        Array of bin edges used to group confidence scores.

    Returns
    -------
    None
        Displays the reliability diagram as a matplotlib plot.

    Notes
    -----
    - The blue line represents the confidence vs accuracy before calibration.
    - The red line represents the confidence vs accuracy after calibration.
    - The gray dashed line represents a perfectly calibrated model.

    """  # noqa: D205
    bins = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins
    bin_indices = np.digitize(confidences, bin_edges, right=True) - 1
    bin_accuracies = np.zeros(len(bins))

    for i in range(len(bins)):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) > 0:
            bin_accuracies[i] = np.mean(labels[bin_mask])

    plt.figure(figsize=(6, 5))
    plt.plot(
        bins, bin_accuracies, "o-", label="Before Calibration", color="blue"
    )
    plt.plot(bins, bins, "--", color="gray", label="Perfect Calibration")

    bin_indices_cal = (
        np.digitize(calibrated_confidences, bin_edges, right=True) - 1
    )
    bin_accuracies_cal = np.zeros(len(bins))

    for i in range(len(bins)):
        bin_mask = bin_indices_cal == i
        if np.sum(bin_mask) > 0:
            bin_accuracies_cal[i] = np.mean(labels[bin_mask])

    plt.plot(
        bins, bin_accuracies_cal, "s-", label="After Calibration", color="red"
    )

    plt.xlabel("Confidence Score Bins")
    plt.ylabel("Empirical Accuracy")
    plt.title("Reliability Diagram: Before vs. After Calibration")
    plt.legend()
    plt.show()
