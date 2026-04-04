#!/usr/bin/env python3
"""Static figure showing polarization dynamics over time.

Row 1: Overlay frames (1 second apart) - continuous mode
  - Each panel after the first shows two frames superimposed:
    * Current frame (panel_frame[p]) at 100% opacity
    * Previous frame (panel_frame[p] - fps) at 50% opacity (additive ghost)
  - Displacement arrows: velocity_node(t−fps) → velocity_node(t)
  - For sparse mode: no overlay, just current frame

Row 2: Current frame with body axis
  - Shows current frame only (no overlay)
  - Opaque body axis line
  - Square marker at from_node, circle marker at to_node
  - When pair is loaded from AP validation H5: from_node = inferred posterior,
    to_node = inferred anterior (via _order_pair_by_ap). Otherwise
    the user's ordering is taken as-is with no directional validation.
  - Use H5 output from compute_polarization_AP_inference.py to automatically
    select the suggested ordered pair from the AP inference pipeline.
    Element 0 of the suggested pair (inferred posterior node)
    serves as velocity keypoint.

Row 3: Polar plots
  - Orientation polarization (yellow) and heading polarization (orange)

Usage:
    python compute_polarization_viz.py
"""

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast
from urllib.request import urlretrieve

import cv2
import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from movement.io import load_poses
from movement.kinematics import compute_polarization
from movement.kinematics.body_axis import ValidateAPConfig, validate_ap

# ── Duplicated-literal constants ─────────────────────────────────────────────
SLP_GLOB = "*.slp"
INVALID_NUMBER_MSG = "  Invalid. Enter a number."
ENTER_INDEX_PROMPT = "  Enter index: "


class TeeOutput:
    """Context manager that duplicates stdout to both console and a file."""

    def __init__(self, filepath):
        """Initialise with target file path."""
        self.filepath = Path(filepath)
        self.file = None
        self.original_stdout = None

    def __enter__(self):
        """Open log file and redirect stdout."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.filepath, "w")
        self.original_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore stdout and close the log file."""
        sys.stdout = self.original_stdout
        if self.file:
            self.file.close()
        return False

    def write(self, text):
        """Write *text* to both console and log file."""
        self.original_stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        """Flush both console and log file streams."""
        self.original_stdout.flush()
        self.file.flush()


# ── Pyramid blending helpers (extracted from adaptive_blend_frames) ──────────


def _build_pyramid(img, levels=4):
    """Build Gaussian and Laplacian pyramids."""
    gaussian = [img.astype(np.float32)]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        gaussian.append(img.astype(np.float32))
    laplacian = []
    for i in range(levels):
        size = (gaussian[i].shape[1], gaussian[i].shape[0])
        expanded = cv2.pyrUp(gaussian[i + 1], dstsize=size)
        laplacian.append(gaussian[i] - expanded)
    laplacian.append(gaussian[-1])
    return laplacian


def _blend_pyramids(lap1, lap2, mask_pyr):
    """Blend two Laplacian pyramids using mask pyramid."""
    blended = []
    for l1, l2, m in zip(lap1, lap2, mask_pyr, strict=True):
        if m.ndim == 2 and l1.ndim == 3:
            m = m[:, :, np.newaxis]
        blended.append(l1 * (1 - m) + l2 * m)
    return blended


def _reconstruct(pyramid):
    """Reconstruct image from Laplacian pyramid."""
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        size = (pyramid[i].shape[1], pyramid[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size) + pyramid[i]
    return img


def _to_grayscale(img_u8):
    """Convert to grayscale, handling both color and single-channel."""
    if img_u8.ndim == 3:
        return cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    return img_u8


def _build_position_masks(h, w, prev_positions, curr_positions, mask_radius):
    """Create ghost and protection masks from tracked animal positions."""
    ghost_mask = np.zeros((h, w), dtype=np.float32)
    protect_mask = np.zeros((h, w), dtype=np.float32)

    if prev_positions is not None:
        for px, py in prev_positions:
            if not np.isnan(px) and not np.isnan(py):
                cv2.circle(
                    ghost_mask, (int(px), int(py)), mask_radius, 1.0, -1
                )

    if curr_positions is not None:
        for cx, cy in curr_positions:
            if not np.isnan(cx) and not np.isnan(cy):
                cv2.circle(
                    protect_mask, (int(cx), int(cy)), mask_radius, 1.0, -1
                )

    ghost_mask = cast(
        "np.ndarray",
        cv2.GaussianBlur(ghost_mask, (51, 51), 0),
    )
    protect_mask = cast(
        "np.ndarray",
        cv2.GaussianBlur(protect_mask, (51, 51), 0),
    )
    return ghost_mask, protect_mask


def _enhance_ghost(prev_u8, clahe):
    """Apply CLAHE enhancement and tint to ghost frame."""
    if prev_u8.ndim == 3:
        prev_lab = cv2.cvtColor(prev_u8, cv2.COLOR_RGB2LAB)
        prev_lab[:, :, 0] = clahe.apply(prev_lab[:, :, 0])
        prev_enhanced = cv2.cvtColor(prev_lab, cv2.COLOR_LAB2RGB)
    else:
        prev_enhanced = clahe.apply(prev_u8)

    ghost_tinted = prev_enhanced.astype(np.float32)
    if ghost_tinted.ndim == 3:
        ghost_tinted[:, :, 0] *= 0.85  # reduce red
        ghost_tinted[:, :, 1] *= 0.95  # slight reduce green
        ghost_tinted[:, :, 2] = np.clip(ghost_tinted[:, :, 2] * 1.1, 0, 255)
    return ghost_tinted


def _compute_alpha_map(
    base_alpha,
    ghost_mask,
    protect_mask,
    motion_alpha,
    prev_positions,
    curr_positions,
    curr_gray,
):
    """Compute adaptive alpha map from masks and motion data."""
    if prev_positions is not None or curr_positions is not None:
        alpha_map = (
            base_alpha * (0.3 + 0.7 * ghost_mask) * (1 - 0.5 * protect_mask)
        )
        alpha_map = np.maximum(alpha_map, base_alpha * 0.5 * motion_alpha)
    else:
        alpha_map = base_alpha * (0.3 + 0.7 * motion_alpha)

    brightness = curr_gray.astype(np.float32) / 255.0
    alpha_map = alpha_map * (1 - 0.5 * brightness)
    return cv2.GaussianBlur(alpha_map, (15, 15), 0)


def adaptive_blend_frames(
    curr_img: np.ndarray,
    prev_img: np.ndarray,
    base_alpha: float = 0.5,
    curr_positions: list[tuple[float, float]] | None = None,
    prev_positions: list[tuple[float, float]] | None = None,
    mask_radius: int = 80,
) -> np.ndarray:
    """Blend two frames using CV2-based adaptive compositing.

    Uses multiple techniques for optimal visibility of both frames:
    1. Motion-based ghost detection via frame differencing
    2. Position-aware masking if tracking coordinates provided
    3. Laplacian pyramid blending for seamless compositing
    4. Local contrast enhancement (CLAHE) for ghost visibility
    5. Adaptive tone mapping to prevent bleaching

    Parameters
    ----------
    curr_img : ndarray
        Current frame as float array in [0, 1], shape (H, W, C) or (H, W).
    prev_img : ndarray
        Previous frame (ghost) as float array in [0, 1], same shape.
    base_alpha : float
        Base ghost opacity (0-1). Default 0.5.
    curr_positions : list of (x, y) tuples, optional
        Current positions of tracked animals (arrow heads).
    prev_positions : list of (x, y) tuples, optional
        Previous positions of tracked animals (arrow stems/ghosts).
    mask_radius : int
        Radius around animal positions for masking. Default 80 pixels.

    Returns
    -------
    ndarray
        Blended image with same shape as inputs, values in [0, 1].

    """
    curr_u8 = (curr_img * 255).astype(np.uint8)
    prev_u8 = (prev_img * 255).astype(np.uint8)
    h, w = curr_u8.shape[:2]

    # Step 1: Detect motion via frame difference
    curr_gray = _to_grayscale(curr_u8)
    prev_gray = _to_grayscale(prev_u8)
    diff = cv2.absdiff(curr_gray, prev_gray)
    _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    kernel: np.ndarray = np.ones((3, 3), dtype=np.uint8)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=3)
    motion_mask = cv2.GaussianBlur(motion_mask, (21, 21), 0)

    # Step 2: Position-aware masks
    ghost_mask, protect_mask = _build_position_masks(
        h,
        w,
        prev_positions,
        curr_positions,
        mask_radius,
    )

    # Step 3: Enhance ghost with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ghost_tinted = _enhance_ghost(prev_u8, clahe)

    # Step 4: Compute adaptive alpha
    motion_alpha = motion_mask.astype(np.float32) / 255.0
    alpha_map = _compute_alpha_map(
        base_alpha,
        ghost_mask,
        protect_mask,
        motion_alpha,
        prev_positions,
        curr_positions,
        curr_gray,
    )

    # Step 5: Laplacian pyramid blending
    levels = 4
    curr_lap = _build_pyramid(curr_u8, levels)
    ghost_lap = _build_pyramid(ghost_tinted.astype(np.uint8), levels)

    mask_pyr = [alpha_map]
    for _ in range(levels):
        mask_pyr.append(cv2.pyrDown(mask_pyr[-1]))

    blended_lap = _blend_pyramids(curr_lap, ghost_lap, mask_pyr)
    result = _reconstruct(blended_lap)

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result.astype(np.float32) / 255.0


@dataclass
class VideoMetadata:
    """Metadata extracted from video file."""

    fps: float | None = None
    frame_count: int | None = None
    width: int | None = None
    height: int | None = None
    duration: float | None = None
    path: Path | None = None

    @property
    def resolution(self) -> tuple[int, int] | None:
        """Return (width, height) tuple if available."""
        if self.width is not None and self.height is not None:
            return (self.width, self.height)
        return None


# Disable navigation toolbar
matplotlib.rcParams["toolbar"] = "None"

# CONFIGURATION
ROOT_PATH = Path(__file__).parent / "datasets" / "multi-animal"
SLP_DIR = ROOT_PATH / "slp"
MP4_DIR = ROOT_PATH / "mp4"

DEMO_DATASETS = {
    "free-moving-2flies-ID-13nodes-1024x1024x1-30_3pxmm": {
        "mp4": "https://storage.googleapis.com/sleap-data/datasets/wt_gold.13pt/clips/talk_title_slide%4013150-14500.mp4",  # noqa: E501
        "slp": "https://storage.googleapis.com/sleap-data/datasets/wt_gold.13pt/clips/talk_title_slide%4013150-14500.slp",  # noqa: E501
    },
    "free-moving-5mice-noID-11nodes-1280x1024x1-1_97pxmm": {
        "mp4": "https://storage.googleapis.com/sleap-data/datasets/wang_4mice_john/clips/OFTsocial5mice-0000-00%4015488-18736.mp4",  # noqa: E501
        "slp": "https://storage.googleapis.com/sleap-data/datasets/wang_4mice_john/clips/OFTsocial5mice-0000-00%4015488-18736.slp",  # noqa: E501
    },
    "free-moving-2mice-noID-5nodes-1280x1024x1-1_9pxmm": {
        "mp4": "https://storage.googleapis.com/sleap-data/datasets/eleni_mice/clips/20200111_USVpairs_court1_M1_F1_top-01112020145828-0000%400-2560.mp4",  # noqa: E501
        "slp": "https://storage.googleapis.com/sleap-data/datasets/eleni_mice/clips/20200111_USVpairs_court1_M1_F1_top-01112020145828-0000%400-2560.slp",  # noqa: E501
    },
    "freemoving-2bees-noID-21nodes-1535x2048x1-14pxmm": {
        "mp4": "https://storage.googleapis.com/sleap-data/datasets/yan_bees/clips/bees_demo%4021000-23000.mp4",  # noqa: E501
        "slp": "https://storage.googleapis.com/sleap-data/datasets/yan_bees/clips/bees_demo%4021000-23000.slp",  # noqa: E501
    },
    "free-moving-4gerbils-ID-14nodes-1024x1280x3-2pxmm": {
        "mp4": "https://storage.googleapis.com/sleap-data/datasets/nyu-gerbils/clips/2020-3-10_daytime_5mins_compressedTalmo%403200-5760.mp4",  # noqa: E501
        "slp": "https://storage.googleapis.com/sleap-data/datasets/nyu-gerbils/clips/2020-3-10_daytime_5mins_compressedTalmo%403200-5760.slp",  # noqa: E501
    },
}

# Frame interval is computed dynamically as fps (1 second between panels)
NUM_PANELS = 5  # Number of time points to display (0s to 4s)
MIN_CONTINUOUS_PANELS = 3  # Minimum panels for partial continuous mode

OVERLAY_ALPHA = 0.7  # Ghost frame opacity (Row 1 overlay)

ANIMAL_COLORS = [
    "red",
    "#39FF14",
    "#1E90FF",
    "magenta",
    "cyan",
    "yellow",
    "white",
]
NET_ORIENTATION_COLOR = [
    1.00,
    0.85,
    0.20,
]  # Orientation polarization vector (yellow)
NET_HEADING_COLOR = [1.00, 0.55, 0.10]  # Heading polarization vector (orange)
DARK_BG = (0.08, 0.08, 0.10)  # Figure background
DARK_FG = (0.92, 0.92, 0.92)  # Text and axis color

SAVE_PNG = False
SAVE_SVG = True

# When True, prompt user interactively; when False, use auto-detected values
INTERACTIVE_NODE_SELECTION = True

# Minimum polarization for visually meaningful vectors
MIN_VISUAL_POL = 0.15

# When not None, skip interactive mode and process each dataset
# with specified nodes.
# Format: {slp_file_index: (from_node_idx, to_node_idx)}
# - from_node: should be posterior keypoint (e.g., tail_base, abdomen)
# - to_node: should be anterior keypoint (e.g., nose, head)
# NOTE: manual BATCH_CONFIG has no automatic AP-inference ordering unless
# BATCH_CONFIG_AP_VALIDATED is set to True.
# Example: {0: (3, 0), 1: (4, 0)}
BATCH_CONFIG: dict[int, tuple[int, int]] | None = None

# When True, treat BATCH_CONFIG pairs as already AP-validated
# (posterior→anterior ordering), skipping the AP inference feedback.
BATCH_CONFIG_AP_VALIDATED = False

# When True, auto-load suggested pairs from AP validation H5 file.
# This populates BATCH_CONFIG automatically: element 0 = inferred posterior,
# element 1 = inferred anterior
# (ordered by _order_pair_by_ap in collective.py).
USE_AP_VALIDATION_H5 = True


# ── Shared helpers ───────────────────────────────────────────────────────────


def _list_slp_files(directory: Path) -> list[Path]:
    """Return sorted SLP files from *directory*, or empty list if missing."""
    return sorted(directory.glob(SLP_GLOB)) if directory.exists() else []


def _prompt_int_in_range(prompt: str, low: int, high: int) -> int:
    """Repeatedly prompt until the user enters an integer in [low, high)."""
    while True:
        try:
            value = int(input(prompt).strip())
            if low <= value < high:
                return value
            print(f"  Invalid. Enter {low}-{high - 1}.")
        except ValueError:
            print(INVALID_NUMBER_MSG)


def _download_file(url: str, destination: Path) -> None:
    """Download a single file to the requested destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {destination.name}...")
    urlretrieve(url, destination)


def ensure_demo_datasets() -> None:
    """Ensure the expected demo SLP/MP4 files exist locally."""
    ROOT_PATH.mkdir(parents=True, exist_ok=True)
    SLP_DIR.mkdir(parents=True, exist_ok=True)
    MP4_DIR.mkdir(parents=True, exist_ok=True)

    slp_files = sorted(SLP_DIR.glob(SLP_GLOB))
    mp4_files = sorted(MP4_DIR.glob("*.mp4"))

    should_bootstrap = len(slp_files) < len(DEMO_DATASETS) or len(
        mp4_files
    ) < len(DEMO_DATASETS)
    if not should_bootstrap:
        return

    print(
        "Bootstrapping demo datasets "
        f"(found {len(slp_files)} .slp and {len(mp4_files)} .mp4 files)..."
    )

    for file_stem, urls in DEMO_DATASETS.items():
        slp_target = SLP_DIR / f"{file_stem}.slp"
        mp4_target = MP4_DIR / f"{file_stem}.mp4"

        if not slp_target.exists():
            _download_file(urls["slp"], slp_target)
        if not mp4_target.exists():
            _download_file(urls["mp4"], mp4_target)


# ── H5 loading helpers ──────────────────────────────────────────────────────


def _decode_h5_strings(arr):
    """Decode bytes to str if necessary for H5 string arrays."""
    return [s.decode() if isinstance(s, bytes) else s for s in arr]


def _find_best_per_file(files, rxm, suggested_from, suggested_to):
    """Find the best individual per file based on highest R×M score."""
    best_per_file: dict[str, tuple[float, int, int]] = {}
    for i in range(len(files)):
        file_stem = files[i]
        if not file_stem:
            continue
        current_rxm = rxm[i]
        if np.isnan(current_rxm):
            continue
        is_new = file_stem not in best_per_file
        is_better = current_rxm > best_per_file.get(file_stem, (0,))[0]
        if is_new or is_better:
            best_per_file[file_stem] = (
                current_rxm,
                int(suggested_from[i]),
                int(suggested_to[i]),
            )
    return best_per_file


def load_suggested_pairs_from_h5() -> dict[int, tuple[int, int]] | None:
    """Load suggested AP node pairs from the AP validation H5 file.

    Finds the most recent ap_validation_*.h5 file in the exports directory
    and extracts the suggested (from_node, to_node) pair for each file's
    best individual (highest R×M score).
    Element 0 = inferred posterior (lower AP
    coord), element 1 = inferred anterior (higher AP coord).

    Returns
    -------
    dict[int, tuple[int, int]] or None
        Mapping of {slp_file_index: (posterior_node_idx, anterior_node_idx)}.
        Returns None if no H5 file found or extraction fails.

    """
    h5_dir = ROOT_PATH / "exports" / "AP-inference-demo" / "h5"
    if not h5_dir.exists():
        print(f"AP validation H5 directory not found: {h5_dir}")
        return None

    h5_files = sorted(h5_dir.glob("ap_validation_*.h5"))
    if not h5_files:
        print(f"No AP validation H5 files found in {h5_dir}")
        return None

    h5_path = h5_files[-1]
    print(f"Loading suggested pairs from: {h5_path.name}")

    slp_path = ROOT_PATH / "slp"
    slp_files = _list_slp_files(slp_path)
    if not slp_files:
        print("No SLP files found")
        return None

    stem_to_idx = {sf.stem: idx for idx, sf in enumerate(slp_files)}

    try:
        with h5py.File(h5_path, "r") as f:
            files = _decode_h5_strings(f["file"][:])
            _individuals = _decode_h5_strings(f["individual"][:])
            rxm = f["rxm"][:]
            suggested_from = f["suggested_from_idx"][:]
            suggested_to = f["suggested_to_idx"][:]

            best_per_file = _find_best_per_file(
                files,
                rxm,
                suggested_from,
                suggested_to,
            )

            batch_config = {}
            for file_stem, (_, from_idx, to_idx) in best_per_file.items():
                if file_stem not in stem_to_idx:
                    continue
                slp_idx = stem_to_idx[file_stem]
                if from_idx >= 0 and to_idx >= 0:
                    batch_config[slp_idx] = (from_idx, to_idx)
                    print(
                        f"  [{slp_idx}] {file_stem}: "
                        f"[{from_idx} → {to_idx}] "
                        f"(posterior → anterior)"
                    )

            if batch_config:
                return batch_config
            print("No valid suggested pairs found in H5")
            return None

    except Exception as e:
        print(f"Error loading H5 file: {e}")
        return None


def _run_ap_inference(
    ds: xr.Dataset,
    from_keypoint: str,
    to_keypoint: str,
) -> tuple[dict, dict, float] | None:
    """Run AP inference on a pair and return the best individual's results.

    Parameters
    ----------
    ds : xr.Dataset
        Movement dataset with ``position`` DataArray.
    from_keypoint : str
        Name of the from_node.
    to_keypoint : str
        Name of the to_node.

    Returns
    -------
    tuple of (result_dict, pair_report, best_rxm) or None
        The best individual's result dict, its ``APNodePairReport``,
        and the R×M score.  Returns ``None`` if all individuals failed.

    """
    config = ValidateAPConfig()
    individuals = list(ds.coords["individuals"].values)

    all_results: list[dict] = []
    for individual in individuals:
        pos_data = ds.position.sel(individuals=individual)
        result = validate_ap(
            pos_data,
            from_node=from_keypoint,
            to_node=to_keypoint,
            config=config,
            verbose=False,
        )
        result["individual"] = individual
        all_results.append(result)

    best_idx = -1
    best_rxm = -1.0
    for i, result in enumerate(all_results):
        if not result["success"]:
            continue
        pr = result.get("pair_report")
        if pr is None or not pr.success:
            continue
        rxm = result["resultant_length"] * result["vote_margin"]
        if rxm > best_rxm:
            best_rxm = rxm
            best_idx = i

    if best_idx < 0:
        return None

    result = all_results[best_idx]
    pr = result["pair_report"]
    return result, pr, best_rxm


# ── AP scenario / ordering helpers ───────────────────────────────────────────


def _find_suggested_alternative(pr, input_set):
    """Return a suggested alternative pair from AP inference, or None."""
    if len(pr.max_separation_distal_nodes) == 2:
        alt = pr.max_separation_distal_nodes
        if frozenset([int(alt[0]), int(alt[1])]) != input_set:
            return (int(alt[0]), int(alt[1])), "max-separation distal"

    if len(pr.max_separation_nodes) == 2:
        alt = pr.max_separation_nodes
        if frozenset([int(alt[0]), int(alt[1])]) != input_set:
            return (int(alt[0]), int(alt[1])), "max-separation overall"

    return None


def _prompt_yn(prompt_text: str) -> bool:
    """Prompt for y/n response, returning True for 'y'."""
    while True:
        reply = input(prompt_text).strip().lower()
        if reply in ("y", "yes"):
            return True
        if reply in ("n", "no"):
            return False
        print("    Please enter 'y' or 'n'.")


def report_ap_scenario(
    ds: xr.Dataset,
    from_keypoint: str,
    to_keypoint: str,
    node_names: list[str],
    inference_result: tuple[dict, dict, float] | None = None,
) -> tuple[int, int] | None:
    """Report the AP scenario for the user's pair and suggest alternatives.

    Runs the AP inference pipeline (or reuses pre-computed results),
    reports which of the 13 mutually exclusive scenarios the user's
    pair falls into, and interactively offers to switch to a better
    pair if one exists.

    Parameters
    ----------
    ds : xr.Dataset
        Movement dataset with ``position`` DataArray.
    from_keypoint : str
        Name of the user-specified from_node.
    to_keypoint : str
        Name of the user-specified to_node.
    node_names : list[str]
        List of all keypoint names (for index↔name conversion).
    inference_result : tuple or None, optional
        Pre-computed output from ``_run_ap_inference``.  If ``None``,
        inference is run internally.

    Returns
    -------
    tuple[int, int] or None
        ``(posterior_idx, anterior_idx)`` if the user accepted a
        suggested alternative pair (already AP-ordered), or ``None``
        if no switch was made.

    """
    print("\n  AP scenario report (running inference on user's pair)...")

    if inference_result is None:
        inference_result = _run_ap_inference(ds, from_keypoint, to_keypoint)

    if inference_result is None:
        print("  AP inference failed for all individuals.")
        return None

    result, pr, best_rxm = inference_result
    ind_label = result.get("individual", "unknown")

    print(f"    Best individual: {ind_label} (R×M = {best_rxm:.3f})")
    print(f"    Scenario {pr.scenario}: {pr.outcome}")
    if pr.warning_message:
        print(f"    {pr.warning_message}")

    from_idx = node_names.index(from_keypoint)
    to_idx = node_names.index(to_keypoint)
    input_set = frozenset([from_idx, to_idx])

    alt_result = _find_suggested_alternative(pr, input_set)
    if alt_result is None:
        return None

    suggested, suggested_label = alt_result
    s_from, s_to = suggested
    print(
        f"    Suggested alternative ({suggested_label}): "
        f"{node_names[s_from]}[{s_from}] -> "
        f"{node_names[s_to]}[{s_to}] "
        f"(posterior -> anterior)"
    )
    if _prompt_yn("    Switch to suggested pair? [y/n]: "):
        print(
            f"    Switching to: "
            f"{node_names[s_from]}[{s_from}] -> "
            f"{node_names[s_to]}[{s_to}]"
        )
        return suggested
    print("    Keeping original pair.")
    return None


def check_ap_ordering(
    ds: xr.Dataset,
    from_keypoint: str,
    to_keypoint: str,
    inference_result: tuple[dict, dict, float] | None = None,
) -> bool:
    """Run AP inference and warn if ordering is reversed.

    Uses ``validate_ap`` on each individual (or reuses pre-computed
    results), selects the best individual (highest R×M score), and
    checks whether the user's from_node is actually posterior to their
    to_node according to the inferred AP axis.  If reversed,
    interactively prompts the user to consider flipping.

    Parameters
    ----------
    ds : xr.Dataset
        Movement dataset with ``position`` DataArray.
    from_keypoint : str
        Name of the user-specified from_node.
    to_keypoint : str
        Name of the user-specified to_node.
    inference_result : tuple or None, optional
        Pre-computed output from ``_run_ap_inference``.  If ``None``,
        inference is run internally.

    Returns
    -------
    bool
        True if the user accepted the flip (from/to should be swapped),
        False otherwise (ordering assumed correct, user declined flip,
        or inference failed).

    """
    print("\n  AP ordering check...")

    if inference_result is None:
        inference_result = _run_ap_inference(ds, from_keypoint, to_keypoint)

    if inference_result is None:
        print(
            "  AP inference could not validate this pair"
            " (all individuals failed)"
        )
        return False

    result, pr, best_rxm = inference_result
    sign_str = "+" if result["anterior_sign"] > 0 else "−"
    margin = result["vote_margin"]
    ind_label = result.get("individual", "unknown")

    if pr.input_pair_order_matches_inference:
        print(
            f"  AP inference agrees: {from_keypoint} is "
            f"posterior to {to_keypoint} "
            f"(anterior = {sign_str}PC1, "
            f"vote margin M = {margin:.3f}, "
            f"best individual = {ind_label}, "
            f"R×M = {best_rxm:.3f})"
        )
        return False

    print(
        f"  WARNING: AP inference reports {from_keypoint} is "
        f"ANTERIOR to {to_keypoint} - ordering likely reversed!"
    )
    print(
        f"    Inferred AP direction: {sign_str}PC1, "
        f"vote margin M = {margin:.3f}, "
        f"best individual = {ind_label}, "
        f"R×M = {best_rxm:.3f}"
    )
    print(
        f"    Appropriate order would likely be: "
        f"{to_keypoint} (posterior) -> "
        f"{from_keypoint} (anterior)"
    )

    if _prompt_yn("    Flip from_node and to_node? [y/n]: "):
        print(
            f"    Flipping: from={to_keypoint} (posterior), "
            f"to={from_keypoint} (anterior)"
        )
        return True
    print("    Keeping original ordering as entered.")
    return False


def prompt_dataset_selection():
    """List available datasets and prompt user to select one.

    Returns
    -------
    selected_index : int
        Selected dataset index (0-based).

    """
    slp_path = ROOT_PATH / "slp"
    slp_files = _list_slp_files(slp_path)

    if not slp_files:
        raise FileNotFoundError(f"No SLP files found in {slp_path}")

    print("\n")
    print("AVAILABLE DATASETS")
    print(f"\nFound {len(slp_files)} SLP file(s):\n")

    for idx, sf in enumerate(slp_files):
        print(f"  [{idx}] {sf.name}")

    print()

    sel_idx = _prompt_int_in_range(
        f"Select dataset [0-{len(slp_files) - 1}]: ",
        0,
        len(slp_files),
    )
    print(f"\nSelected: [{sel_idx}] {slp_files[sel_idx].name}")
    print("─" * 60)
    return sel_idx


# ── Video metadata helpers ───────────────────────────────────────────────────


def _validate_fps_from_duration(metadata, video_path, actual_duration_ms):
    """Cross-check stored FPS against actual video duration.

    Overwrites ``metadata.fps`` and ``metadata.duration`` in-place if the
    stored value is too far from the computed one.
    """
    if actual_duration_ms is None or actual_duration_ms <= 0:
        metadata.duration = metadata.frame_count / metadata.fps
        return

    actual_duration_sec = actual_duration_ms / 1000.0
    computed_fps = metadata.frame_count / actual_duration_sec
    fps_diff = abs(metadata.fps - computed_fps)

    if fps_diff <= 1.0:
        metadata.duration = metadata.frame_count / metadata.fps
        return

    import warnings

    warnings.warn(
        f"FPS mismatch in {video_path.name}:\n"
        f"  Stored FPS: {metadata.fps:.2f}\n"
        f"  Computed FPS: {computed_fps:.2f}\n"
        f"  Frame count: {metadata.frame_count}\n"
        f"  Actual duration: {actual_duration_sec:.2f}s\n"
        f"  Difference: {fps_diff:.2f} fps\n"
        f"  Using COMPUTED FPS value.",
        UserWarning,
        stacklevel=3,
    )
    metadata.fps = computed_fps
    metadata.duration = actual_duration_sec


def extract_video_metadata(video_path: Path) -> VideoMetadata:
    """Extract metadata from video file.

    Parameters
    ----------
    video_path : Path
        Path to the video file

    Returns
    -------
    VideoMetadata
        Dataclass containing fps, frame_count, width, height, duration, path.
        Fields are None if extraction fails.

    """
    metadata = VideoMetadata(path=video_path)
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return metadata

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            metadata.fps = fps

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            metadata.frame_count = frame_count

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width > 0 and height > 0:
            metadata.width = width
            metadata.height = height

        actual_duration_ms = None
        if frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            actual_duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        if metadata.fps and metadata.frame_count:
            _validate_fps_from_duration(
                metadata, video_path, actual_duration_ms
            )

        cap.release()
    except Exception:
        pass
    return metadata


# ── Data loading helpers ─────────────────────────────────────────────────────


def _resolve_fps(fps, video_metadata):
    """Determine FPS from user input or video metadata.

    Returns (fps, source) tuple.
    """
    if fps is not None:
        return int(fps), "user-defined"
    if video_metadata is not None and video_metadata.fps is not None:
        return int(video_metadata.fps), "video metadata"
    return None, None


def _print_video_info(video_metadata, fps, fps_source):
    """Print video and FPS information."""
    if fps is not None:
        print(f"FPS: {fps} (from {fps_source})")
    else:
        print("FPS: Unknown (will use sparse frame selection)")

    if video_metadata is None:
        return
    if video_metadata.frame_count is not None:
        print(f"Video frames: {video_metadata.frame_count}")
    if video_metadata.resolution is not None:
        print(
            f"Video resolution: {video_metadata.width}x{video_metadata.height}"
        )
    if video_metadata.duration is not None:
        print(f"Video duration: {video_metadata.duration:.2f}s")


def load_data(dataset_idx, fps=None, node_config=None):
    """Load tracking data, returning poses as xarray Dataset.

    Video loading is deferred to plotting stage.

    Parameters
    ----------
    dataset_idx : int
        Index of the dataset to load (0-based).
    fps : int, optional
        Frames per second. If provided, overrides auto-detection.
        If None, attempts: video metadata → None (sparse mode)
    node_config : dict, optional
        Pre-defined node configuration.

    Returns
    -------
    ds : xarray.Dataset
        Movement-compatible dataset.
    video_file : Path or None
        Path to the video file if it exists, None otherwise
    video_metadata : VideoMetadata or None
        Video metadata if video exists.
    slp_filename : str
        Name of the SLP file (used for output naming)
    node_config : dict
        Node configuration for orientation and velocity computation

    """
    slp_files = _list_slp_files(SLP_DIR)
    video_files = sorted(MP4_DIR.glob("*.mp4")) if MP4_DIR.exists() else []

    slp_file = slp_files[dataset_idx]

    # Find matching video file
    video_file = None
    for vf in video_files:
        if vf.stem == slp_file.stem or slp_file.stem in vf.stem:
            video_file = vf
            break

    video_metadata = None
    if video_file is not None:
        video_metadata = extract_video_metadata(video_file)

    fps, fps_source = _resolve_fps(fps, video_metadata)

    print(f"Loading SLP: {slp_file.name}")
    if video_file:
        print(f"Video found: {video_file.name}")
    else:
        print("Video: Not found (will use sparse mode without frames)")

    ds = load_poses.from_sleap_file(slp_file, fps=fps)

    node_names = list(ds.keypoints.values)
    track_names = list(ds.individuals.values)
    total_frames = ds.sizes["time"]
    n_keypoints = ds.sizes["keypoints"]
    total_animals = ds.sizes["individuals"]

    _print_video_info(video_metadata, fps, fps_source)

    print(f"Tracking frames: {total_frames}")
    print(f"Individuals ({total_animals}): {track_names}")
    print(f"Keypoints ({n_keypoints}): {node_names}")

    if node_config is not None:
        _print_batch_node_config(node_config, node_names)
    else:
        node_config = prompt_node_selection(ds, frames=None)

    return ds, video_file, video_metadata, slp_file.name, node_config


def _print_batch_node_config(node_config, node_names):
    """Print the batch-mode node configuration summary."""
    from_idx = node_config["orientation_from"]
    to_idx = node_config["orientation_to"]
    vel_idx = node_config["velocity_node"]
    ap_ok = node_config.get("ap_validated", False)
    dir_label = "(posterior → anterior)" if ap_ok else "(from → to)"
    print("\nUsing batch config:")
    print(f"  Velocity:    {node_names[vel_idx]}[{vel_idx}]")
    print(
        f"  Orientation: {node_names[from_idx]}[{from_idx}] → "
        f"{node_names[to_idx]}[{to_idx}] "
        f"{dir_label}"
    )


def find_first_valid_frame(ds: xr.Dataset, min_animals: int = 2):
    """Find the first frame where min_animals have all keypoints visible.

    Parameters
    ----------
    ds : xarray.Dataset
        Movement dataset with position DataArray.
    min_animals : int
        Minimum number of individuals with all keypoints visible

    Returns
    -------
    tuple (frame_idx, list of individual_indices) or (None, None) if not found

    """
    position = ds.position.values
    n_frames = ds.sizes["time"]
    n_animals = ds.sizes["individuals"]

    for f in range(n_frames):
        valid_animals = []
        for a in range(n_animals):
            animal_data = position[f, :, :, a]
            if not np.any(np.isnan(animal_data)):
                valid_animals.append(a)
        if len(valid_animals) >= min_animals:
            return f, valid_animals
    return None, None


def show_keypoint_reference(
    frames: np.ndarray, ds: xr.Dataset, frame_idx: int, animal_indices: list
):
    """Display a reference frame with labeled keypoints.

    Parameters
    ----------
    frames : ndarray
        Video frames
    ds : xarray.Dataset
        Movement dataset with position DataArray.
    frame_idx : int
        Frame index to display.
    animal_indices : list
        List of individual indices to show.

    """
    node_names = list(ds.keypoints.values)
    position = ds.position.values

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.imshow(frames[frame_idx])
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(node_names)))

    for animal_idx in animal_indices:
        keypoints_xy = position[frame_idx, :, :, animal_idx]

        for i, _name in enumerate(node_names):
            x, y = keypoints_xy[0, i], keypoints_xy[1, i]
            if not np.isnan(x) and not np.isnan(y):
                ax.plot(x, y, "o", color=colors[i], markersize=5)
                ax.annotate(
                    f"{i}",
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_xlim(0, frames[frame_idx].shape[1])
    ax.set_ylim(frames[frame_idx].shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Keypoint Reference (Frame {frame_idx}, Animals {animal_indices})",
        color=DARK_FG,
        fontsize=12,
        pad=10,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    return fig


def prompt_node_selection(ds: xr.Dataset, frames: np.ndarray | None = None):
    """Prompt user to select keypoints for velocity and orientation.

    Parameters
    ----------
    ds : xarray.Dataset
        Movement dataset with position DataArray.
    frames : ndarray, optional
        Video frames (for showing reference image).

    Returns
    -------
    dict with keys:
        velocity_node: int - keypoint index for velocity/heading.
        orientation_from: int - "from" keypoint (ideally posterior).
        orientation_to: int - "to" keypoint (ideally anterior).
        ap_validated: bool - always False for interactive selection.

    """
    node_names = list(ds.keypoints.values)
    n_nodes = len(node_names)
    ref_fig = None

    node_list_str = "  ".join(
        f"{name}[{i}]" for i, name in enumerate(node_names)
    )

    if not INTERACTIVE_NODE_SELECTION:
        raise ValueError(
            "Interactive mode is required."
            " Set INTERACTIVE_NODE_SELECTION=True."
        )

    print("\n")
    print("NODE SELECTION")
    print(f"\nAvailable nodes: {node_list_str}")

    if frames is not None:
        frame_idx, animal_indices = find_first_valid_frame(ds, min_animals=2)
        if frame_idx is not None:
            msg = f"Showing keypoint reference (Frame {frame_idx})"
            print(f"\n{msg}, Animals {animal_indices}...")
            ref_fig = show_keypoint_reference(
                frames, ds, frame_idx, animal_indices
            )

    print("\nFROM node (should be posterior, e.g., tail_base):")
    orientation_from = _prompt_int_in_range(ENTER_INDEX_PROMPT, 0, n_nodes)

    print("\nTO node (should be anterior, e.g., head):")
    while True:
        orientation_to = _prompt_int_in_range(ENTER_INDEX_PROMPT, 0, n_nodes)
        if orientation_to != orientation_from:
            break
        print("  Must differ from FROM node.")

    print("\nVelocity node (for heading):")
    velocity_node = _prompt_int_in_range(ENTER_INDEX_PROMPT, 0, n_nodes)

    if ref_fig is not None:
        plt.close(ref_fig)

    print("\n" + "-" * 60)
    print("Final configuration:")
    print(f"  Velocity:    {node_names[velocity_node]}[{velocity_node}]")
    from_name = node_names[orientation_from]
    to_name = node_names[orientation_to]
    print(
        f"  Orientation: {from_name}[{orientation_from}] → "
        f"{to_name}[{orientation_to}] "
        f"(from → to)"
    )
    print("-" * 60 + "\n")

    return {
        "velocity_node": velocity_node,
        "orientation_from": orientation_from,
        "orientation_to": orientation_to,
        "ap_validated": False,
        "auto_log": None,
    }


# ── Sparse frame selection helpers ───────────────────────────────────────────


def _collect_candidate_frames(orientation_valid, vel_valid, total_frames):
    """Return frames where both orientation and velocity are valid."""
    return [
        f
        for f in range(1, total_frames)
        if orientation_valid[f] and vel_valid[f]
    ]


def _compute_combined_polarization(
    candidate_frames, orientation_pol, heading_pol, total_frames
):
    """Compute combined (orientation + heading) polarization per frame."""
    combined_pol = np.zeros(total_frames)
    for f in candidate_frames:
        vals = []
        if not np.isnan(orientation_pol[f]):
            vals.append(orientation_pol[f])
        if not np.isnan(heading_pol[f]):
            vals.append(heading_pol[f])
        if vals:
            combined_pol[f] = np.mean(vals)
    return combined_pol


def _greedy_select_frames(source_frames, combined_pol, selected, target_count):
    """Greedily add frames from *source_frames* to maximise variance.

    Mutates *selected* in-place and returns it.
    """
    remaining = set(source_frames)
    if not selected and remaining:
        first = max(remaining, key=lambda f: combined_pol[f])
        selected.append(first)
        remaining.remove(first)

    while len(selected) < target_count and remaining:
        current_vals = [combined_pol[f] for f in selected]
        best_f = max(
            remaining,
            key=lambda f: np.var(current_vals + [combined_pol[f]]),
        )
        selected.append(best_f)
        remaining.remove(best_f)

    return selected


def _find_sparse_frames(
    orientation_pol, heading_pol, orientation_valid, vel_valid, total_frames
):
    """Select sparse frames with maximal variance in polarization.

    Used when continuous segments aren't available or when fps is unknown.
    Each selected frame requires:
    - orientation_valid[f]: valid orientation at frame f
    - vel_valid[f]: valid velocity at frame f (implies pos_valid at f AND f-1)

    Returns
    -------
    Same as find_best_segment

    """
    candidate_frames = _collect_candidate_frames(
        orientation_valid,
        vel_valid,
        total_frames,
    )

    if len(candidate_frames) < NUM_PANELS:
        n_cand = len(candidate_frames)
        raise ValueError(
            f"Not enough valid frames ({n_cand}) for {NUM_PANELS} panels"
        )

    combined_pol = _compute_combined_polarization(
        candidate_frames,
        orientation_pol,
        heading_pol,
        total_frames,
    )

    visible_frames = [
        f for f in candidate_frames if combined_pol[f] >= MIN_VISUAL_POL
    ]
    low_vis_frames = [
        f for f in candidate_frames if combined_pol[f] < MIN_VISUAL_POL
    ]

    n_total = len(candidate_frames)
    n_vis = len(visible_frames)
    print(
        f"Candidate frames: {n_total} total, "
        f"{n_vis} visually clear (>= {MIN_VISUAL_POL})"
    )

    selected: list = []
    _greedy_select_frames(visible_frames, combined_pol, selected, NUM_PANELS)
    if len(selected) < NUM_PANELS:
        _greedy_select_frames(
            low_vis_frames, combined_pol, selected, NUM_PANELS
        )

    panel_frames = np.array(sorted(selected))
    prev_frames = panel_frames - 1

    has_orientation = np.ones(NUM_PANELS, dtype=bool)
    has_heading = np.ones(NUM_PANELS, dtype=bool)

    selected_pol = [combined_pol[f] for f in panel_frames]
    n_visible = sum(1 for p in selected_pol if p >= MIN_VISUAL_POL)

    print(f"Selected sparse frames: {panel_frames}")
    mvp = MIN_VISUAL_POL
    print(f"Previous frames (f-1): {prev_frames}")
    print(f"Polarization values: {[f'{p:.3f}' for p in selected_pol]}")
    print(f"Visually clear panels: {n_visible}/{NUM_PANELS} (>= {mvp})")
    print("Orientation valid: all True (guaranteed by selection)")
    print("Heading valid: all True (guaranteed by selection)")

    return (
        panel_frames,
        prev_frames,
        False,
        has_orientation,
        has_heading,
        1,
        NUM_PANELS,
    )


# ── Segment scoring helpers ──────────────────────────────────────────────────


def _safe_unwrap(theta_arr):
    """Unwrap angles, interpolating through NaN gaps."""
    t = theta_arr.copy()
    v = ~np.isnan(t)
    if np.sum(v) < 2:
        return t
    t[~v] = np.interp(np.where(~v)[0], np.where(v)[0], t[v])
    t = np.unwrap(t)
    t[~v] = np.nan
    return t


def _metric_scores(win):
    """Vectorised range × monotonicity + mean for each window."""
    wmin = np.nanmin(win, axis=1)
    wmax = np.nanmax(win, axis=1)
    rng = wmax - wmin
    delta = np.abs(win[:, -1] - win[:, 0])
    mono = np.where(rng > 0.01, delta / rng, 0.0)
    mean = np.nanmean(win, axis=1)
    return rng * (0.3 + 0.7 * mono) + 0.15 * mean


def _compute_sweep_bonus(ot_uw, ht_uw, ng, w):
    """Compute the collective-sweep scoring bonus."""
    orientation_sweep = np.abs(ot_uw[w - 1 : ng] - ot_uw[: ng - w + 1]) / np.pi
    head_sweep = np.abs(ht_uw[w - 1 : ng] - ht_uw[: ng - w + 1]) / np.pi
    orientation_signed = ot_uw[w - 1 : ng] - ot_uw[: ng - w + 1]
    head_signed = ht_uw[w - 1 : ng] - ht_uw[: ng - w + 1]

    mean_sw = np.nanmean(np.stack([orientation_sweep, head_sweep]), axis=0)
    agreement = np.where(
        orientation_signed * head_signed > 0,
        1.0,
        np.where(
            (np.abs(orientation_signed) < 0.1) | (np.abs(head_signed) < 0.1),
            0.6,
            0.3,
        ),
    )
    return 8.0 * mean_sw * agreement


def _compute_motion_bonus(min_disp, hp_win, md_win):
    """Compute the collective-motion scoring bonus."""
    if min_disp is None:
        return 0.0
    vd = min_disp[~np.isnan(min_disp)]
    speed_scale = max(np.percentile(vd, 75), 1.0) if len(vd) > 0 else 1.0
    norm_speed = np.nanmean(md_win, axis=1) / speed_scale
    mean_hpol = np.nanmean(hp_win, axis=1)
    return 10.0 * mean_hpol * norm_speed


def _try_continuous_window(
    w,
    grid,
    ng,
    both_valid,
    orientation_pol,
    heading_pol,
    heading_theta,
    orientation_theta,
    min_disp,
):
    """Score all candidate windows of width *w* on the panel grid.

    Returns (best_grid_idx, best_score) or (None, -inf).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    if ng < w + 1:
        return None, -np.inf

    gv = both_valid[grid]
    win_valid = sliding_window_view(gv, w)
    all_ok = np.all(win_valid, axis=1)
    prev_ok = np.empty(ng - w + 1, dtype=bool)
    prev_ok[0] = False
    prev_ok[1:] = gv[: ng - w]
    candidate = all_ok & prev_ok

    if not np.any(candidate):
        return None, -np.inf

    op_win = sliding_window_view(orientation_pol[grid], w)
    hp_arr = (
        heading_pol[grid] if heading_pol is not None else np.full(ng, np.nan)
    )
    hp_win = sliding_window_view(hp_arr, w)
    md_arr = min_disp[grid] if min_disp is not None else np.full(ng, np.nan)
    md_win = sliding_window_view(md_arr, w)

    ot_uw = _safe_unwrap(orientation_theta[grid])
    ht_raw = (
        heading_theta[grid]
        if heading_theta is not None
        else np.full(ng, np.nan)
    )
    ht_uw = _safe_unwrap(ht_raw)

    op_metric = _metric_scores(op_win)
    hp_metric = _metric_scores(hp_win)
    sweep_bonus = _compute_sweep_bonus(ot_uw, ht_uw, ng, w)
    motion_bonus = _compute_motion_bonus(min_disp, hp_win, md_win)

    scores = op_metric + hp_metric + sweep_bonus + motion_bonus
    scores[~candidate] = -np.inf

    best_i = np.argmax(scores)
    if scores[best_i] == -np.inf:
        return None, -np.inf
    return best_i, scores[best_i]


def _log_continuous_segment(
    panel_frames,
    has_orientation,
    has_heading,
    orientation_pol,
    heading_pol,
    n_panels,
    start_frame,
    best_score,
):
    """Print diagnostic information for a found continuous segment."""
    print(
        f"\n{n_panels}-panel continuous segment starting at "
        f"frame {start_frame} (score={best_score:.3f})"
    )
    print(f"Panel frames: {panel_frames}")
    if np.any(has_orientation):
        vb = orientation_pol[panel_frames][has_orientation]
        n_vis = np.sum(vb >= MIN_VISUAL_POL)
        print(
            f"Orientation pol range: {np.nanmin(vb):.3f} to "
            f"{np.nanmax(vb):.3f} ({n_vis}/{len(vb)} panels >= "
            f"{MIN_VISUAL_POL})"
        )
    if np.any(has_heading) and heading_pol is not None:
        vh = heading_pol[panel_frames][has_heading]
        n_vis_h = np.sum(vh >= MIN_VISUAL_POL)
        print(
            f"Heading pol range: {np.nanmin(vh):.3f} to "
            f"{np.nanmax(vh):.3f} ({n_vis_h}/{len(vh)} panels >= "
            f"{MIN_VISUAL_POL})"
        )


def find_best_segment(
    orientation_pol,
    orientation_theta,
    heading_pol_1f,
    heading_pol,
    heading_theta,
    min_disp,
    orientation_valid,
    vel_valid,
    pos_valid,
    both_valid,
    total_frames,
    fps,
):
    """Find the best continuous segment by scanning library output arrays.

    Subsamples all timeseries at the panel spacing (= fps), then scores
    every candidate window with vectorised numpy.

    Parameters
    ----------
    orientation_pol, orientation_theta : ndarray
        Orientation polarization magnitude and angle per frame.
    heading_pol_1f : ndarray
        1-frame heading polarization (sparse fallback only).
    heading_pol, heading_theta : ndarray or None
        fps-frame heading polarization and angle per frame.
    min_disp : ndarray or None
        Min displacement magnitude per frame (across individuals).
    orientation_valid, vel_valid, pos_valid, both_valid : ndarray
        Per-frame boolean validity masks.
    total_frames, fps : int or None
        Frame count and frame rate.

    Returns
    -------
    panel_frames, prev_frames, is_continuous, has_orientation, has_heading,
    frame_interval, num_panels

    """
    if fps is None:
        print("FPS unknown - using sparse frame selection")
        return _find_sparse_frames(
            orientation_pol,
            heading_pol_1f,
            orientation_valid,
            vel_valid,
            total_frames,
        )

    frame_interval = int(fps)
    grid = np.arange(0, total_frames, frame_interval)
    ng = len(grid)

    window_args = (
        grid,
        ng,
        both_valid,
        orientation_pol,
        heading_pol,
        heading_theta,
        orientation_theta,
        min_disp,
    )

    best_i, best_score = _try_continuous_window(NUM_PANELS, *window_args)
    n_panels = NUM_PANELS

    if best_i is None:
        print(
            f"\nNo full {NUM_PANELS}-panel segment found. "
            "Searching for partial continuous..."
        )
        for n_panels in range(NUM_PANELS - 1, MIN_CONTINUOUS_PANELS - 1, -1):
            best_i, best_score = _try_continuous_window(n_panels, *window_args)
            if best_i is not None:
                print(
                    f"Found {n_panels}-panel segment (score={best_score:.3f})"
                )
                break

    if best_i is not None:
        panel_frames = grid[best_i] + np.arange(n_panels) * frame_interval
        prev_frames = panel_frames - frame_interval

        has_orientation = np.array(
            [orientation_valid[f] for f in panel_frames]
        )
        has_heading = np.array(
            [
                pos_valid[prev_frames[p]] and pos_valid[panel_frames[p]]
                for p in range(n_panels)
            ]
        )

        _log_continuous_segment(
            panel_frames,
            has_orientation,
            has_heading,
            orientation_pol,
            heading_pol,
            n_panels,
            grid[best_i],
            best_score,
        )

        return (
            panel_frames,
            prev_frames,
            True,
            has_orientation,
            has_heading,
            frame_interval,
            n_panels,
        )

    print(
        f"\nNo continuous segment (>={MIN_CONTINUOUS_PANELS} panels) found. "
        "Selecting sparse frames..."
    )
    return _find_sparse_frames(
        orientation_pol,
        heading_pol_1f,
        orientation_valid,
        vel_valid,
        total_frames,
    )


# ── Polarization log helpers ─────────────────────────────────────────────────


def _log_header(
    log,
    video_filename,
    fps,
    frame_interval,
    num_panels,
    is_continuous,
    auto_log,
    node_config,
    node_names,
):
    """Write the log file header and configuration block."""
    log("POLARIZATION ANALYSIS LOG")
    log(f"\nVideo: {video_filename}")
    log(f"FPS: {fps if fps is not None else 'Unknown'}")
    if fps is not None:
        log(f"Frame interval: {frame_interval} frames (1 second)")
    else:
        log(f"Frame interval: {frame_interval} frames (estimated)")
    log(f"Number of panels: {num_panels}")
    log(f"Segment type: {'CONTINUOUS' if is_continuous else 'SPARSE'}")

    if auto_log is not None:
        log("\n" + auto_log)

    log("\n")
    log("FINAL NODE CONFIGURATION")
    from_idx = node_config["orientation_from"]
    to_idx = node_config["orientation_to"]
    vel_idx = node_config["velocity_node"]
    ap_ok = node_config.get("ap_validated", False)

    from_label = "(posterior)" if ap_ok else "(from_node)"
    to_label = "(anterior)" if ap_ok else "(to_node)"

    log(f"\n  Velocity node:     {node_names[vel_idx]}[{vel_idx}]")
    log(
        f"  Orientation FROM:  {node_names[from_idx]}[{from_idx}] {from_label}"
    )
    log(f"  Orientation TO:    {node_names[to_idx]}[{to_idx}] {to_label}")
    log("  Fallback TO:       None (strict mode)")

    log("\n")
    log("POLARIZATION VERIFICATION")
    log("\nCoordinate system notes:")
    log("  - Image coords: y increases DOWNWARD (top=0)")
    log("  - Polar plot: math convention (0=E, 90=N, CCW positive)")
    log("  - Y-flip: dy = -(to_y - from_y) for image→Cartesian")
    log("  - Heading from 1-second displacement (matches Row 1)")


def _compute_unit_vectors_and_log(coords, total, log, label_prefix):
    """Compute unit vectors from coordinate pairs, logging each animal.

    Parameters
    ----------
    coords : list of tuples
        Each element: (animal_idx, from_x, from_y, to_x, to_y).
    total : int
        Total number of animals.
    log : callable
        Logging function.
    label_prefix : str
        Prefix for log messages.

    Returns
    -------
    list of (ux, uy)
        Unit vectors for valid animals.

    """
    unit_vectors = []
    for a, fx, fy, tx, ty in coords:
        if np.isnan(fx) or np.isnan(tx):
            nan_row = (
                f"  {a:<8} {'NaN':>8} {'NaN':>8} "
                f"{'NaN':>8} {'NaN':>8} │ {'---':>8} "
                f"{'---':>8} {'---':>8} │ {'---':>8}"
            )
            log(nan_row)
            continue

        dx = tx - fx
        dy_img = ty - fy
        dy_cart = -dy_img
        length = np.hypot(dx, dy_cart)

        if length > 0:
            angle_deg = np.degrees(np.arctan2(dy_cart, dx))
            unit_vectors.append((dx / length, dy_cart / length))
            row = (
                f"  {a:<8} {fx:>8.1f} {fy:>8.1f} "
                f"{tx:>8.1f} {ty:>8.1f} │ {dx:>8.1f} "
                f"{dy_img:>8.1f} {dy_cart:>8.1f} │ {angle_deg:>8.1f}"
            )
            log(row)
    return unit_vectors


def _log_polarization_check(
    log, unit_vectors, stored_pol, stored_theta, total_animals, label
):
    """Log a polarization verification check."""
    if not unit_vectors:
        return

    sum_ux = sum(u[0] for u in unit_vectors)
    sum_uy = sum(u[1] for u in unit_vectors)
    n = len(unit_vectors)
    mean_ux, mean_uy = sum_ux / n, sum_uy / n
    computed_pol = np.hypot(mean_ux, mean_uy)
    computed_theta_deg = np.degrees(np.arctan2(mean_uy, mean_ux))

    stored_theta_deg = (
        np.degrees(-stored_theta) if not np.isnan(stored_theta) else np.nan
    )

    log(f"\n  {label} POLARIZATION CHECK:")
    log(f"    Mean unit vector: ({mean_ux:.4f}, {mean_uy:.4f})")
    log(
        f"    Computed:  pol={computed_pol:.4f}, "
        f"theta={computed_theta_deg:.1f}°"
    )
    log(f"    Stored:    pol={stored_pol:.4f}, theta={stored_theta_deg:.1f}°")

    if n < total_animals:
        log(f"    MATCH: SKIPPED ({n}/{total_animals} valid)")
    elif np.isnan(stored_pol):
        log("    MATCH: ✗ NO (stored is NaN unexpectedly)")
    else:
        match = np.isclose(computed_pol, stored_pol, atol=1e-6)
        log(f"    MATCH: {'✓ YES' if match else '✗ NO'}")


def _log_panel(
    log,
    p,
    curr_f,
    prev_f,
    head_x,
    head_y,
    tail_x,
    tail_y,
    vel_x,
    vel_y,
    orientation_pol,
    heading_pol,
    orientation_theta,
    heading_theta,
    total_animals,
    node_config,
    node_names,
):
    """Log a single panel's orientation and heading data."""
    from_idx = node_config["orientation_from"]
    ap_ok = node_config.get("ap_validated", False)
    from_name = node_names[from_idx]
    to_name = node_names[node_config["orientation_to"]]

    log(f"\n{'─' * 80}")
    log(f"PANEL {p} ({p}s) | Frame {curr_f} | Prev frame {prev_f}")
    log(f"{'─' * 80}")

    # Orientation
    log(f"\n  ORIENTATION ({from_name} → {to_name}):")
    col_from_x = "post_x" if ap_ok else "from_x"
    col_from_y = "post_y" if ap_ok else "from_y"
    col_to_x = "ant_x" if ap_ok else "to_x"
    col_to_y = "ant_y" if ap_ok else "to_y"
    hdr = (
        f"  {'Animal':<8} {col_from_x:>8} {col_from_y:>8} "
        f"{col_to_x:>8} {col_to_y:>8} │ {'dx':>8} "
        f"{'dy_img':>8} {'dy_cart':>8} │ {'angle°':>8}"
    )
    log(hdr)
    sep = (
        f"  {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} │ "
        f"{'-' * 8} {'-' * 8} {'-' * 8} │ {'-' * 8}"
    )
    log(sep)

    orient_coords = [
        (
            a,
            tail_x[curr_f, a],
            tail_y[curr_f, a],
            head_x[curr_f, a],
            head_y[curr_f, a],
        )
        for a in range(total_animals)
    ]
    orient_uvs = _compute_unit_vectors_and_log(
        orient_coords,
        total_animals,
        log,
        "ORIENTATION",
    )
    _log_polarization_check(
        log,
        orient_uvs,
        orientation_pol[curr_f],
        orientation_theta[curr_f],
        total_animals,
        "ORIENTATION",
    )

    # Heading
    log(f"\n  HEADING/VELOCITY (frame {prev_f} to {curr_f}):")
    hdr_h = (
        f"  {'Animal':<8} {'prev_x':>8} {'prev_y':>8} "
        f"{'curr_x':>8} {'curr_y':>8} │ {'dx':>8} "
        f"{'dy_img':>8} {'dy_cart':>8} │ {'angle°':>8}"
    )
    log(hdr_h)
    log(sep)

    heading_coords = [
        (
            a,
            vel_x[prev_f, a],
            vel_y[prev_f, a],
            vel_x[curr_f, a],
            vel_y[curr_f, a],
        )
        for a in range(total_animals)
    ]
    heading_uvs = _compute_unit_vectors_and_log(
        heading_coords,
        total_animals,
        log,
        "HEADING",
    )
    _log_polarization_check(
        log,
        heading_uvs,
        heading_pol[p],
        heading_theta[p],
        total_animals,
        "HEADING",
    )


def write_polarization_log(
    panel_frames,
    prev_frames,
    head_x,
    head_y,
    tail_x,
    tail_y,
    vel_x,
    vel_y,
    orientation_pol,
    panel_heading_pol,
    orientation_theta,
    panel_heading_theta,
    total_animals,
    node_config,
    node_names,
    auto_log,
    video_filename,
    fps,
    is_continuous,
    frame_interval,
    num_panels,
    output_file=None,
):
    """Write a detailed textual log of polarization computations.

    The log includes per-panel and per-frame diagnostics about orientation and
    heading, velocity displacements, computed angles and
    resultant magnitudes.

    Parameters
    ----------
    panel_frames : Sequence[int]
        The list of frame indices corresponding to the current panel frames.
    prev_frames : Sequence[int]
        The list of previous frame indices used for ghost overlays.
    head_x, head_y : np.ndarray
        2‑D arrays of shape (frames, animals) for the to_node keypoint.
    tail_x, tail_y : np.ndarray
        2‑D arrays of shape (frames, animals) for the from_node keypoint.
    vel_x, vel_y : np.ndarray
        2‑D arrays of shape (frames, animals) for the velocity node.
    orientation_pol : np.ndarray
        Orientation polarization values per frame.
    panel_heading_pol : np.ndarray
        Heading polarization values per panel.
    orientation_theta : np.ndarray
        Orientation angles (radians) per frame.
    panel_heading_theta : np.ndarray
        Heading angles (radians) per panel.
    total_animals : int
        Number of individuals.
    node_config : dict
        Node configuration dictionary.
    node_names : list[str]
        Keypoint names.
    auto_log : str | None
        Optional auto-detection log text.
    video_filename : str
        Name of the video file.
    fps : int | None
        Frames per second.
    is_continuous : bool
        Whether a continuous segment was found.
    frame_interval : int
        Frames between panels.
    num_panels : int
        Number of panels.
    output_file : Path | str | None, optional
        Optional log file path.

    """
    lines = []

    def log(text=""):
        print(text)
        lines.append(text)

    _log_header(
        log,
        video_filename,
        fps,
        frame_interval,
        num_panels,
        is_continuous,
        auto_log,
        node_config,
        node_names,
    )

    for p, curr_f in enumerate(panel_frames):
        _log_panel(
            log,
            p,
            curr_f,
            prev_frames[p],
            head_x,
            head_y,
            tail_x,
            tail_y,
            vel_x,
            vel_y,
            orientation_pol,
            panel_heading_pol,
            orientation_theta,
            panel_heading_theta,
            total_animals,
            node_config,
            node_names,
        )

    log("\n")
    log("END LOG")

    if output_file is not None:
        with open(output_file, "w") as f:
            f.write("\n".join(lines))
        print(f"Log saved to: {output_file}")


# ── Figure data container (reduces create_figure parameters) ─────────────────


@dataclass
class FigureData:
    """Container for the data arrays used to render the polarization figure."""

    frames: dict
    panel_frames: np.ndarray
    prev_frames: np.ndarray
    head_x: np.ndarray
    head_y: np.ndarray
    tail_x: np.ndarray
    tail_y: np.ndarray
    vel_x: np.ndarray
    vel_y: np.ndarray
    orientation_pol: np.ndarray
    heading_pol: np.ndarray
    orientation_theta: np.ndarray
    heading_theta: np.ndarray
    total_animals: int
    has_orientation: np.ndarray
    has_heading: np.ndarray


@dataclass
class FigureConfig:
    """Container for configuration options for the polarization figure."""

    video_filename: str
    fps: float | None
    node_config: dict
    node_names: list[str]
    is_continuous: bool
    frame_interval: int
    num_panels: int
    video_metadata: VideoMetadata | None = None
    video_available: bool = True
    overlay_alpha: float = OVERLAY_ALPHA


# ── Figure rendering helpers ─────────────────────────────────────────────────


def _get_frame_dimensions(cfg, data):
    """Return (width, height) from metadata or frames dict."""
    if (
        cfg.video_metadata is not None
        and cfg.video_metadata.resolution is not None
    ):
        return cfg.video_metadata.resolution
    if cfg.video_available and data.frames:
        sample = next(iter(data.frames.values()))
        frame_h, frame_w = sample.shape[:2]
        return frame_w, frame_h
    raise ValueError("Cannot determine frame dimensions")


def _add_figure_title(fig, cfg):
    """Add filename and optional sparse-mode note to the figure."""
    fig.text(
        0.03,
        0.975,
        cfg.video_filename,
        color=DARK_FG,
        fontsize=7,
        ha="left",
        va="top",
    )

    if not cfg.is_continuous:
        sparse_note = (
            "Sparse mode: Each panel shows frame f overlaid with f-1.\n"
            "Heading vectors show single-frame displacement.\n"
            "Polar plots use same consecutive frame pairs."
        )
        fig.text(
            0.03,
            0.955,
            sparse_note,
            color=[0.7, 0.7, 0.7],
            fontsize=7,
            ha="left",
            va="top",
            linespacing=1.3,
        )


def _style_panel_ax(ax, frame_w, frame_h):
    """Apply common panel axis styling."""
    ax.set_xlim(0, frame_w)
    ax.set_ylim(frame_h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _render_heading_panel(ax, p, data, cfg, frame_w, frame_h):
    """Render a single heading (Row 1) panel."""
    curr_f = data.panel_frames[p]
    prev_f = data.prev_frames[p]

    ax.set_facecolor(DARK_BG)

    if not cfg.video_available:
        ax.text(
            0.5,
            0.5,
            "Skipped - video not available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=15,
            color=DARK_FG,
            wrap=True,
        )
        return

    if not data.has_heading[p]:
        curr_img = data.frames[curr_f].astype(float) / 255
        ax.imshow(curr_img)
        return

    curr_img = data.frames[curr_f].astype(float) / 255
    prev_img = data.frames[prev_f].astype(float) / 255

    curr_positions = [
        (data.vel_x[curr_f, a], data.vel_y[curr_f, a])
        for a in range(data.total_animals)
    ]
    prev_positions = [
        (data.vel_x[prev_f, a], data.vel_y[prev_f, a])
        for a in range(data.total_animals)
    ]

    blended = adaptive_blend_frames(
        curr_img,
        prev_img,
        cfg.overlay_alpha,
        curr_positions=curr_positions,
        prev_positions=prev_positions,
        mask_radius=100,
    )
    ax.imshow(blended)

    _draw_displacement_arrows(ax, p, data, curr_f, prev_f)


def _draw_displacement_arrows(ax, p, data, curr_f, prev_f):
    """Draw velocity displacement arrows for all animals."""
    for a in range(data.total_animals):
        c = ANIMAL_COLORS[a % len(ANIMAL_COLORS)]
        curr_vx = data.vel_x[curr_f, a]
        curr_vy = data.vel_y[curr_f, a]
        prev_vx = data.vel_x[prev_f, a]
        prev_vy = data.vel_y[prev_f, a]

        if np.isnan(curr_vx) or np.isnan(prev_vx):
            continue
        dx = curr_vx - prev_vx
        dy = curr_vy - prev_vy
        if dx == 0 and dy == 0:
            continue

        quiver_common = {
            "angles": "xy",
            "scale_units": "xy",
            "scale": 1,
            "minshaft": 1.5,
            "minlength": 0.5,
        }
        ax.quiver(
            prev_vx,
            prev_vy,
            dx,
            dy,
            color="black",
            width=0.012,
            headwidth=4.5,
            headlength=5,
            headaxislength=4.5,
            zorder=4,
            **quiver_common,
        )
        ax.quiver(
            prev_vx,
            prev_vy,
            dx,
            dy,
            color=c,
            width=0.008,
            headwidth=4,
            headlength=5,
            headaxislength=4,
            zorder=5,
            **quiver_common,
        )


def _set_panel_title(ax, p, data, cfg, show_prev=False):
    """Set the time/frame title for a panel."""
    curr_f = data.panel_frames[p]
    prev_f = data.prev_frames[p]
    is_edge = p == 0 or p == cfg.num_panels - 1

    if cfg.is_continuous:
        if is_edge:
            ax.set_title(
                f"Frame: {curr_f}\n{p}s",
                color=DARK_FG,
                fontsize=12,
                pad=8,
            )
        else:
            ax.set_title(f"{p}s", color=DARK_FG, fontsize=12)
    elif show_prev:
        if is_edge:
            ax.set_title(
                f"Frames: {prev_f}, {curr_f}",
                color=DARK_FG,
                fontsize=11,
                pad=8,
            )
        else:
            ax.set_title(f"{prev_f}, {curr_f}", color=DARK_FG, fontsize=12)
    else:
        if is_edge:
            ax.set_title(
                f"Frame: {curr_f}",
                color=DARK_FG,
                fontsize=12,
                pad=8,
            )
        else:
            ax.set_title(f"{curr_f}", color=DARK_FG, fontsize=12)


def _render_heading_row(fig, gs, data, cfg, frame_w, frame_h, current_row):
    """Render the heading row (Row 1) of the figure."""
    vel_node_name = cfg.node_names[cfg.node_config["velocity_node"]]
    fig.text(
        0.355,
        0.96,
        "Heading  Across  Frames",
        color=DARK_FG,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    fig.text(
        0.570,
        0.96,
        f"({vel_node_name} → {vel_node_name})",
        color=DARK_FG,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    for p in range(cfg.num_panels):
        ax = fig.add_subplot(gs[current_row, p])
        _render_heading_panel(ax, p, data, cfg, frame_w, frame_h)
        _style_panel_ax(ax, frame_w, frame_h)
        _set_panel_title(ax, p, data, cfg, show_prev=True)


def _render_orientation_panel(ax, p, data, cfg):
    """Render a single orientation (Row 2) panel."""
    curr_f = data.panel_frames[p]
    ax.set_facecolor(DARK_BG)

    if not cfg.video_available:
        ax.text(
            0.5,
            0.5,
            "Skipped - video not available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=15,
            color=DARK_FG,
            wrap=True,
        )
        return

    curr_img = data.frames[curr_f].astype(float) / 255
    ax.imshow(curr_img)

    if not data.has_orientation[p]:
        return

    for a in range(data.total_animals):
        c = ANIMAL_COLORS[a % len(ANIMAL_COLORS)]
        ctx = data.tail_x[curr_f, a]
        cty = data.tail_y[curr_f, a]
        chx = data.head_x[curr_f, a]
        chy = data.head_y[curr_f, a]

        if np.isnan(ctx) or np.isnan(chx):
            continue

        ax.plot([ctx, chx], [cty, chy], "-", color=c, linewidth=2)
        ax.plot(
            ctx,
            cty,
            "s",
            color=c,
            markersize=5,
            markerfacecolor=c,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )
        ax.plot(
            chx,
            chy,
            "o",
            color=c,
            markersize=5,
            markerfacecolor=c,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )


def _render_orientation_row(
    fig, gs, data, cfg, frame_w, frame_h, current_row, show_heading_row
):
    """Render the orientation row (Row 2) of the figure."""
    from_node_name = cfg.node_names[cfg.node_config["orientation_from"]]
    to_node_name = cfg.node_names[cfg.node_config["orientation_to"]]
    title_y = 0.62 if show_heading_row else 0.96

    fig.text(
        0.355,
        title_y,
        "Orientation  Within  Frame",
        color=DARK_FG,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    fig.text(
        0.560,
        title_y,
        f"({from_node_name} → {to_node_name})",
        color=DARK_FG,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    ap_ok = cfg.node_config.get("ap_validated", False)
    from_sub = "Posterior" if ap_ok else "(from_node)"
    to_sub = "Anterior" if ap_ok else "(to_node)"
    fig.text(
        0.82,
        title_y,
        f"□ = {from_node_name}",
        color=DARK_FG,
        fontsize=10,
        ha="left",
        va="bottom",
    )
    fig.text(
        0.82,
        title_y - 0.02,
        f"     {from_sub}",
        color=DARK_FG,
        fontsize=9,
        ha="left",
        va="bottom",
    )
    fig.text(
        0.90,
        title_y,
        f"○ = {to_node_name}",
        color=DARK_FG,
        fontsize=10,
        ha="left",
        va="bottom",
    )
    fig.text(
        0.90,
        title_y - 0.02,
        f"     {to_sub}",
        color=DARK_FG,
        fontsize=9,
        ha="left",
        va="bottom",
    )

    for p in range(cfg.num_panels):
        ax = fig.add_subplot(gs[current_row, p])
        _render_orientation_panel(ax, p, data, cfg)
        _style_panel_ax(ax, frame_w, frame_h)
        if not show_heading_row:
            _set_panel_title(ax, p, data, cfg, show_prev=False)


def _render_polar_panel(ax, p, data):
    """Render a single polar plot panel."""
    curr_f = data.panel_frames[p]
    ax.set_facecolor(DARK_BG)

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlim(0, 1)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.tick_params(colors=DARK_FG, labelsize=8)
    ax.spines["polar"].set_color([0.4, 0.4, 0.45])

    ax.set_rgrids(
        [0.25, 0.5, 0.75, 1.0],
        labels=["", "0.5", "", "1.0"],
        color=DARK_FG,
        fontsize=6,
    )
    ax.grid(
        True,
        color=[0.5, 0.5, 0.55],
        alpha=0.7,
        linestyle="-",
        linewidth=0.8,
    )

    b_pol = data.orientation_pol[curr_f] if data.has_orientation[p] else np.nan
    b_theta = (
        -data.orientation_theta[curr_f] if data.has_orientation[p] else np.nan
    )
    h_pol = data.heading_pol[p] if data.has_heading[p] else np.nan
    h_theta = -data.heading_theta[p] if data.has_heading[p] else np.nan

    if (
        data.has_orientation[p]
        and not np.isnan(b_pol)
        and not np.isnan(b_theta)
    ):
        ax.annotate(
            "",
            xy=(b_theta, b_pol),
            xytext=(b_theta, 0),
            arrowprops={
                "arrowstyle": "->",
                "color": NET_ORIENTATION_COLOR,
                "lw": 2.5,
                "mutation_scale": 12,
            },
        )

    if data.has_heading[p] and not np.isnan(h_pol) and not np.isnan(h_theta):
        ax.annotate(
            "",
            xy=(h_theta, h_pol),
            xytext=(h_theta, 0),
            arrowprops={
                "arrowstyle": "->",
                "color": NET_HEADING_COLOR,
                "lw": 2.5,
                "mutation_scale": 12,
            },
        )

    if data.has_orientation[p] and not np.isnan(b_pol):
        ax.text(
            0.5,
            1.32,
            f"$p_b$={b_pol:.2f}",
            transform=ax.transAxes,
            color=NET_ORIENTATION_COLOR,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
        )
    if data.has_heading[p] and not np.isnan(h_pol):
        ax.text(
            0.5,
            1.20,
            f"$p_h$={h_pol:.2f}",
            transform=ax.transAxes,
            color=NET_HEADING_COLOR,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
        )


def _get_polar_labels(p, num_panels):
    """Return theta grid labels for polar panel *p*."""
    all_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    if p == 0:
        labels = ["", "45°", "90°", "135°", "180°", "225°", "270°", "315°"]
    elif p == num_panels - 1:
        labels = ["0°", "45°", "90°", "135°", "", "225°", "270°", "315°"]
    else:
        labels = ["", "45°", "90°", "135°", "", "225°", "270°", "315°"]
    return all_angles, labels


def _render_polar_row(fig, gs, data, cfg, current_row):
    """Render the polar plots row."""
    for p in range(cfg.num_panels):
        ax = fig.add_subplot(gs[current_row, p], projection="polar")
        _render_polar_panel(ax, p, data)
        all_angles, labels = _get_polar_labels(p, cfg.num_panels)
        ax.set_thetagrids(all_angles, labels=labels)


def _render_legend_and_footer(
    fig, cfg, show_heading_row, show_orientation_row
):
    """Render legend and footer text on the figure."""
    from_node_name = cfg.node_names[cfg.node_config["orientation_from"]]
    to_node_name = cfg.node_names[cfg.node_config["orientation_to"]]
    vel_node_name = cfg.node_names[cfg.node_config["velocity_node"]]

    legend_x = 0.03
    if show_orientation_row:
        fig.text(
            legend_x,
            0.015,
            f"$p_b$ = Orientation ({from_node_name}→{to_node_name})  ",
            color=NET_ORIENTATION_COLOR,
            fontsize=9,
            ha="left",
            va="bottom",
        )
        legend_x = 0.22
    if show_heading_row:
        fig.text(
            legend_x,
            0.015,
            f"$p_h$ = Heading ({vel_node_name} velocity)",
            color=NET_HEADING_COLOR,
            fontsize=9,
            ha="left",
            va="bottom",
        )

    fig.text(
        0.5,
        0.015,
        "Computed Polarization",
        color=DARK_FG,
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    if cfg.is_continuous:
        fig.text(
            0.97,
            0.015,
            f"Frame interval={cfg.frame_interval} "
            f"({1000 * cfg.frame_interval / cfg.fps:.0f}ms)",
            color=DARK_FG,
            fontsize=10,
            ha="right",
            va="bottom",
        )
    else:
        fig.text(
            0.97,
            0.015,
            "Sparse frames (non-continuous)",
            color=DARK_FG,
            fontsize=10,
            ha="right",
            va="bottom",
        )


def create_figure(data: FigureData, cfg: FigureConfig):
    """Create the static figure with frame panels and polar plots.

    Parameters
    ----------
    data : FigureData
        Container holding all data arrays for rendering.
    cfg : FigureConfig
        Container holding configuration and metadata.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object containing all subplots.

    """
    show_heading_row = np.any(data.has_heading)
    show_orientation_row = np.any(data.has_orientation)

    if show_heading_row and show_orientation_row:
        height_ratios = [1.2, 1.2, 1]
        n_rows = 3
    elif show_heading_row or show_orientation_row:
        height_ratios = [1.2, 1]
        n_rows = 2
    else:
        raise ValueError("No valid data to visualize")

    fig_width = 2.5 * cfg.num_panels
    fig_height = 11 if n_rows == 3 else 8
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=DARK_BG)
    gs = gridspec.GridSpec(
        n_rows,
        cfg.num_panels,
        height_ratios=height_ratios,
        hspace=0.12,
        wspace=0.08,
        left=0.03,
        right=0.97,
        top=0.94,
        bottom=0.04,
    )

    frame_w, frame_h = _get_frame_dimensions(cfg, data)

    _add_figure_title(fig, cfg)

    current_row = 0

    if show_heading_row:
        _render_heading_row(fig, gs, data, cfg, frame_w, frame_h, current_row)
        current_row += 1

    if show_orientation_row:
        _render_orientation_row(
            fig,
            gs,
            data,
            cfg,
            frame_w,
            frame_h,
            current_row,
            show_heading_row,
        )
        current_row += 1

    _render_polar_row(fig, gs, data, cfg, current_row)

    _render_legend_and_footer(fig, cfg, show_heading_row, show_orientation_row)

    return fig


# ── Dataset processing ───────────────────────────────────────────────────────


def _determine_overlay_alpha(video_file):
    """Return the overlay alpha based on the video filename."""
    if video_file is None:
        return OVERLAY_ALPHA
    video_name = video_file.name
    if (
        "free-moving-4gerbils" in video_name
        or "freemoving-2bees" in video_name
    ):
        return 0.7
    return OVERLAY_ALPHA


def _apply_ap_corrections(
    ds, node_config, node_names, from_keypoint, to_keypoint
):
    """Run AP inference and apply corrections to node_config if needed.

    Returns the (possibly updated) from_keypoint and to_keypoint.
    """
    inference_result = _run_ap_inference(ds, from_keypoint, to_keypoint)

    suggested = report_ap_scenario(
        ds,
        from_keypoint,
        to_keypoint,
        node_names,
        inference_result=inference_result,
    )
    if suggested is not None:
        s_from, s_to = suggested
        node_config["orientation_from"] = s_from
        node_config["orientation_to"] = s_to
        node_config["ap_validated"] = True
        return node_names[s_from], node_names[s_to]

    should_flip = check_ap_ordering(
        ds,
        from_keypoint,
        to_keypoint,
        inference_result=inference_result,
    )
    if should_flip:
        old_from = node_config["orientation_from"]
        old_to = node_config["orientation_to"]
        node_config["orientation_from"] = old_to
        node_config["orientation_to"] = old_from
        node_config["ap_validated"] = True
        return node_names[node_config["orientation_from"]], node_names[
            node_config["orientation_to"]
        ]

    return from_keypoint, to_keypoint


def _compute_validity_masks(ds, from_keypoint, to_keypoint, velocity_keypoint):
    """Compute per-frame validity masks."""
    from_pos = ds.position.sel(keypoints=from_keypoint).values
    to_pos = ds.position.sel(keypoints=to_keypoint).values
    vel_pos_all = ds.position.sel(keypoints=velocity_keypoint).values

    from_ok = ~np.isnan(from_pos).any(axis=1)
    to_ok = ~np.isnan(to_pos).any(axis=1)
    is_valid = from_ok & to_ok
    is_pos_valid = ~np.isnan(vel_pos_all).any(axis=1)
    is_vel_valid = np.zeros_like(is_pos_valid)
    is_vel_valid[1:] = is_pos_valid[1:] & is_pos_valid[:-1]

    orientation_valid = np.all(is_valid, axis=1)
    vel_valid = np.all(is_vel_valid, axis=1)
    pos_valid = np.all(is_pos_valid, axis=1)
    both_valid = orientation_valid & pos_valid

    return (
        from_pos,
        to_pos,
        vel_pos_all,
        is_valid,
        is_pos_valid,
        orientation_valid,
        vel_valid,
        pos_valid,
        both_valid,
    )


def _load_needed_video_frames(video_file, needed, total_frames):
    """Load only the needed video frames into a dict."""
    frames = {}
    video_available = False
    if video_file is None:
        print("\nNo video file - frames will not be displayed")
        return frames, video_available

    print(f"\nLoading selected video frames from: {video_file.name}")
    cap = cv2.VideoCapture(str(video_file))
    video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_total < total_frames:
        print(
            f"  Warning: Video has fewer frames ({video_total}) "
            f"than tracking data ({total_frames})"
        )
        print("  Video frames will not be displayed")
        cap.release()
        return frames, video_available

    for f in sorted(needed):
        if f < 0 or f >= video_total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if ret:
            frames[f] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_available = len(frames) > 0
    print(f"  Loaded {len(frames)} frames (of {video_total} total)")
    cap.release()
    return frames, video_available


def process_dataset(dataset_idx, node_config=None):
    """Process a dataset: load, analyze, and create visualization.

    Parameters
    ----------
    dataset_idx : int
        Index of the dataset to process (0-based).
    node_config : dict, optional
        Pre-defined node configuration.

    """
    ds, video_file, video_metadata, slp_filename, node_config = load_data(
        dataset_idx, node_config=node_config
    )

    overlay_alpha = _determine_overlay_alpha(video_file)

    node_names = list(ds.keypoints.values)
    total_animals = ds.sizes["individuals"]
    total_frames = ds.sizes["time"]
    fps = ds.attrs.get("fps")

    from_keypoint = node_names[node_config["orientation_from"]]
    to_keypoint = node_names[node_config["orientation_to"]]
    velocity_keypoint = node_names[node_config["velocity_node"]]

    if not node_config.get("ap_validated", False):
        from_keypoint, to_keypoint = _apply_ap_corrections(
            ds,
            node_config,
            node_names,
            from_keypoint,
            to_keypoint,
        )

    # Phase 1: Validity masks
    print("\nComputing validity masks...")
    (
        from_pos,
        to_pos,
        vel_pos_all,
        is_valid,
        is_pos_valid,
        orientation_valid,
        vel_valid,
        pos_valid,
        both_valid,
    ) = _compute_validity_masks(
        ds, from_keypoint, to_keypoint, velocity_keypoint
    )

    # Phase 2: Polarization
    print("Computing polarization metrics...")
    orientation_pol_da, orientation_theta_da = compute_polarization(
        ds.position,
        body_axis_keypoints=(from_keypoint, to_keypoint),
        return_angle=True,
        validate_ap=False,
    )
    heading_pol_da, heading_theta_da = compute_polarization(
        ds.position.sel(keypoints=velocity_keypoint),
        displacement_frames=1,
        return_angle=True,
    )
    orientation_pol = orientation_pol_da.values
    orientation_theta = orientation_theta_da.values
    heading_pol = heading_pol_da.values
    heading_theta = heading_theta_da.values

    # Phase 3: Segment selection
    print("Finding best segment...")
    fps_for_segment, fps_heading_pol, fps_heading_theta, min_disp_full = (
        _compute_fps_heading(
            ds, velocity_keypoint, vel_pos_all, fps, total_frames
        )
    )

    segment_result = find_best_segment(
        orientation_pol,
        orientation_theta,
        heading_pol,
        fps_heading_pol,
        fps_heading_theta,
        min_disp_full,
        orientation_valid,
        vel_valid,
        pos_valid,
        both_valid,
        total_frames,
        fps_for_segment,
    )
    (
        panel_frames,
        prev_frames,
        is_continuous,
        has_orientation,
        has_heading,
        frame_interval,
        num_panels,
    ) = segment_result

    if is_continuous:
        panel_heading_pol = fps_heading_pol[panel_frames]
        panel_heading_theta = fps_heading_theta[panel_frames]
    else:
        print(
            "Using precomputed heading polarization (1-frame displacement)..."
        )
        panel_heading_pol = heading_pol[panel_frames]
        panel_heading_theta = heading_theta[panel_frames]

    # Phase 4: Extract per-animal coordinates
    print("Extracting per-animal coordinates for selected frames...")
    needed = np.unique(np.concatenate([panel_frames, prev_frames]))

    head_x, head_y, tail_x, tail_y = _extract_orientation_coords(
        needed,
        total_frames,
        total_animals,
        is_valid,
        from_pos,
        to_pos,
    )
    vel_x, vel_y = _extract_velocity_coords(
        needed,
        total_frames,
        total_animals,
        is_pos_valid,
        vel_pos_all,
    )

    frames, video_available = _load_needed_video_frames(
        video_file,
        needed,
        total_frames,
    )

    output_dir = ROOT_PATH / "exports" / "polarization-demos"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    base_stem = Path(slp_filename).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    video_filename = (
        video_file.name
        if video_file is not None
        else f"{slp_filename} (no video)"
    )
    write_polarization_log(
        panel_frames,
        prev_frames,
        head_x,
        head_y,
        tail_x,
        tail_y,
        vel_x,
        vel_y,
        orientation_pol,
        panel_heading_pol,
        orientation_theta,
        panel_heading_theta,
        total_animals,
        node_config,
        node_names,
        node_config.get("auto_log"),
        video_filename,
        fps,
        is_continuous,
        frame_interval,
        num_panels,
    )

    print("\nCreating figure...")
    fig_data = FigureData(
        frames=frames,
        panel_frames=panel_frames,
        prev_frames=prev_frames,
        head_x=head_x,
        head_y=head_y,
        tail_x=tail_x,
        tail_y=tail_y,
        vel_x=vel_x,
        vel_y=vel_y,
        orientation_pol=orientation_pol,
        heading_pol=panel_heading_pol,
        orientation_theta=orientation_theta,
        heading_theta=panel_heading_theta,
        total_animals=total_animals,
        has_orientation=has_orientation,
        has_heading=has_heading,
    )
    fig_cfg = FigureConfig(
        video_filename=slp_filename,
        fps=fps,
        node_config=node_config,
        node_names=node_names,
        is_continuous=is_continuous,
        frame_interval=frame_interval,
        num_panels=num_panels,
        video_metadata=video_metadata,
        video_available=video_available,
        overlay_alpha=overlay_alpha,
    )
    fig = create_figure(fig_data, fig_cfg)

    base_name = base_stem + f"_{timestamp}_polarization-plot"
    _save_figure(fig, figures_dir, base_name, fps)
    plt.close(fig)


def _compute_fps_heading(
    ds, velocity_keypoint, vel_pos_all, fps, total_frames
):
    """Compute fps-frame heading polarization for segment scoring."""
    if fps is None:
        return None, None, None, None

    print(f"  Computing {int(fps)}-frame heading for segment scoring...")
    vel_pos_da = ds.position.sel(keypoints=velocity_keypoint)
    hp_da, ht_da = compute_polarization(
        vel_pos_da,
        displacement_frames=int(fps),
        return_angle=True,
    )

    fi = int(fps)
    disp = vel_pos_all[fi:] - vel_pos_all[:-fi]
    disp_mag = np.sqrt(np.nansum(disp**2, axis=1))
    min_disp = np.nanmin(disp_mag, axis=1)
    min_disp_full = np.full(total_frames, np.nan)
    min_disp_full[fi:] = min_disp

    return fps, hp_da.values, ht_da.values, min_disp_full


def _extract_orientation_coords(
    needed, total_frames, total_animals, is_valid, from_pos, to_pos
):
    """Extract head/tail coordinates for the needed frames."""
    head_x = np.full((total_frames, total_animals), np.nan)
    head_y = np.full((total_frames, total_animals), np.nan)
    tail_x = np.full((total_frames, total_animals), np.nan)
    tail_y = np.full((total_frames, total_animals), np.nan)
    for f in needed:
        if f < 0 or f >= total_frames:
            continue
        mask = is_valid[f]
        tail_x[f, mask] = from_pos[f, 0, mask]
        tail_y[f, mask] = from_pos[f, 1, mask]
        head_x[f, mask] = to_pos[f, 0, mask]
        head_y[f, mask] = to_pos[f, 1, mask]
    return head_x, head_y, tail_x, tail_y


def _extract_velocity_coords(
    needed, total_frames, total_animals, is_pos_valid, vel_pos_all
):
    """Extract velocity node coordinates for the needed frames."""
    vel_x = np.full((total_frames, total_animals), np.nan)
    vel_y = np.full((total_frames, total_animals), np.nan)
    for f in needed:
        if f < 0 or f >= total_frames:
            continue
        mask = is_pos_valid[f]
        vel_x[f, mask] = vel_pos_all[f, 0, mask]
        vel_y[f, mask] = vel_pos_all[f, 1, mask]
    return vel_x, vel_y


def _save_figure(fig, figures_dir, base_name, fps):
    """Save the figure in configured formats."""
    if SAVE_PNG:
        png_file = figures_dir / (base_name + ".png")
        fig.savefig(
            png_file,
            dpi=900,
            facecolor=DARK_BG,
            edgecolor="none",
            bbox_inches="tight",
            pad_inches=0.02,
        )
        print(f"\nSaved PNG: {png_file}")

    if SAVE_SVG:
        svg_file = figures_dir / (base_name + ".svg")
        fig.savefig(
            svg_file,
            format="svg",
            facecolor=DARK_BG,
            edgecolor="none",
            bbox_inches="tight",
            pad_inches=0,
        )
        print(f"Saved SVG: {svg_file}")


# ── Main entry point helpers ─────────────────────────────────────────────────


def _load_h5_config_interactive():
    """Attempt to load config from H5 file with user confirmation."""
    print("\n")
    print("AP VALIDATION H5 DETECTED")
    h5_config = load_suggested_pairs_from_h5()
    if h5_config is None:
        return None, False

    print()
    if _prompt_yn("Use suggested node pairs from H5 file? [y/n]: "):
        return h5_config, True
    print("Skipping H5 pairs — proceeding to interactive node selection.")
    return None, False


def _run_batch_mode(batch_config, ap_validated):
    """Run batch mode processing for all configured datasets."""
    print("\n")
    print("BATCH MODE")
    print(f"\nProcessing {len(batch_config)} dataset(s)...")

    for dataset_idx, (from_node, to_node) in batch_config.items():
        print(f"\n{'─' * 60}")
        print(f"Dataset [{dataset_idx}]")
        print(f"{'─' * 60}")

        node_config = {
            "velocity_node": from_node,
            "orientation_from": from_node,
            "orientation_to": to_node,
            "ap_validated": ap_validated,
            "auto_log": None,
        }
        process_dataset(dataset_idx, node_config=node_config)

    print("\n")
    print(f"BATCH COMPLETED ({len(batch_config)} datasets)")


def _run_interactive_mode():
    """Run interactive mode: prompt user, process single dataset."""
    dataset_idx = prompt_dataset_selection()
    process_dataset(dataset_idx)
    print("\n")
    print("COMPLETED")


def main():
    """Run the program from the command line.

    This entry point determines whether to run in batch mode or to load
    suggested node pairs interactively from the AP validation H5 file.
    It iterates through each configured dataset, computes polarization
    measures and visualizations, and writes out figures and logs.
    """
    ensure_demo_datasets()

    output_dir = ROOT_PATH / "exports" / "polarization-demos"
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    console_log = logs_dir / f"polarization_viz_{timestamp}.log"

    with TeeOutput(console_log):
        print(f"Polarization Viz started at {datetime.now().isoformat()}")
        print(f"Console log: {console_log}")
        print()

        batch_config = BATCH_CONFIG
        ap_validated = (
            BATCH_CONFIG_AP_VALIDATED if BATCH_CONFIG is not None else False
        )

        if batch_config is None and USE_AP_VALIDATION_H5:
            batch_config, ap_validated = _load_h5_config_interactive()

        if batch_config is not None:
            _run_batch_mode(batch_config, ap_validated)
        else:
            _run_interactive_mode()

        print(f"\nConsole log saved to: {console_log}")


if __name__ == "__main__":
    main()
