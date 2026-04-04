#!/usr/bin/env python3
"""AP inference demo script: PC1-based body-axis inference vs hand-curated GT.

Tests the prior-free AP inference pipeline against manually curated ground
truth rankings across 5 multi-animal SLEAP datasets. Operates in three
passes:
  Pass 1 — R×M Selection: find best individual per file
  Pass 2 — Cross-Individual Ordering Consistency: compare raw PC1-based
           orderings against best individual's ordering (pseudo GT)
  Pass 3 — Inferred AP Concordance: for each
           individual, compare the velocity-inferred AP ordering
           (anterior_sign × PC1) of GT nodes against hand-curated GT

After the passes, all GT pair permutations × all individuals are run
through validate_ap and stored in HDF5. analyze_results then reads back
the H5 to produce GT coverage analysis, suggested pair analysis, and
the data for Figure 2.

Generates two types of figures:
  Figure 1 — Per-file detail (2×2 tile per best individual)
  Figure 2 — Cross-dataset comparison (skeletons + coverage + ordering)

Parallelizes at pair×individual level for max throughput.
"""

import itertools
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

# Configuration
ROOT_PATH = Path(__file__).parent / "datasets" / "multi-animal"
SLP_DIR = ROOT_PATH / "slp"
MP4_DIR = ROOT_PATH / "mp4"
OUTPUT_DIR = ROOT_PATH / "exports" / "AP-inference-demo"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"
H5_DIR = OUTPUT_DIR / "h5"
N_WORKERS = mp.cpu_count()

# Hand-curated ground truth AP rankings (not exhaustive)
# Format: {file_stem: {node_index: rank}}
# Convention: higher rank = more anterior (rank 1 = most posterior in subset)
GROUND_TRUTH = {
    # Higher rank = more anterior
    "free-moving-2flies-ID-13nodes-1024x1024x1-30_3pxmm": {
        0: 3,  # head (most anterior, rank 3)
        1: 2,  # thorax
        2: 1,  # abdomen (most posterior, rank 1)
    },
    "free-moving-2mice-noID-5nodes-1280x1024x1-1_9pxmm": {
        0: 2,  # snout (anterior, rank 2)
        3: 1,  # tail-base (posterior, rank 1)
    },
    "free-moving-4gerbils-ID-14nodes-1024x1280x3-2pxmm": {
        0: 6,  # nose (most anterior, rank 6)
        5: 5,  # spine1
        6: 4,  # spine2
        7: 3,  # spine3
        8: 2,  # spine4
        9: 1,  # spine5 (most posterior, rank 1)
    },
    "free-moving-5mice-noID-11nodes-1280x1024x1-1_97pxmm": {
        0: 3,  # nose (most anterior, rank 3)
        1: 2,  # neck
        6: 1,  # tail_base (most posterior, rank 1)
    },
    "freemoving-2bees-noID-21nodes-1535x2048x1-14pxmm": {
        1: 3,  # head (most anterior, rank 3)
        0: 2,  # thorax
        2: 1,  # abdomen (most posterior, rank 1)
    },
}

# Display labels for files
FILE_LABELS = {
    "free-moving-2flies-ID-13nodes-1024x1024x1-30_3pxmm": "2Flies.slp",
    "free-moving-2mice-noID-5nodes-1280x1024x1-1_9pxmm": "2Mice.slp",
    "free-moving-4gerbils-ID-14nodes-1024x1280x3-2pxmm": "4Gerbils.slp",
    "free-moving-5mice-noID-11nodes-1280x1024x1-1_97pxmm": "5Mice.slp",
    "freemoving-2bees-noID-21nodes-1535x2048x1-14pxmm": "2Bees.slp",
}

DEMO_DATASETS = {
    "free-moving-2flies-ID-13nodes-1024x1024x1-30_3pxmm": {
        "mp4": "https://storage.googleapis.com/sleap-data/datasets/wt_gold.13pt/clips/talk_title_slide%4013150-14500.mp4",
        "slp": "https://storage.googleapis.com/sleap-data/datasets/wt_gold.13pt/clips/talk_title_slide%4013150-14500.slp",
    },
    "free-moving-5mice-noID-11nodes-1280x1024x1-1_97pxmm": {
        "mp4": "https://storage.googleapis.com/sleap-data/datasets/wang_4mice_john/clips/OFTsocial5mice-0000-00%4015488-18736.mp4",
        "slp": "https://storage.googleapis.com/sleap-data/datasets/wang_4mice_john/clips/OFTsocial5mice-0000-00%4015488-18736.slp",
    },
    "free-moving-2mice-noID-5nodes-1280x1024x1-1_9pxmm": {
        "mp4": "https://storage.googleapis.com/sleap-data/datasets/eleni_mice/clips/20200111_USVpairs_court1_M1_F1_top-01112020145828-0000%400-2560.mp4",
        "slp": "https://storage.googleapis.com/sleap-data/datasets/eleni_mice/clips/20200111_USVpairs_court1_M1_F1_top-01112020145828-0000%400-2560.slp",
    },
    "freemoving-2bees-noID-21nodes-1535x2048x1-14pxmm": {
        "mp4": "https://storage.googleapis.com/sleap-data/datasets/yan_bees/clips/bees_demo%4021000-23000.mp4",
        "slp": "https://storage.googleapis.com/sleap-data/datasets/yan_bees/clips/bees_demo%4021000-23000.slp",
    },
    "free-moving-4gerbils-ID-14nodes-1024x1280x3-2pxmm": {
        "mp4": "https://storage.googleapis.com/sleap-data/datasets/nyu-gerbils/clips/2020-3-10_daytime_5mins_compressedTalmo%403200-5760.mp4",
        "slp": "https://storage.googleapis.com/sleap-data/datasets/nyu-gerbils/clips/2020-3-10_daytime_5mins_compressedTalmo%403200-5760.slp",
    },
}


class TeeOutput:
    """Context manager that duplicates stdout to both console and a file."""

    def __init__(self, filepath):
        """Initialize with the target file path."""
        self.filepath = Path(filepath)
        self.file = None
        self.original_stdout = None

    def __enter__(self):
        """Open the file and redirect stdout."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.filepath, "w")
        self.original_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore stdout and close the file."""
        sys.stdout = self.original_stdout
        if self.file:
            self.file.close()
        return False

    def write(self, text):
        """Write text to both stdout and file."""
        self.original_stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        """Flush both stdout and file."""
        self.original_stdout.flush()
        self.file.flush()


def _download_file(url, destination):
    """Download a single file to the requested destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {destination.name}...")
    urlretrieve(url, destination)


def ensure_demo_datasets():
    """Ensure the expected demo SLP/MP4 files exist locally."""
    ROOT_PATH.mkdir(parents=True, exist_ok=True)
    SLP_DIR.mkdir(parents=True, exist_ok=True)
    MP4_DIR.mkdir(parents=True, exist_ok=True)

    slp_files = sorted(SLP_DIR.glob("*.slp"))
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


def process_single_validation(args):  # noqa: C901
    """Process a single (file, individual, from_kp, to_kp) validation."""
    # Import inside worker to avoid pickling issues
    from movement.io import load_poses
    from movement.kinematics.body_axis import ValidateAPConfig, validate_ap

    (
        slp_path,
        file_stem,
        individual,
        ind_idx,
        from_kp,
        to_kp,
        from_idx,
        to_idx,
        n_kp,
        n_ind,
        n_frames,
    ) = args

    # Suppress stdout using context manager
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull

        try:
            ds = load_poses.from_sleap_file(Path(slp_path))
            if "individuals" in ds.position.dims:
                pos_data = ds.position.sel(individuals=individual)
            else:
                pos_data = ds.position

            config = ValidateAPConfig()
            val = validate_ap(
                pos_data,
                from_node=from_kp,
                to_node=to_kp,
                config=config,
                verbose=False,
            )

            rec = {
                "file": file_stem,
                "individual": str(individual),
                "individual_idx": ind_idx,
                "from_keypoint": from_kp,
                "to_keypoint": to_kp,
                "from_index": from_idx,
                "to_index": to_idx,
                "n_keypoints": n_kp,
                "n_individuals": n_ind,
                "n_frames": n_frames,
                "validation_success": val.get("success", False),
                "resultant_length": val.get("resultant_length", np.nan),
                "vote_margin": val.get("vote_margin", np.nan),
                "num_selected_frames": val.get("num_selected_frames", 0),
                "circ_mean_dir": val.get("circ_mean_dir", np.nan),
                "anterior_sign": val.get("anterior_sign", 0),
                "num_clusters": val.get("num_clusters", 0),
                "error_msg": val.get("error_msg", ""),
                "error": False,
                "error_type": "",
            }

            r, m = rec["resultant_length"], rec["vote_margin"]
            rec["rxm"] = r * m if not (np.isnan(r) or np.isnan(m)) else np.nan

            pr = val.get("pair_report")
            if pr:
                rec.update(
                    {
                        "pr_success": pr.success,
                        "pr_failure_step": pr.failure_step,
                        "pr_failure_reason": pr.failure_reason,
                        "pr_scenario": pr.scenario,
                        "pr_outcome": pr.outcome,
                        "pr_warning_message": pr.warning_message,
                        "pr_input_pair_in_candidates": (
                            pr.input_pair_in_candidates
                        ),
                        "pr_input_pair_opposite_sides": (
                            pr.input_pair_opposite_sides
                        ),
                        "pr_input_pair_separation_abs": (
                            pr.input_pair_separation_abs
                        ),
                        "pr_input_pair_is_distal": pr.input_pair_is_distal,
                        "pr_input_pair_rank": pr.input_pair_rank,
                        "pr_input_pair_order_matches_inference": (
                            pr.input_pair_order_matches_inference
                        ),
                        "pr_max_separation_distal": pr.max_separation_distal,
                        "pr_max_separation": pr.max_separation,
                        "pr_lateral_offset_min": pr.lateral_offset_min,
                        "pr_lateral_offset_max": pr.lateral_offset_max,
                        "pr_midpoint_pc1": pr.midpoint_pc1,
                        "pr_pc1_min": pr.pc1_min,
                        "pr_pc1_max": pr.pc1_max,
                        "pr_midline_dist_max": pr.midline_dist_max,
                        # Cascade counts (same for all records
                        # from the same individual)
                        "n_valid_nodes": int(
                            np.sum(~np.isnan(pr.lateral_offsets_norm))
                        )
                        if len(pr.lateral_offsets_norm) > 0
                        else 0,
                        "n_step1_candidates": len(pr.sorted_candidate_nodes),
                        "n_step2_pairs": len(pr.valid_pairs),
                        "n_step3_distal": len(pr.distal_pairs),
                        "n_step3_proximal": len(pr.proximal_pairs),
                    }
                )

                if len(pr.max_separation_distal_nodes) > 0:
                    distal = pr.max_separation_distal_nodes
                    rec["suggested_from_idx"] = int(distal[0])
                    rec["suggested_to_idx"] = int(distal[1])
                    rec["suggested_type"] = "distal"
                elif len(pr.max_separation_nodes) > 0:
                    prox = pr.max_separation_nodes
                    rec["suggested_from_idx"] = int(prox[0])
                    rec["suggested_to_idx"] = int(prox[1])
                    rec["suggested_type"] = "proximal"
                else:
                    rec["suggested_from_idx"] = -1
                    rec["suggested_to_idx"] = -1
                    rec["suggested_type"] = ""

            # Store avg_skeleton and PC1 from validation result
            avg_skel = val.get("avg_skeleton")
            pc1_vec = val.get("PC1")
            if avg_skel is not None and not np.all(np.isnan(avg_skel)):
                rec["avg_skeleton"] = avg_skel.tolist()  # (n_keypoints, 2)
            else:
                rec["avg_skeleton"] = None
            if pc1_vec is not None:
                rec["PC1"] = pc1_vec.tolist()  # (2,)
            else:
                rec["PC1"] = None

            # Store velocity projections for histogram
            vel_projs = val.get("vel_projs_pc1")
            if vel_projs is not None and len(vel_projs) > 0:
                rec["vel_projs_pc1"] = vel_projs.tolist()
            else:
                rec["vel_projs_pc1"] = None

            return rec

        except Exception as e:
            import traceback

            print(
                f"WARNING: validate_ap failed for "
                f"{file_stem} / {individual} "
                f"({from_kp} → {to_kp}): "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            return {
                "file": file_stem,
                "individual": str(individual),
                "individual_idx": ind_idx,
                "from_keypoint": from_kp,
                "to_keypoint": to_kp,
                "from_index": from_idx,
                "to_index": to_idx,
                "n_keypoints": n_kp,
                "n_individuals": n_ind,
                "n_frames": n_frames,
                "error": True,
                "error_type": f"{type(e).__name__}: {e}",
                "validation_success": False,
                "rxm": np.nan,
                "resultant_length": np.nan,
                "vote_margin": np.nan,
                "anterior_sign": 0,
            }
        finally:
            sys.stdout = old_stdout


def generate_rxm_tasks(slp_files):
    """Pass 1: generate one task per (file, individual) to compute R×M.

    Uses first two GT node indices as an arbitrary pair since R×M depends
    only on the individual's motion and body shape, not the input pair.
    Returns tasks and metadata for subsequent passes.
    """
    from movement.io import load_poses

    tasks = []
    file_metadata = {}

    for slp_file in slp_files:
        ds = load_poses.from_sleap_file(slp_file)
        keypoints = [str(k) for k in ds.coords["keypoints"].values]
        individuals = [str(i) for i in ds.coords["individuals"].values]
        n_frames = ds.sizes["time"]
        n_kp = len(keypoints)
        has_individuals = "individuals" in ds.position.dims
        n_ind = len(individuals) if has_individuals else 1

        # Get GT node indices for this file (use first two as arbitrary pair)
        gt_nodes = GROUND_TRUTH.get(slp_file.stem, {})
        gt_indices = list(gt_nodes.keys())
        if len(gt_indices) < 2:
            continue
        from_idx, to_idx = gt_indices[0], gt_indices[1]

        file_metadata[slp_file.stem] = {
            "slp_file": slp_file,
            "keypoints": keypoints,
            "individuals": individuals,
            "n_frames": n_frames,
            "n_kp": n_kp,
            "n_ind": n_ind,
            "has_individuals": has_individuals,
            "gt_indices": gt_indices,
        }

        # One task per individual
        for ind_idx in range(n_ind):
            individual = individuals[ind_idx] if has_individuals else "single"
            tasks.append(
                (
                    str(slp_file),
                    slp_file.stem,
                    individual,
                    ind_idx,
                    keypoints[from_idx],
                    keypoints[to_idx],
                    from_idx,
                    to_idx,
                    n_kp,
                    n_ind,
                    n_frames,
                )
            )

    return tasks, file_metadata


def find_best_individuals(rxm_results):
    """Pass 1: Select best individual per file by maximum R×M.

    Returns:
        best_individuals: {file_stem: individual_name}
        all_rxm: {file_stem: {individual: rxm_value}} -
            R×M values for all individuals
        file_individual_data: {file_stem: {individual:
            {"avg_skeleton": ..., "pc1": ..., "anterior_sign": ...}}} -
            per-individual skeleton, PC1, and anterior_sign for downstream use

    """
    file_individual_rxm = defaultdict(dict)
    file_individual_data = defaultdict(dict)

    for rec in rxm_results:
        file_stem = rec["file"]
        individual = rec["individual"]
        rxm = rec["rxm"]
        if not np.isnan(rxm):
            file_individual_rxm[file_stem][individual] = rxm
            file_individual_data[file_stem][individual] = {
                "avg_skeleton": rec.get("avg_skeleton"),
                "pc1": rec.get("PC1"),
                "anterior_sign": rec.get("anterior_sign", 0),
            }

    best_individuals = {}
    for file_stem, individuals in file_individual_rxm.items():
        if not individuals:
            continue
        best_individuals[file_stem] = max(individuals, key=individuals.get)

    all_rxm = dict(file_individual_rxm)
    return best_individuals, all_rxm, dict(file_individual_data)


def compute_pc1_orderings(
    file_individual_data, file_metadata, best_individuals
):
    """Project GT nodes onto each individual's PC1 and rank by projection.

    For each individual, GT nodes are projected onto that individual's
    PC1 vector. Nodes are ranked by descending PC1 projection
    (rank 1 = highest projection; whether this corresponds to
    anterior or posterior depends on the individual's anterior_sign).

    Returns:
        best_pc1_orderings: {file_stem: {node_idx: pc1_rank}} -
            best individual's ordering (used as pseudo GT in Pass 2)
        all_pc1_orderings: {file_stem: {individual: {node_idx: pc1_rank}}} -
            all individuals' PC1-based orderings (used in Pass 2)

    """
    best_pc1_orderings = {}
    all_pc1_orderings = {}

    for file_stem, ind_data in file_individual_data.items():
        if file_stem not in file_metadata:
            continue

        gt_indices = file_metadata[file_stem]["gt_indices"]
        best_ind = best_individuals.get(file_stem)
        all_pc1_orderings[file_stem] = {}

        for individual, data in ind_data.items():
            avg_skeleton = data.get("avg_skeleton")
            pc1 = data.get("pc1")

            if avg_skeleton is None or pc1 is None:
                continue

            avg_skeleton = np.array(avg_skeleton)
            pc1 = np.array(pc1)

            # Project GT nodes onto raw PC1 (no anterior_sign correction).
            # Rank 1 = highest raw PC1 projection; whether this corresponds
            # to anterior or posterior depends on anterior_sign.
            gt_projections = {}
            for node_idx in gt_indices:
                if node_idx < len(avg_skeleton):
                    pos = avg_skeleton[node_idx]
                    if not np.any(np.isnan(pos)):
                        proj = np.dot(pos, pc1)
                        gt_projections[node_idx] = proj

            # Rank by raw PC1 projection (highest projection = rank 1)
            sorted_nodes = sorted(
                gt_projections.keys(),
                key=lambda x: gt_projections[x],
                reverse=True,
            )
            ind_ordering = {
                node: rank + 1 for rank, node in enumerate(sorted_nodes)
            }
            all_pc1_orderings[file_stem][individual] = ind_ordering

            # Store best individual's ordering separately
            if individual == best_ind:
                best_pc1_orderings[file_stem] = ind_ordering

    return best_pc1_orderings, all_pc1_orderings


def compute_inferred_ap_concordance(file_individual_data):
    """Pass 3: Compare each individual's inferred AP ordering against GT.

    For each individual, GT nodes are projected onto the inferred AP axis
    (anterior_sign × PC1, where positive = more anterior). All C(n,2)
    unique pairs of GT nodes are compared pairwise: a pair is concordant
    if the node with the higher inferred AP coordinate also has the
    higher hand-curated GT rank (higher rank = more anterior).

    Returns:
        individual_accuracy: {file_stem: {individual:
            {"correct": n, "total": n, "accuracy": pct}}}

    """
    individual_accuracy = {}

    for file_stem, gt_ranks in GROUND_TRUTH.items():
        individual_accuracy[file_stem] = {}
        ind_data = file_individual_data.get(file_stem, {})

        for individual, data in ind_data.items():
            avg_skeleton = data.get("avg_skeleton")
            pc1 = data.get("pc1")
            anterior_sign = data.get("anterior_sign", 0)

            if avg_skeleton is None or pc1 is None or anterior_sign == 0:
                continue

            avg_skeleton = np.array(avg_skeleton)
            pc1 = np.array(pc1)

            # Inferred AP unit vector: anterior_sign × PC1
            pc1_norm = pc1 / np.linalg.norm(pc1)
            e_ap = anterior_sign * pc1_norm

            # Project GT nodes onto inferred AP axis
            gt_ap_coords = {}
            for node_idx in gt_ranks:
                if node_idx < len(avg_skeleton):
                    pos = avg_skeleton[node_idx]
                    if not np.any(np.isnan(pos)):
                        gt_ap_coords[node_idx] = np.dot(pos, e_ap)

            # Compare pairwise ordering against GT
            correct = 0
            total = 0
            for a, b in itertools.combinations(gt_ap_coords.keys(), 2):
                total += 1
                ap_a, ap_b = gt_ap_coords[a], gt_ap_coords[b]
                gt_a, gt_b = gt_ranks[a], gt_ranks[b]

                # Concordant if relative ordering agrees:
                # higher AP coord = more anterior should match
                # higher GT rank = more anterior
                if (ap_a > ap_b and gt_a > gt_b) or (
                    ap_a < ap_b and gt_a < gt_b
                ):
                    correct += 1

            accuracy = 100 * correct / total if total > 0 else 0
            individual_accuracy[file_stem][individual] = {
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
            }

    return individual_accuracy


def compare_orderings_to_pseudo_gt(all_pc1_orderings, pseudo_gt_orderings):
    """Pass 2: Compare each individual's PC1-based ordering against pseudo GT.

    The pseudo GT is the best individual's PC1-based ordering. Comparison
    is strict list equality: the sorted node-index sequences must be
    identical. A single rank swap between any two nodes counts as a
    mismatch.

    Returns:
        ordering_matches: {file_stem: {individual: bool}} -
            True if individual's PC1-based ordering matches pseudo GT

    """
    ordering_matches = {}

    for file_stem, best_ordering in pseudo_gt_orderings.items():
        ordering_matches[file_stem] = {}
        ind_orderings = all_pc1_orderings.get(file_stem, {})

        for individual, ind_ordering in ind_orderings.items():
            # Compare orderings - match if relative order of all nodes same
            # Convert to sorted list of node indices by rank to compare
            best_sorted = sorted(
                best_ordering.keys(), key=lambda x: best_ordering[x]
            )
            ind_sorted = sorted(
                ind_ordering.keys(), key=lambda x: ind_ordering[x]
            )

            matches = best_sorted == ind_sorted
            ordering_matches[file_stem][individual] = matches

    return ordering_matches


def generate_gt_validation_tasks(file_metadata, best_individuals):
    """Generate H5 tasks for all GT node pair permutations.

    Tests all permutations of GT node pairs on ALL individuals.
    Results compared against hand-curated GROUND_TRUTH (higher rank = more
    anterior).

    Convention: (from_idx, to_idx) is "correctly ordered" if from is
    posterior (lower GT rank) and to is anterior (higher GT rank), matching
    the posterior→anterior direction used by compute_polarization's
    body_axis_keypoints.
    """
    tasks = []
    keypoints_by_file = {}

    for file_stem, meta in file_metadata.items():
        if file_stem not in best_individuals:
            continue

        keypoints = meta["keypoints"]
        individuals = meta["individuals"]
        gt_indices = meta["gt_indices"]
        slp_file = meta["slp_file"]

        keypoints_by_file[file_stem] = keypoints

        # Test all ordered GT pairs (permutations) on ALL individuals
        for from_idx, to_idx in itertools.permutations(gt_indices, 2):
            for ind_idx in range(meta["n_ind"]):
                has_ind = meta["has_individuals"]
                individual = individuals[ind_idx] if has_ind else "single"
                tasks.append(
                    (
                        str(slp_file),
                        file_stem,
                        individual,
                        ind_idx,
                        keypoints[from_idx],
                        keypoints[to_idx],
                        from_idx,
                        to_idx,
                        meta["n_kp"],
                        meta["n_ind"],
                        meta["n_frames"],
                    )
                )

    return tasks, keypoints_by_file


def save_to_h5(  # noqa: C901
    all_results,
    keypoints_by_file,
    output_path,
    individual_accuracy=None,
    all_rxm=None,
):
    """Save results to HDF5."""
    n = len(all_results)
    dt_str = h5py.special_dtype(vlen=str)

    str_fields = [
        "file",
        "individual",
        "from_keypoint",
        "to_keypoint",
        "error_msg",
        "error_type",
        "pr_failure_step",
        "pr_failure_reason",
        "pr_outcome",
        "pr_warning_message",
        "suggested_type",
    ]
    int_fields = [
        "individual_idx",
        "from_index",
        "to_index",
        "n_keypoints",
        "n_individuals",
        "n_frames",
        "num_selected_frames",
        "anterior_sign",
        "num_clusters",
        "pr_scenario",
        "pr_input_pair_rank",
        "suggested_from_idx",
        "suggested_to_idx",
        "n_valid_nodes",
        "n_step1_candidates",
        "n_step2_pairs",
        "n_step3_distal",
        "n_step3_proximal",
    ]
    float_fields = [
        "resultant_length",
        "vote_margin",
        "rxm",
        "circ_mean_dir",
        "pr_input_pair_separation_abs",
        "pr_max_separation_distal",
        "pr_max_separation",
        "pr_lateral_offset_min",
        "pr_lateral_offset_max",
        "pr_midpoint_pc1",
        "pr_pc1_min",
        "pr_pc1_max",
        "pr_midline_dist_max",
    ]
    bool_fields = [
        "validation_success",
        "error",
        "pr_success",
        "pr_input_pair_in_candidates",
        "pr_input_pair_opposite_sides",
        "pr_input_pair_is_distal",
        "pr_input_pair_order_matches_inference",
    ]

    with h5py.File(output_path, "w") as f:
        f.attrs["created"] = datetime.now().isoformat()
        f.attrs["n_files"] = len(keypoints_by_file)
        f.attrs["n_records"] = n

        kp_grp = f.create_group("keypoints")
        for fname, kps in keypoints_by_file.items():
            kp_grp.create_dataset(fname, data=np.array(kps, dtype="S"))

        for field in str_fields:
            f.create_dataset(
                field,
                data=np.array(
                    [r.get(field, "") for r in all_results], dtype=dt_str
                ),
            )

        for field in int_fields:
            f.create_dataset(
                field,
                data=np.array(
                    [r.get(field, -1) for r in all_results], dtype=np.int32
                ),
            )

        for field in float_fields:
            f.create_dataset(
                field,
                data=np.array(
                    [r.get(field, np.nan) for r in all_results],
                    dtype=np.float64,
                ),
            )

        for field in bool_fields:
            f.create_dataset(
                field,
                data=np.array(
                    [r.get(field, False) for r in all_results], dtype=bool
                ),
            )

        # Save avg_skeleton, PC1, and vel_projs arrays
        # Group by (file, individual) - these are per-individual, not per-pair
        skel_grp = f.create_group("skeletons")
        pc1_grp = f.create_group("pc1_vectors")
        velprojs_grp = f.create_group("vel_projs_pc1")

        # First pass: collect best data for each (file, individual) pair
        # Need to search all records to find valid data (not just first)
        # {(file, individual): {"skeleton": ..., "pc1": ..., "vel_projs": ...}}
        pair_data = {}
        for r in all_results:
            file_stem = r.get("file", "")
            individual = r.get("individual", "")
            if not file_stem or not individual:
                continue

            key = (file_stem, individual)
            if key not in pair_data:
                pair_data[key] = {
                    "skeleton": None,
                    "pc1": None,
                    "vel_projs": None,
                }

            # Update with valid data if current is None
            pd = pair_data[key]
            if pd["skeleton"] is None and r.get("avg_skeleton") is not None:
                pd["skeleton"] = r.get("avg_skeleton")
            if pd["pc1"] is None and r.get("PC1") is not None:
                pd["pc1"] = r.get("PC1")
            if pd["vel_projs"] is None and r.get("vel_projs_pc1") is not None:
                pd["vel_projs"] = r.get("vel_projs_pc1")

        # Second pass: save collected data to H5
        for (file_stem, individual), data in pair_data.items():
            # Create file group if needed
            if file_stem not in skel_grp:
                skel_grp.create_group(file_stem)
                pc1_grp.create_group(file_stem)
                velprojs_grp.create_group(file_stem)

            if data["skeleton"] is not None:
                arr = np.array(data["skeleton"], dtype=np.float64)
                skel_grp[file_stem].create_dataset(individual, data=arr)
            if data["pc1"] is not None:
                arr = np.array(data["pc1"], dtype=np.float64)
                pc1_grp[file_stem].create_dataset(individual, data=arr)
            if data["vel_projs"] is not None:
                arr = np.array(data["vel_projs"], dtype=np.float64)
                velprojs_grp[file_stem].create_dataset(individual, data=arr)

        # Save individual accuracy (raw PC1 direction diag for Fig 1)
        if individual_accuracy:
            acc_grp = f.create_group("individual_accuracy")
            for file_stem, accuracies in individual_accuracy.items():
                file_grp = acc_grp.create_group(file_stem)
                for individual, acc in accuracies.items():
                    ind_grp = file_grp.create_group(individual)
                    ind_grp.create_dataset("correct", data=acc["correct"])
                    ind_grp.create_dataset("total", data=acc["total"])
                    ind_grp.create_dataset("accuracy", data=acc["accuracy"])

        # Save R×M values for all individuals
        if all_rxm:
            rxm_grp = f.create_group("individual_rxm")
            for file_stem, individuals in all_rxm.items():
                file_grp = rxm_grp.create_group(file_stem)
                for individual, rxm_val in individuals.items():
                    file_grp.create_dataset(individual, data=rxm_val)


# Analysis functions


def load_h5_data(h5_path):  # noqa: C901
    """Load relevant fields from H5 file."""

    def decode_str(x):
        return x.decode() if isinstance(x, bytes) else x

    data = {}
    with h5py.File(h5_path, "r") as f:
        # String fields
        data["file"] = [decode_str(x) for x in f["file"][:]]
        data["individual"] = [decode_str(x) for x in f["individual"][:]]
        data["suggested_type"] = [
            decode_str(x) for x in f["suggested_type"][:]
        ]

        # Index fields
        data["from_index"] = np.array(f["from_index"])
        data["to_index"] = np.array(f["to_index"])
        data["suggested_from_idx"] = np.array(f["suggested_from_idx"])
        data["suggested_to_idx"] = np.array(f["suggested_to_idx"])

        # Validation results
        data["validation_success"] = np.array(f["validation_success"])
        data["anterior_sign"] = np.array(f["anterior_sign"])
        data["pr_input_pair_order_matches_inference"] = np.array(
            f["pr_input_pair_order_matches_inference"]
        )
        data["pr_success"] = np.array(f["pr_success"])
        data["pr_input_pair_in_candidates"] = np.array(
            f["pr_input_pair_in_candidates"]
        )
        data["rxm"] = np.array(f["rxm"])
        data["vote_margin"] = np.array(f["vote_margin"])
        data["resultant_length"] = np.array(f["resultant_length"])
        n = len(data["file"])
        if "circ_mean_dir" in f:
            data["circ_mean_dir"] = np.array(f["circ_mean_dir"])
        else:
            data["circ_mean_dir"] = np.full(n, np.nan)
        if "num_selected_frames" in f:
            data["num_selected_frames"] = np.array(f["num_selected_frames"])
        else:
            data["num_selected_frames"] = np.zeros(n, dtype=np.int32)
        if "n_frames" in f:
            data["n_frames"] = np.array(f["n_frames"])
        else:
            data["n_frames"] = np.zeros(n, dtype=np.int32)

        # Cascade counts (backward-compatible)
        for cascade_field in [
            "n_valid_nodes",
            "n_step1_candidates",
            "n_step2_pairs",
            "n_step3_distal",
            "n_step3_proximal",
        ]:
            if cascade_field in f:
                data[cascade_field] = np.array(f[cascade_field])
            else:
                data[cascade_field] = np.zeros(n, dtype=np.int32)

        # Load skeleton and PC1 data (if present)
        data["skeletons"] = {}  # {file_stem: {individual: np.array}}
        data["pc1_vectors"] = {}  # {file_stem: {individual: np.array}}

        if "skeletons" in f:
            for file_stem in f["skeletons"]:
                data["skeletons"][file_stem] = {}
                for individual in f["skeletons"][file_stem]:
                    data["skeletons"][file_stem][individual] = np.array(
                        f["skeletons"][file_stem][individual]
                    )

        if "pc1_vectors" in f:
            for file_stem in f["pc1_vectors"]:
                data["pc1_vectors"][file_stem] = {}
                for individual in f["pc1_vectors"][file_stem]:
                    data["pc1_vectors"][file_stem][individual] = np.array(
                        f["pc1_vectors"][file_stem][individual]
                    )

        # Load velocity projections for histogram
        data["vel_projs_pc1"] = {}
        if "vel_projs_pc1" in f:
            for file_stem in f["vel_projs_pc1"]:
                data["vel_projs_pc1"][file_stem] = {}
                for individual in f["vel_projs_pc1"][file_stem]:
                    data["vel_projs_pc1"][file_stem][individual] = np.array(
                        f["vel_projs_pc1"][file_stem][individual]
                    )

        # Load keypoint names
        data["keypoints"] = {}
        if "keypoints" in f:
            for file_stem in f["keypoints"]:
                kp_data = f["keypoints"][file_stem][:]
                data["keypoints"][file_stem] = [
                    x.decode() if isinstance(x, bytes) else x for x in kp_data
                ]

        # Load individual accuracy data (raw PC1 direction diagnostic)
        data["individual_accuracy"] = {}
        if "individual_accuracy" in f:
            for file_stem in f["individual_accuracy"]:
                data["individual_accuracy"][file_stem] = {}
                for individual in f["individual_accuracy"][file_stem]:
                    ind_grp = f["individual_accuracy"][file_stem][individual]
                    data["individual_accuracy"][file_stem][individual] = {
                        "correct": int(ind_grp["correct"][()]),
                        "total": int(ind_grp["total"][()]),
                        "accuracy": float(ind_grp["accuracy"][()]),
                    }

        # Load R×M values for all individuals
        data["individual_rxm"] = {}
        if "individual_rxm" in f:
            for file_stem in f["individual_rxm"]:
                data["individual_rxm"][file_stem] = {}
                for individual in f["individual_rxm"][file_stem]:
                    data["individual_rxm"][file_stem][individual] = float(
                        f["individual_rxm"][file_stem][individual][()]
                    )

    return data


def find_best_individual_per_file(data):
    """Find the individual with highest mean R×M for each file."""
    file_individual_rxm = defaultdict(lambda: defaultdict(list))

    for i in range(len(data["file"])):
        file_stem = data["file"][i]
        individual = data["individual"][i]
        rxm = data["rxm"][i]
        if not np.isnan(rxm):
            file_individual_rxm[file_stem][individual].append(rxm)

    best_individual = {}
    for file_stem, individuals in file_individual_rxm.items():
        best_rxm = -1
        best_ind = None
        for ind, rxm_list in individuals.items():
            mean_rxm = np.mean(rxm_list)
            if mean_rxm > best_rxm:
                best_rxm = mean_rxm
                best_ind = ind
        best_individual[file_stem] = best_ind

    return best_individual


def find_step1_surviving_nodes(data, best_individual):
    """Find nodes surviving the lateral alignment filter (Step 1) per file.

    A node survives Step 1 if it appears in any pair where
    pr_input_pair_in_candidates=True for the file's best individual.
    """
    file_surviving_nodes = defaultdict(set)

    for i in range(len(data["file"])):
        file_stem = data["file"][i]
        individual = data["individual"][i]

        # Only consider best individual
        if individual != best_individual.get(file_stem):
            continue

        # If this pair survived Step 1, both nodes are candidates
        if data["pr_input_pair_in_candidates"][i]:
            from_idx = int(data["from_index"][i])
            to_idx = int(data["to_index"][i])
            file_surviving_nodes[file_stem].add(from_idx)
            file_surviving_nodes[file_stem].add(to_idx)

    return file_surviving_nodes


def compute_gt_coverage(file_surviving_nodes):
    """Compute lateral filter coverage: fraction of GT nodes surviving Step 1.

    Returns dict: {file_stem: {surviving_in_gt, gt_total, coverage_pct, ...}}
    """
    coverage = {}

    for file_stem, gt_ranks in GROUND_TRUTH.items():
        gt_nodes = set(gt_ranks.keys())
        n_gt_total = len(gt_nodes)

        surviving = file_surviving_nodes.get(file_stem, set())
        surviving_in_gt = surviving & gt_nodes
        n_surviving_in_gt = len(surviving_in_gt)

        if n_gt_total > 0:
            coverage_pct = 100 * n_surviving_in_gt / n_gt_total
        else:
            coverage_pct = 0

        coverage[file_stem] = {
            "surviving_in_gt": n_surviving_in_gt,
            "gt_total": n_gt_total,
            "coverage_pct": coverage_pct,
            "surviving_nodes": sorted(surviving),
            "gt_nodes": sorted(gt_nodes),
            "surviving_in_gt_nodes": sorted(surviving_in_gt),
            "gt_not_surviving": sorted(gt_nodes - surviving),
        }

    return coverage


def analyze_suggested_pairs(data, best_individual):  # noqa: C901
    """Report which node pair the 3-step filter cascade auto-selected.

    For each file's best individual, retrieves the suggested pair
    (posterior→anterior, distal vs proximal) and notes whether the
    selected nodes happen to fall within the hand-curated GT subset.
    GT membership is incidental context — the pipeline often selects
    nodes outside the GT subset, which is expected.
    """
    # Get suggested pair for each file (use first record for best individual)
    file_suggested = {}

    for i in range(len(data["file"])):
        file_stem = data["file"][i]
        individual = data["individual"][i]

        if individual != best_individual.get(file_stem):
            continue

        # Only record once per file (suggested pair same for all input pairs)
        if file_stem in file_suggested:
            continue

        suggested_from = int(data["suggested_from_idx"][i])
        suggested_to = int(data["suggested_to_idx"][i])
        suggested_type = data["suggested_type"][i]

        file_suggested[file_stem] = {
            "from_idx": suggested_from,
            "to_idx": suggested_to,
            "type": suggested_type,
        }

    # Analyze each file
    results = {}
    for file_stem, gt_ranks in GROUND_TRUTH.items():
        gt_nodes = set(gt_ranks.keys())
        default = {"from_idx": -1, "to_idx": -1, "type": ""}
        suggested = file_suggested.get(file_stem, default)

        from_idx = suggested["from_idx"]
        to_idx = suggested["to_idx"]
        stype = suggested["type"]

        # Check if suggested pair is valid
        if from_idx < 0 or to_idx < 0:
            results[file_stem] = {
                "suggested_from": from_idx,
                "suggested_to": to_idx,
                "suggested_type": stype,
                "from_in_gt": False,
                "to_in_gt": False,
                "both_in_gt": False,
                "order_correct": None,
                "status": "NO SUGGESTION",
            }
            continue

        from_in_gt = from_idx in gt_nodes
        to_in_gt = to_idx in gt_nodes
        both_in_gt = from_in_gt and to_in_gt

        # Check ordering if both are in GT
        order_correct = None
        if both_in_gt:
            from_rank = gt_ranks[from_idx]
            to_rank = gt_ranks[to_idx]
            # Correct: from=posterior (lower rank), to=anterior (higher)
            order_correct = from_rank < to_rank

        # Determine status (GT membership is incidental — the pipeline
        # often selects nodes outside the hand-curated GT subset)
        if both_in_gt and order_correct:
            status = "Both in GT, order correct"
        elif both_in_gt and not order_correct:
            status = "Both in GT, order reversed"
        elif from_in_gt or to_in_gt:
            status = "One node in GT"
        else:
            status = "Neither node in GT"

        results[file_stem] = {
            "suggested_from": from_idx,
            "suggested_to": to_idx,
            "suggested_type": stype,
            "from_in_gt": from_in_gt,
            "to_in_gt": to_in_gt,
            "both_in_gt": both_in_gt,
            "order_correct": order_correct,
            "from_rank": gt_ranks.get(from_idx, "NaN"),
            "to_rank": gt_ranks.get(to_idx, "NaN"),
            "status": status,
        }

    return results


def get_ground_truth_order(file_stem, from_idx, to_idx):
    """Determine hand-curated GT AP ordering for a node pair.

    Convention: (from, to) = (posterior, anterior). This matches
    compute_polarization's body_axis_keypoints convention where the vector
    points FROM posterior TO anterior.

    Returns:
        1 if from_idx is posterior to to_idx (correctly ordered)
        -1 if from_idx is anterior to to_idx (reversed)
        None if either index is not in the hand-curated ground truth

    """
    if file_stem not in GROUND_TRUTH:
        return None

    gt = GROUND_TRUTH[file_stem]
    if from_idx not in gt or to_idx not in gt:
        return None

    from_rank = gt[from_idx]
    to_rank = gt[to_idx]

    # Higher rank = more anterior, lower rank = more posterior
    # Correct: from=posterior (lower rank), to=anterior (higher rank)
    if from_rank < to_rank:
        return 1  # from is posterior, to is anterior (correct order)
    elif from_rank > to_rank:
        return -1  # from is anterior, to is posterior (incorrect order)
    else:
        return 0  # same rank (shouldn't happen with non-NaN unique ranks)


def analyze_results(h5_path):  # noqa: C901
    """Report filter cascade, suggested pairs, and Figure 2 data.

    Read the H5 file produced by the parallel validate_ap runs. For each
    file's best individual (highest mean R×M), log the 3-step filter
    cascade progression (with GT coverage folded into Step 1) and the
    suggested pair analysis. Return cascade stats, GT coverage, and
    suggested pair data for Figure 2 rendering.
    """
    print("\n" + "─" * 70)
    print("REPORTING: Filter Cascade & Suggested Pairs")
    print("─" * 70)

    data = load_h5_data(h5_path)
    n_records = len(data["file"])

    # Find best individual per file (already reported in Pass 1,
    # recomputed here from H5 using mean R×M across all records)
    best_individual = find_best_individual_per_file(data)

    # Find Step 1 surviving nodes and compute GT coverage
    file_surviving_nodes = find_step1_surviving_nodes(data, best_individual)
    gt_coverage = compute_gt_coverage(file_surviving_nodes)

    # Extract cascade progression stats for each file's best individual
    cascade_stats = {}
    seen_files = set()
    for i in range(n_records):
        file_stem = data["file"][i]
        individual = data["individual"][i]

        if individual != best_individual.get(file_stem):
            continue
        if file_stem in seen_files:
            continue
        seen_files.add(file_stem)

        n_valid = int(data["n_valid_nodes"][i])
        n_s1 = int(data["n_step1_candidates"][i])
        n_s2 = int(data["n_step2_pairs"][i])
        n_s3d = int(data["n_step3_distal"][i])
        n_s3p = int(data["n_step3_proximal"][i])

        n_candidate_pairs = n_s1 * (n_s1 - 1) // 2 if n_s1 >= 2 else 0

        cascade_stats[file_stem] = {
            "n_valid_nodes": n_valid,
            "n_step1_candidates": n_s1,
            "n_candidate_pairs": n_candidate_pairs,
            "n_step2_pairs": n_s2,
            "n_step3_distal": n_s3d,
            "n_step3_proximal": n_s3p,
        }

    # Log cascade progression with GT coverage folded into Step 1
    print("\n3-STEP FILTER CASCADE (best individual per file):")
    for file_stem in sorted(cascade_stats.keys()):
        cs = cascade_stats[file_stem]
        cov = gt_coverage.get(file_stem, {})
        label = FILE_LABELS.get(file_stem, file_stem[:15])
        n_valid = cs["n_valid_nodes"]
        n_s1 = cs["n_step1_candidates"]
        n_cp = cs["n_candidate_pairs"]
        n_s2 = cs["n_step2_pairs"]
        n_s3d = cs["n_step3_distal"]

        gt_surv = cov.get("surviving_in_gt", 0)
        gt_tot = cov.get("gt_total", 0)
        gt_str = f"  (GT: {gt_surv}/{gt_tot} nodes)" if gt_tot > 0 else ""

        print(f"\n  {label}:")
        print(f"    Step 1 (lateral):   {n_s1}/{n_valid} nodes{gt_str}")
        print(f"    Step 2 (opposite):  {n_s2}/{n_cp} candidate pairs")
        print(f"    Step 3 (distal):    {n_s3d}/{n_s2} pairs")

    # Analyze suggested pairs
    suggested_analysis = analyze_suggested_pairs(data, best_individual)

    print("\n" + "─" * 70)
    print("SUGGESTED AP PAIR: Auto-Selected Node Pair (3-Step Filter Cascade)")
    print("─" * 70)
    for file_stem in sorted(suggested_analysis.keys()):
        r = suggested_analysis[file_stem]
        label = FILE_LABELS.get(file_stem, file_stem[:15])
        print(f"\n{label} ({file_stem}):")
        gt_nodes = sorted(GROUND_TRUTH[file_stem].keys())
        print(f"  Hand-curated GT node indices: {gt_nodes}")
        sf, st = r["suggested_from"], r["suggested_to"]
        print(
            f"  Suggested pair: [{sf} → {st}] "
            f"(posterior → anterior, type: {r['suggested_type']})"
        )

        if r["suggested_from"] >= 0:
            print(
                f"  Posterior node {sf}: "
                f"in GT = {r['from_in_gt']}, "
                f"GT rank = {r['from_rank']}"
            )
            print(
                f"  Anterior node {st}: "
                f"in GT = {r['to_in_gt']}, "
                f"GT rank = {r['to_rank']}"
            )

            if r["both_in_gt"]:
                if r["order_correct"]:
                    order_str = (
                        "CORRECT (from_node is posterior, "
                        "to_node is anterior per GT)"
                    )
                else:
                    order_str = (
                        "REVERSED (from_node is anterior, "
                        "to_node is posterior per GT)"
                    )
                print(f"  Ordering: {order_str}")

        print(f"  Status: {r['status']}")
    print()

    return {
        "best_individual": best_individual,
        "gt_coverage": gt_coverage,
        "suggested_analysis": suggested_analysis,
        "cascade_stats": cascade_stats,
        "skeletons": data.get("skeletons", {}),
        "pc1_vectors": data.get("pc1_vectors", {}),
        "vel_projs_pc1": data.get("vel_projs_pc1", {}),
        "keypoints": data.get("keypoints", {}),
        # For individual plot generation
        "file": data["file"],
        "individual": data["individual"],
        "rxm": data["rxm"],
        "vote_margin": data["vote_margin"],
        "resultant_length": data["resultant_length"],
        "anterior_sign": data["anterior_sign"],
        "circ_mean_dir": data["circ_mean_dir"],
        "num_selected_frames": data["num_selected_frames"],
        "n_frames": data["n_frames"],
        # Per-individual diagnostic data: GT concordance and R×M values
        "individual_accuracy": data.get("individual_accuracy", {}),
        "individual_rxm": data.get("individual_rxm", {}),
    }


def plot_validation_results(analysis_data, output_path):  # noqa: C901
    """Create a tiled layout figure showing validation results.

    Parameters
    ----------
    analysis_data : dict
        Analysis results from analyze_results()
    output_path : Path
        Output path for saving figure

    """
    _gt_coverage = analysis_data["gt_coverage"]  # noqa: F841
    suggested_analysis = analysis_data["suggested_analysis"]
    cascade_stats = analysis_data["cascade_stats"]

    # Short labels for files (filename.slp format)
    file_labels = {
        "free-moving-2flies-ID-13nodes-1024x1024x1-30_3pxmm": "2Flies.slp",
        "free-moving-2mice-noID-5nodes-1280x1024x1-1_9pxmm": "2Mice.slp",
        "free-moving-4gerbils-ID-14nodes-1024x1280x3-2pxmm": "4Gerbils.slp",
        "free-moving-5mice-noID-11nodes-1280x1024x1-1_97pxmm": "5Mice.slp",
        "freemoving-2bees-noID-21nodes-1535x2048x1-14pxmm": "2Bees.slp",
    }

    files = sorted(GROUND_TRUTH.keys())
    n_files = len(files)

    bg_color = "white"
    text_color = "black"
    axis_color = "black"
    midline_color = "black"
    ap_arrow_color = "black"
    gt_node_color = "black"
    label_bg_alpha = 0.8

    fig = plt.figure(figsize=(14, 10), facecolor=bg_color)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.3, wspace=0.3
    )
    fig.suptitle(
        "AP Validation: Average Skeleton (Best Individual by R×M)",
        fontsize=14,
        fontweight="bold",
        color=text_color,
    )

    # Color palette
    colors = [
        (0.12, 0.47, 0.71, 1.0),  # Blue (2 Flies)
        (0.17, 0.63, 0.17, 1.0),  # Green (2 Mice)
        (0.80, 0.70, 0.10, 1.0),  # Gold (4 Gerbils)
        (1.00, 0.50, 0.05, 1.0),  # Orange (5 Mice)
        (0.58, 0.40, 0.74, 1.0),  # Purple (2 Bees)
    ]

    ax1 = fig.add_subplot(gs[0, :], facecolor=bg_color)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # Get skeletons and PC1 from stored H5 data
    best_individual = analysis_data["best_individual"]
    stored_skeletons = analysis_data.get("skeletons", {})
    stored_pc1 = analysis_data.get("pc1_vectors", {})
    skeleton_data = []

    for f in files:
        best_ind = best_individual.get(f)
        if best_ind is None:
            continue

        # Try to get stored skeleton from H5
        if f in stored_skeletons and best_ind in stored_skeletons[f]:
            avg_skel = stored_skeletons[f][best_ind]  # (n_keypoints, 2)
            x = avg_skel[:, 0]
            y = avg_skel[:, 1]
            pc1 = stored_pc1.get(f, {}).get(best_ind)  # (2,) or None
        else:
            # Fallback: load from .slp file
            from movement.io import load_poses

            slp_path = SLP_DIR / f"{f}.slp"
            if not slp_path.exists():
                continue

            ds = load_poses.from_sleap_file(slp_path)
            if "individuals" in ds.position.dims:
                pos = ds.position.sel(individuals=best_ind)
            else:
                pos = ds.position

            x = np.nanmean(pos.sel(space="x").values, axis=0)
            y = np.nanmean(pos.sel(space="y").values, axis=0)
            pc1 = None  # Will compute below if needed

        # Get suggested pair for this file
        default_sug = {
            "suggested_from": -1,
            "suggested_to": -1,
            "status": "NO SUGGESTION",
        }
        suggested = suggested_analysis.get(f, default_sug)

        skeleton_data.append(
            {
                "file": f,
                "label": file_labels.get(f, f[:10]),
                "best_ind": best_ind,  # Animal identity
                "x": x.copy(),
                "y": y.copy(),
                "pc1": pc1,  # Stored PC1 vector (or None)
                "gt": GROUND_TRUTH.get(f, {}),
                "suggested_from": suggested.get("suggested_from", -1),
                "suggested_to": suggested.get("suggested_to", -1),
                "from_in_gt": suggested.get("from_in_gt", False),
                "to_in_gt": suggested.get("to_in_gt", False),
                "status": suggested.get("status", "NO SUGGESTION"),
            }
        )

    # Process and plot skeletons side by side
    n_skeletons = len(skeleton_data)
    midline_plotted = False  # For legend

    if n_skeletons > 0:
        x_offset = 0
        spacing = 1.8  # Space between skeletons

        for idx, skel in enumerate(skeleton_data):
            x, y = skel["x"], skel["y"]
            gt = skel["gt"]
            stored_pc1 = skel.get("pc1")

            # Get valid (non-NaN) node indices
            valid_mask = ~np.isnan(x) & ~np.isnan(y)
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) < 2:
                continue

            # avg_skeleton from collective.py is already centered, use directly
            x_centered = x.copy()
            y_centered = y.copy()

            # Use stored PC1 from collective.py if available, else compute
            if stored_pc1 is not None:
                pc1 = np.array(stored_pc1)
            else:
                # Fallback: compute PC1 via PCA on valid nodes
                xv = x_centered[valid_mask]
                yv = y_centered[valid_mask]
                valid_coords = np.column_stack([xv, yv])
                cov_matrix = np.cov(valid_coords.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                pc1 = eigenvectors[:, np.argmax(eigenvalues)]

            # Ensure PC1 points toward the most anterior GT node
            # (highest rank = most anterior in GT convention)
            anterior_node = None
            max_rank = -1
            for node_idx, rank in gt.items():
                if rank > max_rank:
                    max_rank = rank
                    anterior_node = node_idx

            ant_ok = anterior_node is not None and not np.isnan(
                x_centered[anterior_node]
            )
            if ant_ok:
                ax, ay = x_centered[anterior_node], y_centered[anterior_node]
                ant_vec = np.array([ax, ay])
                if np.dot(pc1, ant_vec) < 0:
                    pc1 = -pc1  # Flip to point toward anterior

            # Normalize scale (max of x/y range for square proportions)
            x_range = np.nanmax(x_centered) - np.nanmin(x_centered)
            y_range = np.nanmax(y_centered) - np.nanmin(y_centered)
            max_range = max(x_range, y_range)
            if max_range > 0:
                scale = 1.5 / max_range  # Scale up for larger display
                x_centered *= scale
                y_centered *= scale

            # Offset horizontally for side-by-side placement
            x_plot = x_centered + x_offset
            y_plot = y_centered

            file_color = colors[idx]

            # Compute axis extent based on projections onto PC1
            xv, yv = x_centered[valid_mask], y_centered[valid_mask]
            proj_pc1 = xv * pc1[0] + yv * pc1[1]
            min_proj, max_proj = np.min(proj_pc1), np.max(proj_pc1)
            axis_extent = (max_proj - min_proj) * 0.7

            # Draw PC1 axis at actual angle (matching color)
            ae = axis_extent
            ax1.plot(
                [x_offset - ae * pc1[0], x_offset + ae * pc1[0]],
                [0 - ae * pc1[1], 0 + ae * pc1[1]],
                "-",
                color=(*file_color[:3], 0.5),
                linewidth=1.5,
                zorder=1,
            )

            # Draw AP midline (dashed black, perpendicular to PC1)
            pc1_perp = np.array([-pc1[1], pc1[0]])  # Perpendicular to PC1
            midline_extent = 0.2
            mid_y = (min_proj + max_proj) / 2  # Midpoint along PC1
            mid_x_pos = x_offset + mid_y * pc1[0]
            mid_y_pos = mid_y * pc1[1]

            me = midline_extent
            mx0 = mid_x_pos - me * pc1_perp[0]
            mx1 = mid_x_pos + me * pc1_perp[0]
            my0 = mid_y_pos - me * pc1_perp[1]
            my1 = mid_y_pos + me * pc1_perp[1]
            if not midline_plotted:
                ax1.plot(
                    [mx0, mx1],
                    [my0, my1],
                    "--",
                    color=midline_color,
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=2,
                    label="AP midline",
                )
                midline_plotted = True
            else:
                ax1.plot(
                    [mx0, mx1],
                    [my0, my1],
                    "--",
                    color=midline_color,
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=2,
                )

            # Draw AP bidirectional arrow next to midline
            ap_off = midline_extent * 1.1  # Offset from midline center
            ap_half = axis_extent * 0.125  # Half-length of arrow shaft

            # Arrow center (offset perpendicular to PC1 from midline center)
            acx = mid_x_pos + ap_off * pc1_perp[0]
            acy = mid_y_pos + ap_off * pc1_perp[1]

            # Arrow endpoints along PC1 direction
            arrow_ant_x = acx + ap_half * pc1[0]  # Anterior end (+PC1)
            arrow_ant_y = acy + ap_half * pc1[1]
            arrow_post_x = acx - ap_half * pc1[0]  # Posterior end (-PC1)
            arrow_post_y = acy - ap_half * pc1[1]

            # Draw arrow shaft
            ax1.plot(
                [arrow_post_x, arrow_ant_x],
                [arrow_post_y, arrow_ant_y],
                "-",
                color=ap_arrow_color,
                linewidth=1.5,
                zorder=2,
            )

            # Draw triangular arrowheads at both ends
            head_len = 0.06  # Arrowhead length
            head_wid = 0.03  # Arrowhead half-width

            # Anterior arrowhead (pointing toward +PC1)
            ant_head_base_x = arrow_ant_x - head_len * pc1[0]
            ant_head_base_y = arrow_ant_y - head_len * pc1[1]
            hw = head_wid
            ahbx, ahby = ant_head_base_x, ant_head_base_y
            ant_head_tri = np.array(
                [
                    [arrow_ant_x, arrow_ant_y],
                    [ahbx + hw * pc1_perp[0], ahby + hw * pc1_perp[1]],
                    [ahbx - hw * pc1_perp[0], ahby - hw * pc1_perp[1]],
                ]
            )
            ax1.fill(
                ant_head_tri[:, 0],
                ant_head_tri[:, 1],
                color=ap_arrow_color,
                zorder=3,
            )

            # Posterior arrowhead (pointing toward -PC1)
            phbx = arrow_post_x + head_len * pc1[0]
            phby = arrow_post_y + head_len * pc1[1]
            post_head_tri = np.array(
                [
                    [arrow_post_x, arrow_post_y],
                    [phbx + hw * pc1_perp[0], phby + hw * pc1_perp[1]],
                    [phbx - hw * pc1_perp[0], phby - hw * pc1_perp[1]],
                ]
            )
            ax1.fill(
                post_head_tri[:, 0],
                post_head_tri[:, 1],
                color=ap_arrow_color,
                zorder=3,
            )

            # Add "A" and "P" labels beyond arrow ends (along PC1 direction)
            ap_lbl = 0.08
            ax1.text(
                arrow_ant_x + ap_lbl * pc1[0],
                arrow_ant_y + ap_lbl * pc1[1],
                "A",
                fontsize=10,
                fontweight="bold",
                color=ap_arrow_color,
                ha="center",
                va="center",
                zorder=10,
            )
            ax1.text(
                arrow_post_x - ap_lbl * pc1[0],
                arrow_post_y - ap_lbl * pc1[1],
                "P",
                fontsize=10,
                fontweight="bold",
                color=ap_arrow_color,
                ha="center",
                va="center",
                zorder=10,
            )

            # Add +PC1 and -PC1 labels at ends of axis
            lo = 0.05
            ax1.text(
                x_offset + ae * pc1[0] + lo * pc1_perp[0],
                ae * pc1[1] + lo * pc1_perp[1],
                "+PC1",
                fontsize=10,
                fontweight="bold",
                color=file_color,
                ha="center",
                va="center",
                alpha=0.8,
                zorder=10,
            )
            ax1.text(
                x_offset - ae * pc1[0] + lo * pc1_perp[0],
                -ae * pc1[1] + lo * pc1_perp[1],
                "-PC1",
                fontsize=10,
                fontweight="bold",
                color=file_color,
                ha="center",
                va="center",
                alpha=0.8,
                zorder=10,
            )

            # Get suggested pair info first (needed for node coloring)
            from_idx = skel.get("suggested_from", -1)
            to_idx = skel.get("suggested_to", -1)
            status = skel.get("status", "NO SUGGESTION")
            from_in_gt = skel.get("from_in_gt", False)
            to_in_gt = skel.get("to_in_gt", False)

            # Status colors (same as status legend)
            status_colors_map = {
                "Both in GT, order correct": "#2ecc71",
                "Both in GT, order reversed": "#f39c12",
                "One node in GT": "#e74c3c",
                "Neither node in GT": "#95a5a6",
                "NO SUGGESTION": "#bdc3c7",
            }
            line_color = status_colors_map.get(status, "#bdc3c7")

            # Plot nodes
            n_nodes = len(x_plot)

            for node_idx in range(n_nodes):
                if np.isnan(x_plot[node_idx]) or np.isnan(y_plot[node_idx]):
                    continue

                # Check if node is NaN (not in GT)
                is_nan = node_idx not in gt
                # Check if node is part of suggested pair
                is_suggested = node_idx in (from_idx, to_idx)

                # Set alpha and size based on node type
                size = 30 if is_nan else 50

                # Dot color: GT=theme, suggested non-GT=status, else file_color
                if not is_nan:
                    dot_color = gt_node_color  # GT nodes use theme color
                    alpha = 1.0
                elif is_suggested:
                    # Suggested non-GT nodes use status color
                    dot_color = line_color
                    alpha = 1.0
                else:
                    # Purple non-GT, non-suggested nodes
                    dot_color = file_color
                    alpha = 0.30

                ax1.scatter(
                    x_plot[node_idx],
                    y_plot[node_idx],
                    c=[dot_color],
                    s=size,
                    alpha=alpha,
                    edgecolors="white" if not is_nan else "none",
                    linewidths=0.5,
                    zorder=5,
                )

                # Add node index label for ranked nodes only
                if not is_nan:
                    ax1.annotate(
                        str(node_idx),
                        (x_plot[node_idx], y_plot[node_idx]),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=13,
                        color=gt_node_color,
                        fontweight="bold",
                        zorder=10,
                    )

            # Draw arrow connecting suggested pair (color based on status)
            # Arrow points from from_node (posterior) to to_node (anterior)
            if from_idx >= 0 and to_idx >= 0:
                from_ok = not np.isnan(x_plot[from_idx]) and not np.isnan(
                    y_plot[from_idx]
                )
                to_ok = not np.isnan(x_plot[to_idx]) and not np.isnan(
                    y_plot[to_idx]
                )
                if from_ok and to_ok:
                    # Draw arrow from posterior to anterior
                    ax1.annotate(
                        "",
                        xy=(x_plot[to_idx], y_plot[to_idx]),
                        xytext=(x_plot[from_idx], y_plot[from_idx]),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=line_color,
                            lw=1.5,
                            mutation_scale=10,
                            alpha=0.85,
                        ),
                        zorder=4,
                    )

                    # Redraw suggested nodes with higher zorder to be on top
                    # GT nodes use theme color, non-GT nodes use line_color
                    if not np.isnan(x_plot[from_idx]):
                        from_color = (
                            gt_node_color if from_idx in gt else line_color
                        )
                        ax1.scatter(
                            x_plot[from_idx],
                            y_plot[from_idx],
                            c=[from_color],
                            s=50,
                            alpha=1.0,
                            edgecolors="white",
                            linewidths=0.5,
                            zorder=6,
                        )
                        ax1.annotate(
                            str(from_idx),
                            (x_plot[from_idx], y_plot[from_idx]),
                            xytext=(4, 4),
                            textcoords="offset points",
                            fontsize=13,
                            color=from_color,
                            fontweight="bold",
                            zorder=10,
                        )

                    if not np.isnan(x_plot[to_idx]):
                        to_color = (
                            gt_node_color if to_idx in gt else line_color
                        )
                        ax1.scatter(
                            x_plot[to_idx],
                            y_plot[to_idx],
                            c=[to_color],
                            s=50,
                            alpha=1.0,
                            edgecolors="white",
                            linewidths=0.5,
                            zorder=6,
                        )
                        ax1.annotate(
                            str(to_idx),
                            (x_plot[to_idx], y_plot[to_idx]),
                            xytext=(4, 4),
                            textcoords="offset points",
                            fontsize=13,
                            color=to_color,
                            fontweight="bold",
                            zorder=10,
                        )

            # Add label below skeleton
            ax1.text(
                x_offset,
                -1.25,
                skel["label"],
                ha="center",
                va="top",
                fontsize=12,
                fontweight="bold",
                color=file_color,
                clip_on=False,
            )
            ax1.text(
                x_offset,
                -1.45,
                skel["best_ind"],
                ha="center",
                va="top",
                fontsize=12,
                fontweight="bold",
                color=file_color,
                clip_on=False,
            )

            # Add suggested pair [ # , # ] with color-coded numbers
            if from_idx >= 0 and to_idx >= 0:
                # Determine colors for each number based on GT membership
                from_num_color = gt_node_color if from_in_gt else line_color
                to_num_color = gt_node_color if to_in_gt else line_color

                # Build text parts with spacing
                pair_y = -1.65
                char_width = 0.08  # Approximate character width in data units

                # Calculate total width and starting position for centering
                from_str = str(from_idx)
                to_str = str(to_idx)
                # Format: "[ " (2) + from + " , " (3) + to + " ]" (2)
                total_chars = 2 + len(from_str) + 3 + len(to_str) + 2
                start_x = x_offset - (total_chars * char_width) / 2

                # Draw each part with spaces
                pos = start_x
                ax1.text(
                    pos,
                    pair_y,
                    "[ ",
                    ha="left",
                    va="top",
                    fontsize=14,
                    fontweight="bold",
                    color="black",
                    clip_on=False,
                )
                pos += char_width * 2
                ax1.text(
                    pos,
                    pair_y,
                    from_str,
                    ha="left",
                    va="top",
                    fontsize=14,
                    fontweight="bold",
                    color=from_num_color,
                    clip_on=False,
                )
                pos += char_width * len(from_str)
                ax1.text(
                    pos,
                    pair_y,
                    " , ",
                    ha="left",
                    va="top",
                    fontsize=14,
                    fontweight="bold",
                    color="black",
                    clip_on=False,
                )
                pos += char_width * 3
                ax1.text(
                    pos,
                    pair_y,
                    to_str,
                    ha="left",
                    va="top",
                    fontsize=14,
                    fontweight="bold",
                    color=to_num_color,
                    clip_on=False,
                )
                pos += char_width * len(to_str)
                ax1.text(
                    pos,
                    pair_y,
                    "  ]",
                    ha="left",
                    va="top",
                    fontsize=14,
                    fontweight="bold",
                    color="black",
                    clip_on=False,
                )

            x_offset += spacing

        # Set axis limits (wider since spanning full row)
        x_total = x_offset - spacing
        ax1.set_xlim(-0.8, x_total + 0.8)
        ax1.set_ylim(-1.85, 1.2)  # Extended to accommodate three-line labels

        # Force equal aspect ratio for square skeleton display
        ax1.set_aspect("equal", adjustable="box")

        # Add legend for AP midline and GT Node (positioned lower right)
        midline_handle = Line2D(
            [0],
            [0],
            linestyle="--",
            color=midline_color,
            linewidth=1.5,
            alpha=0.7,
        )
        gt_node_handle = Line2D(
            [0], [0], marker="o", color="black", linestyle="None", markersize=6
        )
        midline_legend = ax1.legend(
            [midline_handle, gt_node_handle],
            ["A/P midline", "GT Node"],
            loc="lower right",
            fontsize=9,
            framealpha=label_bg_alpha,
            bbox_to_anchor=(1.12, -0.05),
            prop={"weight": "bold"},
            facecolor=bg_color,
            labelcolor=text_color,
        )
        ax1.add_artist(midline_legend)

    # Determine dominant status color for arrow text
    status_colors_map = {
        "Both in GT, order correct": "#2ecc71",
        "Both in GT, order reversed": "#f39c12",
        "One node in GT": "#e74c3c",
        "Neither node in GT": "#95a5a6",
        "NO SUGGESTION": "#bdc3c7",
    }
    status_counts = {}
    for f in files:
        status = suggested_analysis.get(f, {}).get("status", "NO SUGGESTION")
        status_counts[status] = status_counts.get(status, 0) + 1
    dominant_status = (
        max(status_counts, key=status_counts.get)
        if status_counts
        else "NO SUGGESTION"
    )
    dominant_color = status_colors_map.get(dominant_status, "#bdc3c7")

    # Add arrow description at top left (bold, dominant status color)
    fig.text(
        0.01,
        0.99,
        "vector points in the inferred P\u2192A direction for suggested pair",
        fontsize=8,
        fontweight="normal",
        color=dominant_color,
        ha="left",
        va="top",
    )
    fig.text(
        0.01,
        0.97,
        "(max AP separation, distal preferred)",
        fontsize=8,
        fontweight="normal",
        color=dominant_color,
        ha="left",
        va="top",
    )

    # Add formula text at top right corner of figure
    line_spacing = 0.018
    y_start = 0.99
    x_right = 1.0
    fig.text(
        x_right,
        y_start,
        "R = √(C² + S²)",
        fontsize=8,
        fontweight="normal",
        color=text_color,
        ha="right",
        va="top",
    )
    fig.text(
        x_right,
        y_start - line_spacing,
        "M = |n₊ − n₋| / (n₊ + n₋)",
        fontsize=8,
        fontweight="normal",
        color=text_color,
        ha="right",
        va="top",
    )
    fig.text(
        x_right,
        y_start - 2 * line_spacing,
        "C, S = mean cos θ, mean sin θ of centroid velocities",
        fontsize=8,
        fontweight="normal",
        color=text_color,
        ha="right",
        va="top",
    )
    fig.text(
        x_right,
        y_start - 3 * line_spacing,
        "n₊, n₋ = # of pos./neg. velocity proj. onto PC1",
        fontsize=8,
        fontweight="normal",
        color=text_color,
        ha="right",
        va="top",
    )
    fig.text(
        0.01,
        y_start - 3 * line_spacing,
        "anterior = +PC1 if n₊ > n₋, else −PC1",
        fontsize=8,
        fontweight="normal",
        color=text_color,
        ha="left",
        va="top",
    )

    # Build labels list for bar charts (use animal identity)
    labels = [best_individual.get(f, f[:15]) for f in files]

    # Panel 3: Suggested pair analysis
    ax3 = fig.add_subplot(gs[1, 0], facecolor=bg_color)

    # Status colors for legend (kept for tile 1)
    status_colors = {
        "Both in GT, order correct": "#2ecc71",
        "Both in GT, order reversed": "#f39c12",
        "One node in GT": "#e74c3c",
        "Neither node in GT": "#95a5a6",
        "NO SUGGESTION": "#bdc3c7",
    }

    # Create custom legend handler for arrow with dots at each end
    class ArrowHandler(HandlerBase):
        def __init__(self, color, left_dot_color=None, right_dot_color=None):
            self.color = color
            self.left_dot_color = left_dot_color if left_dot_color else color
            self.right_dot_color = (
                right_dot_color if right_dot_color else color
            )
            super().__init__()

        def create_artists(
            self,
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        ):
            # Arrow line with triangular head
            x_start = xdescent + 1
            x_end = xdescent + width - 1
            y_mid = ydescent + height / 2

            # Draw arrow shaft (shorter to make room for visible arrowhead)
            line = Line2D(
                [x_start + 5, x_end - 10],
                [y_mid, y_mid],
                color=self.color,
                linewidth=3,
                transform=trans,
            )

            # Draw arrowhead as a triangle (larger and more visible)
            head_width = 5
            head_length = 5
            arrow_head = mpatches.FancyArrow(
                x_end - 10,
                y_mid,
                5,
                0,
                width=0,
                head_width=head_width,
                head_length=head_length,
                fc=self.color,
                ec=self.color,
                transform=trans,
            )

            # Draw dots at each end with specified colors
            ldc, rdc = self.left_dot_color, self.right_dot_color
            dot_start = Line2D(
                [x_start],
                [y_mid],
                marker="o",
                markersize=7,
                markerfacecolor=ldc,
                markeredgecolor=ldc,
                linestyle="None",
                transform=trans,
            )
            dot_end = Line2D(
                [x_end + 4],
                [y_mid],
                marker="o",
                markersize=7,
                markerfacecolor=rdc,
                markeredgecolor=rdc,
                linestyle="None",
                transform=trans,
            )

            return [line, arrow_head, dot_start, dot_end]

    # Create legend for status colors at bottom left of tile 1
    legend_labels = [
        ("Both in GT, order correct", "Both in GT, Match A\u2194P"),
        ("Both in GT, order reversed", "Both in GT, Mismatch A\u2194P"),
        ("One node in GT", "One in GT"),
        ("One node in GT SWAPPED", "One in GT"),
        ("Neither node in GT", "Neither in GT"),
    ]

    # Create legend handles and handler map
    legend_handles = []
    handler_map = {}
    legend_colors = []  # Track colors for text coloring
    for status, label in legend_labels:
        # Get base color (strip SWAPPED suffix for lookup)
        base_status = status.replace(" SWAPPED", "")
        color = status_colors[base_status]
        legend_colors.append(color)
        handle = Line2D([], [], label=label)
        legend_handles.append(handle)
        # "One in GT" has black left dot (GT) and red right dot (non-GT)
        if status == "One node in GT":
            handler_map[handle] = ArrowHandler(
                color, left_dot_color="black", right_dot_color=color
            )
        elif status == "One node in GT SWAPPED":
            handler_map[handle] = ArrowHandler(
                color, left_dot_color=color, right_dot_color="black"
            )
        else:
            handler_map[handle] = ArrowHandler(color)

    leg_status = fig.legend(
        handles=legend_handles,
        handler_map=handler_map,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.58),
        fontsize=10,
        frameon=False,
        prop={"weight": "bold"},
    )

    # Set legend text colors to match arrow colors (One in GT = black)
    zipped = zip(
        leg_status.get_texts(), legend_labels, legend_colors, strict=False
    )
    for text, (status, _label), color in zipped:
        if "One node in GT" in status:
            text.set_color("black")
        else:
            text.set_color(color)

    # Add "Suggested AP Pair:" label below legend
    fig.text(
        0.01,
        0.565,
        "Suggested AP Pair:",
        fontsize=13,
        fontweight="bold",
        color="black",
        ha="left",
        va="center",
    )

    # Panel 3: GT Node Ranks (descending bars)
    # Reassign ranks: highest rank = most anterior (tallest bar)
    # Bar width and group spacing
    group_width = 0.8

    ax3.set_title(
        "Hand-Curated GT Node Rankings",
        fontsize=14,
        fontweight="bold",
        color=text_color,
    )

    for i, f in enumerate(files):
        gt = GROUND_TRUTH.get(f, {})
        file_color = colors[i]

        if not gt:
            continue

        # Higher rank = more anterior, so rank directly maps to bar height
        # Sort by rank descending (most anterior/highest rank first)
        sorted_nodes = sorted(gt.items(), key=lambda x: x[1], reverse=True)
        n_gt_nodes = len(sorted_nodes)

        for j, (node_idx, rank) in enumerate(sorted_nodes):
            # Rank value is the bar height (higher = more anterior)
            height = rank

            # Position within group (most anterior = leftmost = tallest)
            x_pos = (
                i - group_width / 2 + (j + 0.5) * (group_width / n_gt_nodes)
            )

            # Draw bar
            ax3.bar(
                x_pos,
                height,
                width=group_width / n_gt_nodes * 0.85,
                color=file_color,
                edgecolor=axis_color,
                linewidth=0.5,
                alpha=0.75,
            )

            # Add node index as bold text on top of bar
            ax3.text(
                x_pos,
                height + 0.1,
                str(node_idx),
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                color=text_color,
            )

    ax3.set_xticks(range(n_files))
    ax3.set_xticklabels(
        labels, rotation=30, ha="right", fontsize=14, fontweight="bold"
    )
    # Color x-axis labels to match skeleton colors
    for tick_label, color in zip(
        ax3.get_xticklabels(),
        colors,
        strict=True,
    ):
        tick_label.set_color(color)
    ax3.set_ylabel(
        "Rank (higher = more anterior)",
        fontsize=14,
        fontweight="bold",
        color=text_color,
    )
    ax3.set_ylim(0, max(len(gt) for gt in GROUND_TRUTH.values()) + 1)

    # Make tile 3 tick labels bold
    ax3.tick_params(axis="y", labelsize=10)
    for label in ax3.get_yticklabels():
        label.set_fontweight("bold")

    # Panel 4: Filter cascade progression per dataset
    ax4 = fig.add_subplot(gs[1, 1], facecolor=bg_color)

    import matplotlib as mpl

    mpl.rcParams["hatch.linewidth"] = 1.5

    # Compute cascade data per file as raw pair counts.
    # Step 1 shows C(n_candidates, 2) — pair potential of surviving nodes.
    # Steps 2-3 show surviving pair counts directly.
    # Since C(K,2) >= P >= D, bars are guaranteed to decrease.
    all_counts = [[], [], []]
    for f in files:
        cs = cascade_stats.get(f, {})
        n_s1 = cs.get("n_step1_candidates", 0)
        n_s2 = cs.get("n_step2_pairs", 0)
        n_s3d = cs.get("n_step3_distal", 0)

        s1_pairs = n_s1 * (n_s1 - 1) // 2 if n_s1 >= 2 else 0
        all_counts[0].append(s1_pairs)
        all_counts[1].append(n_s2)
        all_counts[2].append(n_s3d)

    x_pos = np.arange(n_files)
    bar_width = 0.25

    # Hatch patterns distinguish steps; bar color matches dataset
    step_hatches = ["--", "||", "+++"]
    step_names = ["Step 1: Lateral", "Step 2: Opposite", "Step 3: Distal"]

    for j, (counts, hatch, _sn) in enumerate(
        zip(
            all_counts,
            step_hatches,
            step_names,
            strict=True,
        )
    ):
        offset = (j - 1) * bar_width
        for i, count in enumerate(counts):
            c = colors[i]
            fc = (c[0], c[1], c[2], 0.30)
            ax4.bar(
                x_pos[i] + offset,
                count,
                bar_width,
                color=fc,
                edgecolor="black",
                linewidth=0.5,
                hatch=hatch,
            )
            bx = x_pos[i] + offset
            by = count + 0.5
            ax4.text(
                bx,
                by,
                str(count),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=text_color,
            )

    # Small legend showing only hatch patterns (no color)
    legend_patches = []
    for hatch, sn in zip(step_hatches, step_names, strict=True):
        legend_patches.append(
            mpatches.Patch(
                facecolor="white",
                edgecolor="black",
                linewidth=0.5,
                hatch=hatch,
                label=sn,
            )
        )
    ax4.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=8,
        framealpha=label_bg_alpha,
        facecolor=bg_color,
        labelcolor=text_color,
        handlelength=1.5,
        handleheight=1.0,
    )

    ax4.set_ylabel(
        "Candidate Pairs",
        fontsize=14,
        fontweight="bold",
        color=text_color,
    )
    # Auto-scale y-axis with padding
    max_count = max(max(c) for c in all_counts) if any(all_counts[0]) else 1
    ax4.set_ylim(0, max_count * 1.25)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(
        labels, rotation=30, ha="right", fontsize=14, fontweight="bold"
    )
    for tick_label, color in zip(
        ax4.get_xticklabels(),
        colors,
        strict=True,
    ):
        tick_label.set_color(color)

    ax4.tick_params(axis="y", labelsize=10, colors=text_color)
    for label in ax4.get_yticklabels():
        label.set_fontweight("bold")

    ax4.set_title(
        "3-Step Filter Cascade Progression",
        fontsize=14,
        fontweight="bold",
        color=text_color,
    )
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08)

    for ax in [ax3, ax4]:
        ax.tick_params(axis="y", colors=text_color, labelsize=10)
        for spine in ax.spines.values():
            spine.set_color(axis_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        if hasattr(ax, "title"):
            ax.title.set_color(text_color)

    for tick_label, color in zip(
        ax3.get_xticklabels(),
        colors,
        strict=True,
    ):
        tick_label.set_color(color)
        tick_label.set_fontsize(14)
    for tick_label, color in zip(
        ax4.get_xticklabels(),
        colors,
        strict=True,
    ):
        tick_label.set_color(color)
        tick_label.set_fontsize(14)

    ax4.yaxis.label.set_color("black")

    # Extract full timestamp (YYYYMMDD_HHMMSS) from H5 filename
    stem_parts = output_path.stem.split("_")
    stem_suffix = "_".join(stem_parts[-2:])
    fig_path = FIGURES_DIR / f"ap_validation_results_{stem_suffix}.svg"
    plt.savefig(
        fig_path, format="svg", facecolor=bg_color, bbox_inches="tight"
    )
    plt.close(fig)

    print(f"\nSaved cross-dataset validation figure to: {fig_path}")
    return fig_path


def create_individual_plot(  # noqa: C901
    file_stem,
    avg_skeleton,
    pc1,
    keypoint_names,
    metrics,
    output_path,
    vel_projs=None,
    individual_name=None,
    all_individuals_metrics=None,
    file_color=None,
    individual_accuracy=None,
):
    """Create a detailed 2×2 plot for a single file/animal.

    The resulting figure contains four tiles showing (1) the longitudinal
    spread of keypoints along the first principal component, (2) the lateral
    spread along the second principal component, (3) the direction of
    anterior–posterior motion via centroid velocity, and (4) a scatter plot
    relating the product of resultant length and vote margin (R×M) to
    ground‑truth ordering accuracy.  Colour schemes and labels are chosen
    consistently with the summary AP validation figure.

    Parameters
    ----------
    file_stem : str
        The stem of the file name (without extension).  Used to derive
        display labels and file names for the output figure.
    avg_skeleton : np.ndarray
        Array of shape ``(n_keypoints, 2)`` containing the average x,y
        coordinates of each keypoint after segmentation and outlier removal.
    pc1 : np.ndarray
        The first principal component vector (length‑2) representing the
        anterior–posterior axis.
    keypoint_names : list[str]
        Names of the keypoints corresponding to rows of ``avg_skeleton``.
    metrics : dict
        Dictionary of scalar metrics summarising the individual's movement
        and orientation.  Expected keys include ``'anterior_sign'``,
        ``'vote_margin'``, ``'resultant_length'``, ``'circ_mean_dir'``,
        ``'num_selected_frames'`` and ``'n_frames'``.
    output_path : Path | str
        Directory or filename where the generated figure will be written.
    vel_projs : np.ndarray | None, optional
        One‑dimensional array of projections of velocity vectors onto
        ``pc1``.  When provided, a histogram of these values is drawn in
        Tile 3.  If ``None`` or empty, the histogram is omitted.
    individual_name : str | None, optional
        Name of the individual animal, shown in the bottom title and used
        for the scatter plot legend.  If ``None``, only the file label
        appears.
    all_individuals_metrics : dict[str, dict[str, float]], optional
        Mapping from individual names to dictionaries containing mean
        resultant length ``'R'`` and vote margin ``'M'`` values.  Used to
        populate the scatter plot (Tile 4).
    file_color : tuple | None, optional
        RGB(A) colour for this file, matching the colours used in the
        summary AP validation figure.  When ``None``, a default colour
        palette is used.
    individual_accuracy : dict[str, dict[str, float]] | None, optional
        Mapping from individual names to accuracy dictionaries.  If
        provided, accuracy values are displayed in the scatter plot legend.

    Returns
    -------
    matplotlib.figure.Figure | None
        The created figure.  If there are fewer than two valid keypoints
        after filtering, the function prints a message and returns ``None``.

    """
    n_keypoints = len(avg_skeleton)

    # Generate unique visually distinct colors for all nodes
    # Start with maximally distinct base colors, then add shades if needed
    base_colors = [
        (0.12, 0.47, 0.71),  # Blue
        (1.00, 0.50, 0.05),  # Orange
        (0.17, 0.63, 0.17),  # Green
        (0.84, 0.15, 0.16),  # Red
        (0.58, 0.40, 0.74),  # Purple
        (0.55, 0.34, 0.29),  # Brown
        (0.89, 0.47, 0.76),  # Pink
        (0.50, 0.50, 0.50),  # Gray
        (0.74, 0.74, 0.13),  # Olive/Yellow
        (0.09, 0.75, 0.81),  # Cyan
        (0.00, 0.80, 0.60),  # Teal
        (0.90, 0.30, 0.50),  # Magenta
        (0.40, 0.20, 0.60),  # Dark purple
        (0.95, 0.70, 0.20),  # Gold
        (0.30, 0.70, 0.90),  # Sky blue
        (0.70, 0.90, 0.30),  # Lime
    ]

    def generate_colors(n):
        """Generate n distinct colors, using shades if needed."""
        colors = []
        n_base = len(base_colors)

        if n <= n_base:
            # Use base colors directly
            colors = [(*base_colors[i], 1.0) for i in range(n)]
        else:
            # Use all base colors, then add lighter/darker shades
            for i in range(n):
                base_idx = i % n_base
                # 0 = normal, 1 = lighter, 2 = darker, etc.
                shade_level = i // n_base

                r, g, b = base_colors[base_idx]

                if shade_level == 0:
                    # Original color
                    pass
                elif shade_level % 2 == 1:
                    # Lighter shade
                    factor = 0.4 * ((shade_level + 1) // 2)
                    r = min(1.0, r + (1 - r) * factor)
                    g = min(1.0, g + (1 - g) * factor)
                    b = min(1.0, b + (1 - b) * factor)
                else:
                    # Darker shade
                    factor = 0.4 * (shade_level // 2)
                    r = max(0.0, r * (1 - factor))
                    g = max(0.0, g * (1 - factor))
                    b = max(0.0, b * (1 - factor))

                colors.append((r, g, b, 1.0))

        return colors

    colors = generate_colors(n_keypoints)

    # Get valid keypoints (non-NaN)
    valid_mask = ~np.any(np.isnan(avg_skeleton), axis=1)
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) < 2:
        print(f"  Skipping {file_stem}: not enough valid keypoints")
        return

    # Compute PC2 as perpendicular to PC1
    pc1 = pc1 / np.linalg.norm(pc1)  # Normalize
    pc2 = np.array([-pc1[1], pc1[0]])  # 90 degree rotation

    # Compute projections onto PC1 and PC2
    proj_pc1 = avg_skeleton @ pc1
    proj_pc2 = avg_skeleton @ pc2

    # Display parameters
    valid_coords = avg_skeleton[valid_mask]
    shape_radius = np.max(np.abs(valid_coords)) * 1.3
    display_radius = shape_radius * 1.2

    # Body axis extent
    body_axis_extent = np.max(np.abs(proj_pc1[valid_mask])) * 1.3

    # Use file color for all keypoints (matching summary figure)

    # Extract metrics
    anterior_sign = int(metrics.get("anterior_sign", 1))
    vote_margin = metrics.get("vote_margin", 0)
    resultant_length = metrics.get("resultant_length", 0)
    circ_mean_dir = metrics.get("circ_mean_dir", np.nan)
    num_selected_frames = int(metrics.get("num_selected_frames", 0))
    n_frames = int(metrics.get("n_frames", 1))

    # Compute Vc (net velocity display vector)
    # Scale so R=1 displays at 3*shape_radius, R scales 0-1
    # Clamp so arrow tip + label stays within display_radius
    vel_display_max = shape_radius * 3.0
    vel_display_len = resultant_length * vel_display_max
    max_vc_len = display_radius * 0.8  # clamp to 80% of axis boundary
    vel_display_len = min(vel_display_len, max_vc_len)
    if not np.isnan(circ_mean_dir):
        cos_dir = np.cos(circ_mean_dir)
        sin_dir = np.sin(circ_mean_dir)
        net_vel_display = vel_display_len * np.array([cos_dir, sin_dir])
    else:
        net_vel_display = np.array([0.0, 0.0])

    centroid_alpha = 0.7
    label_offset = 0.12 * shape_radius

    # Create figure with dark background
    fig = plt.figure(figsize=(14, 10), facecolor="black")

    # Compute R*M
    rxm = resultant_length * vote_margin

    # Top title with description
    fig.suptitle(
        "AP Validation Detail: BBox-Centroid Average Skeleton",
        fontsize=12,
        fontweight="normal",
        color="white",
        y=0.97,
    )

    # Bottom title with frame count, dataset label, and individual name
    # Move to right (beneath tile 4) if >=20 nodes to avoid legend overlap
    file_label = FILE_LABELS.get(file_stem, file_stem[:15])
    _ind_label = f" ({individual_name})" if individual_name else ""
    pct = 100.0 * num_selected_frames / n_frames if n_frames > 0 else 0
    if n_keypoints >= 20:
        text_x, text_ha = 0.78, "center"  # Right side, beneath tile 4
    else:
        text_x, text_ha = 0.5, "center"  # Centered
    fig.text(
        text_x,
        0.045,
        f"{num_selected_frames} segment frames ({pct:.1f}% of all)",
        fontsize=14,
        fontweight="normal",
        color="white",
        ha=text_ha,
        va="bottom",
    )
    # Render file label with individual name
    if individual_name:
        fig.text(
            text_x,
            0.012,
            f"{file_label} ({individual_name})",
            fontsize=14,
            fontweight="bold",
            color="white",
            ha=text_ha,
            va="bottom",
        )
    else:
        fig.text(
            text_x,
            0.012,
            file_label,
            fontsize=14,
            fontweight="bold",
            color="white",
            ha=text_ha,
            va="bottom",
        )

    # Fixed width ratio for Tile 4 scatter plot
    tile4_width_ratio = 0.8

    # Create 2x2 grid with space for center legend
    right_margin = 0.92
    gs = fig.add_gridspec(
        2,
        2,
        hspace=0.25,
        wspace=0.35,
        left=0.08,
        right=right_margin,
        top=0.90,
        bottom=0.10,
        width_ratios=[1, tile4_width_ratio],
    )

    # TILE 1: Longitudinal Spread (PC1)
    ax1 = fig.add_subplot(gs[0, 0], facecolor="black")
    ax1.set_title(
        "Longitudinal Spread (PC1 Projections)",
        fontsize=12,
        fontweight="normal",
        color="white",
    )

    # PC1 axis line segments (broken at keypoint range)
    min_proj = np.min(proj_pc1[valid_mask])
    max_proj = np.max(proj_pc1[valid_mask])

    ax1.plot(
        [-1.5 * body_axis_extent * pc1[0], min_proj * pc1[0]],
        [-1.5 * body_axis_extent * pc1[1], min_proj * pc1[1]],
        "-",
        color=(1, 1, 1, 0.65),
        linewidth=1.5,
    )
    ax1.plot(
        [max_proj * pc1[0], 1.5 * body_axis_extent * pc1[0]],
        [max_proj * pc1[1], 1.5 * body_axis_extent * pc1[1]],
        "-",
        color=(1, 1, 1, 0.65),
        linewidth=1.5,
    )

    # PC1 labels at edges
    pc1_perp = np.array([-pc1[1], pc1[0]])
    pc_label_extent = 1.1 * body_axis_extent
    ax1.text(
        pc_label_extent * pc1[0] + pc1_perp[0] * label_offset,
        pc_label_extent * pc1[1] + pc1_perp[1] * label_offset,
        "+PC1",
        color="white",
        fontsize=10,
        ha="center",
        va="center",
    )
    ax1.text(
        -pc_label_extent * pc1[0] + pc1_perp[0] * label_offset,
        -pc_label_extent * pc1[1] + pc1_perp[1] * label_offset,
        "—PC1",
        color="white",
        fontsize=10,
        ha="center",
        va="center",
    )

    # Draw projection lines and keypoints
    for i in valid_idx:
        tip_x, tip_y = avg_skeleton[i]
        proj_x = proj_pc1[i] * pc1[0]
        proj_y = proj_pc1[i] * pc1[1]

        ax1.plot(
            [tip_x, proj_x],
            [tip_y, proj_y],
            "--",
            color=(*colors[i][:3], 0.5),
            linewidth=1,
        )
        ax1.plot(
            [0, proj_x],
            [0, proj_y],
            "-",
            color=(*colors[i][:3], 0.5),
            linewidth=2,
        )

    for i in valid_idx:
        ax1.scatter(
            avg_skeleton[i, 0],
            avg_skeleton[i, 1],
            s=120,
            c=[colors[i]],
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

    ax1.scatter(
        0,
        0,
        s=120,
        c="gray",
        edgecolors="black",
        linewidths=0.5,
        alpha=centroid_alpha,
        zorder=6,
    )

    ax1.set_xlim(-display_radius, display_radius)
    ax1.set_ylim(-display_radius, display_radius)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X", color="white", fontsize=10)
    ax1.set_ylabel(
        "Y", color="white", fontsize=10, rotation=0, ha="right", va="center"
    )
    ax1.tick_params(colors="gray", labelsize=8)
    ax1.grid(True, color="gray", alpha=0.3, linestyle="-", linewidth=0.5)
    for spine in ax1.spines.values():
        spine.set_color("gray")

    # TILE 2: Lateral Spread (PC2)
    ax2 = fig.add_subplot(gs[0, 1], facecolor="black")
    ax2.set_title(
        "Lateral Spread (PC2 Projections)",
        fontsize=12,
        fontweight="normal",
        color="white",
    )

    min_proj_pc2 = np.min(proj_pc2[valid_mask])
    max_proj_pc2 = np.max(proj_pc2[valid_mask])

    ax2.plot(
        [-1.5 * body_axis_extent * pc2[0], min_proj_pc2 * pc2[0]],
        [-1.5 * body_axis_extent * pc2[1], min_proj_pc2 * pc2[1]],
        "-",
        color=(1, 1, 1, 0.65),
        linewidth=1.5,
    )
    ax2.plot(
        [max_proj_pc2 * pc2[0], 1.5 * body_axis_extent * pc2[0]],
        [max_proj_pc2 * pc2[1], 1.5 * body_axis_extent * pc2[1]],
        "-",
        color=(1, 1, 1, 0.65),
        linewidth=1.5,
    )

    # PC2 labels at edges
    pc2_perp = np.array([-pc2[1], pc2[0]])
    pc_label_extent_pc2 = 1.1 * body_axis_extent
    ax2.text(
        pc_label_extent_pc2 * pc2[0] + pc2_perp[0] * label_offset,
        pc_label_extent_pc2 * pc2[1] + pc2_perp[1] * label_offset,
        "+PC2",
        color="white",
        fontsize=10,
        ha="center",
        va="center",
    )
    ax2.text(
        -pc_label_extent_pc2 * pc2[0] + pc2_perp[0] * label_offset,
        -pc_label_extent_pc2 * pc2[1] + pc2_perp[1] * label_offset,
        "—PC2",
        color="white",
        fontsize=10,
        ha="center",
        va="center",
    )

    for i in valid_idx:
        tip_x, tip_y = avg_skeleton[i]
        proj_x = proj_pc2[i] * pc2[0]
        proj_y = proj_pc2[i] * pc2[1]

        ax2.plot(
            [tip_x, proj_x],
            [tip_y, proj_y],
            "--",
            color=(*colors[i][:3], 0.5),
            linewidth=1,
        )
        ax2.plot(
            [0, proj_x],
            [0, proj_y],
            "-",
            color=(*colors[i][:3], 0.5),
            linewidth=2,
        )

    for i in valid_idx:
        ax2.scatter(
            avg_skeleton[i, 0],
            avg_skeleton[i, 1],
            s=120,
            c=[colors[i]],
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

    ax2.scatter(
        0,
        0,
        s=120,
        c="gray",
        edgecolors="black",
        linewidths=0.5,
        alpha=centroid_alpha,
        zorder=6,
    )

    ax2.set_xlim(-display_radius, display_radius)
    ax2.set_ylim(-display_radius, display_radius)
    ax2.set_aspect("equal")
    ax2.set_xlabel("X", color="white", fontsize=10)
    ax2.set_ylabel(
        "Y", color="white", fontsize=10, rotation=0, ha="right", va="center"
    )
    ax2.tick_params(colors="gray", labelsize=8)
    ax2.grid(True, color="gray", alpha=0.3, linestyle="-", linewidth=0.5)
    for spine in ax2.spines.values():
        spine.set_color("gray")

    # TILE 3: Inferred AP Direction (Velocity Voting)
    ax3 = fig.add_subplot(gs[1, 0], facecolor="black")
    ax3.set_title(
        "Inferred AP Direction (Velocity Voting)",
        fontsize=12,
        fontweight="normal",
        color="white",
    )

    vel_proj_scalar = np.dot(net_vel_display, pc1)
    min_proj_vel = min(0, vel_proj_scalar)
    max_proj_vel = max(0, vel_proj_scalar)

    ax3.plot(
        [-1.5 * body_axis_extent * pc1[0], min_proj_vel * pc1[0]],
        [-1.5 * body_axis_extent * pc1[1], min_proj_vel * pc1[1]],
        "-",
        color=(1, 1, 1, 0.65),
        linewidth=1.5,
    )
    ax3.plot(
        [max_proj_vel * pc1[0], 1.5 * body_axis_extent * pc1[0]],
        [max_proj_vel * pc1[1], 1.5 * body_axis_extent * pc1[1]],
        "-",
        color=(1, 1, 1, 0.65),
        linewidth=1.5,
    )

    vel_proj_x = vel_proj_scalar * pc1[0]
    vel_proj_y = vel_proj_scalar * pc1[1]
    ax3.plot(
        [0, vel_proj_x],
        [0, vel_proj_y],
        "-",
        color=(1, 1, 1, 0.5),
        linewidth=1.5,
    )

    body_perp = np.array([-pc1[1], pc1[0]])

    # Velocity projection histogram along PC1
    if vel_projs is not None and len(vel_projs) > 0:
        # Adapt bins based on data size, keep consistent transparency
        n_vals = len(vel_projs)
        # Use a ternary expression rather than an if–else block (SIM108)
        num_bins = max(5, n_vals // 3) if n_vals < 50 else 25
        hist_alpha = 0.25
        max_bar_height = 1.0 * shape_radius

        vp_min = np.min(vel_projs)
        vp_max = np.max(vel_projs)
        data_range = vp_max - vp_min

        # Ensure minimum bin width similar to 2Mice
        min_bin_width = shape_radius / 12
        min_range = min_bin_width * num_bins
        if data_range < min_range:
            center = (vp_min + vp_max) / 2
            vp_min = center - min_range / 2
            vp_max = center + min_range / 2
            data_range = min_range

        edge_pad = data_range * 0.02
        bin_edges = np.linspace(
            vp_min - edge_pad, vp_max + edge_pad, num_bins + 1
        )

        bin_counts, _ = np.histogram(vel_projs, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        max_count = np.max(bin_counts)
        # Set minimum bar height so even count=1 is visible
        min_bar_height = 0.15 * max_bar_height
        if max_count > 0:
            bar_heights = (bin_counts / max_count) * max_bar_height
            # Apply minimum height for non-zero bins
            bar_heights = np.where(
                bin_counts > 0,
                np.maximum(bar_heights, min_bar_height),
                bar_heights,
            )
        else:
            bar_heights = np.zeros_like(bin_counts)

        # Draw bars as filled polygons in PC1/perp coordinate frame
        for bi in range(len(bin_counts)):
            if bin_counts[bi] == 0:
                continue

            pc1_lo = bin_edges[bi]
            pc1_hi = bin_edges[bi + 1]
            bh = bar_heights[bi]

            # Four corners: bottom-left, bottom-right, top-right, top-left
            corners_pc1 = np.array([pc1_lo, pc1_hi, pc1_hi, pc1_lo])
            corners_perp = np.array([0, 0, bh, bh])

            # Transform to x,y using PC1 and body_perp basis
            cx = corners_pc1 * pc1[0] + corners_perp * body_perp[0]
            cy = corners_pc1 * pc1[1] + corners_perp * body_perp[1]

            # Color by sign: blue for +PC1, red for -PC1
            if bin_centers[bi] > 0:
                bar_color = (0.4, 0.75, 1.0)  # blue
            else:
                bar_color = (1.0, 0.45, 0.35)  # red

            ax3.fill(
                cx,
                cy,
                color=bar_color,
                alpha=hist_alpha,
                edgecolor="black",
                linewidth=0.5,
                zorder=2,
            )

            # Add bin count label at top of bar (centered within bar)
            bin_center_pc1 = (pc1_lo + pc1_hi) / 2
            # At top of bar, slightly inside
            label_perp_offset = bh - 0.02 * shape_radius
            lx = bin_center_pc1 * pc1[0] + label_perp_offset * body_perp[0]
            ly = bin_center_pc1 * pc1[1] + label_perp_offset * body_perp[1]
            # Rotation angle matches PC1 direction
            rot_deg = np.degrees(np.arctan2(pc1[1], pc1[0]))
            ax3.text(
                lx,
                ly,
                str(bin_counts[bi]),
                color="white",
                fontsize=5,
                ha="center",
                va="top",
                rotation=rot_deg,
                rotation_mode="anchor",
                zorder=20,
            )

    for i in valid_idx:
        ax3.scatter(
            avg_skeleton[i, 0],
            avg_skeleton[i, 1],
            s=120,
            c=[colors[i]],
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

    # PC1 labels at endpoints of PC line, on opposite side of histogram
    pc_label_extent = 1.1 * body_axis_extent  # Closer to tile center
    pc_label_offset = label_offset * 1.5  # Offset perpendicular to PC1
    ax3.text(
        pc_label_extent * pc1[0] - body_perp[0] * pc_label_offset,
        pc_label_extent * pc1[1] - body_perp[1] * pc_label_offset,
        "+PC1",
        color="white",
        fontsize=10,
        ha="center",
        va="center",
    )
    ax3.text(
        -pc_label_extent * pc1[0] - body_perp[0] * pc_label_offset,
        -pc_label_extent * pc1[1] - body_perp[1] * pc_label_offset,
        "—PC1",
        color="white",
        fontsize=10,
        ha="center",
        va="center",
    )

    # Vc arrow (circular-mean velocity)
    if np.linalg.norm(net_vel_display) > 0:
        net_vel_angle = np.arctan2(net_vel_display[1], net_vel_display[0])

        ax3.plot(
            [0, net_vel_display[0]],
            [0, net_vel_display[1]],
            "-",
            color="white",
            linewidth=1.2,
            zorder=7,
        )

        arrow_len = 0.06 * shape_radius
        arrow_wid = 0.03 * shape_radius
        local_tri_x = np.array([0, -arrow_len, -arrow_len])
        local_tri_y = np.array([0, arrow_wid, -arrow_wid])
        cos_a, sin_a = np.cos(net_vel_angle), np.sin(net_vel_angle)
        tri_x = local_tri_x * cos_a - local_tri_y * sin_a + net_vel_display[0]
        tri_y = local_tri_x * sin_a + local_tri_y * cos_a + net_vel_display[1]
        ax3.fill(
            tri_x,
            tri_y,
            color="white",
            edgecolor="black",
            linewidth=0.5,
            zorder=8,
            clip_on=True,
        )

        vc_label_offset = 0.12 * shape_radius
        vc_label_x = net_vel_display[0] + vc_label_offset * np.cos(
            net_vel_angle
        )
        vc_label_y = net_vel_display[1] + vc_label_offset * np.sin(
            net_vel_angle
        )
        ax3.text(
            vc_label_x,
            vc_label_y,
            r"$\mathbf{V_c}$",
            color="white",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=10,
            clip_on=True,
        )

    # AP midline and bidirectional arrow (parallel to PC1)
    min_p = np.min(proj_pc1[valid_mask])
    max_p = np.max(proj_pc1[valid_mask])
    midpoint_pc1 = (min_p + max_p) / 2
    midline_center = midpoint_pc1 * pc1  # Position along PC1
    midline_half_len = 0.35 * shape_radius
    pc1_perp = np.array([-pc1[1], pc1[0]])  # Perpendicular to PC1

    # Draw AP midline (dotted beige, perpendicular to PC1)
    ax3.plot(
        [
            midline_center[0] - midline_half_len * pc1_perp[0],
            midline_center[0] + midline_half_len * pc1_perp[0],
        ],
        [
            midline_center[1] - midline_half_len * pc1_perp[1],
            midline_center[1] + midline_half_len * pc1_perp[1],
        ],
        ":",
        color=(0.82, 0.71, 0.55),
        linewidth=1.5,
        zorder=4,
    )

    # AP bidirectional arrow (along PC1, next to midline)
    ap_arrow_offset = midline_half_len * 1.1  # Offset perpendicular to PC1
    # Opposite side of histogram
    ap_arrow_center = midline_center - ap_arrow_offset * pc1_perp
    ap_arrow_half_len = 0.225 * shape_radius  # Shortened by 0.5x

    # Arrow endpoints along PC1
    ap_arrow_ant = ap_arrow_center + ap_arrow_half_len * pc1 * anterior_sign
    ap_arrow_post = ap_arrow_center - ap_arrow_half_len * pc1 * anterior_sign

    # Draw arrow shaft
    ax3.plot(
        [ap_arrow_post[0], ap_arrow_ant[0]],
        [ap_arrow_post[1], ap_arrow_ant[1]],
        "-",
        color=(0.82, 0.71, 0.55),
        linewidth=1.5,
        zorder=4,
    )

    # Draw arrowheads
    head_len = 0.08 * shape_radius
    head_wid = 0.04 * shape_radius

    # Anterior arrowhead
    ant_head_base = ap_arrow_ant - head_len * pc1 * anterior_sign
    ant_tri = np.array(
        [
            ap_arrow_ant,
            ant_head_base + head_wid * pc1_perp,
            ant_head_base - head_wid * pc1_perp,
        ]
    )
    tan_color = (0.82, 0.71, 0.55)
    ax3.fill(
        ant_tri[:, 0],
        ant_tri[:, 1],
        color=tan_color,
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
    )

    # Posterior arrowhead
    post_head_base = ap_arrow_post + head_len * pc1 * anterior_sign
    post_tri = np.array(
        [
            ap_arrow_post,
            post_head_base + head_wid * pc1_perp,
            post_head_base - head_wid * pc1_perp,
        ]
    )
    ax3.fill(
        post_tri[:, 0],
        post_tri[:, 1],
        color=tan_color,
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
    )

    # A and P labels
    ap_label_offset = 0.06 * shape_radius
    ax3.text(
        ap_arrow_ant[0] + ap_label_offset * pc1[0] * anterior_sign,
        ap_arrow_ant[1] + ap_label_offset * pc1[1] * anterior_sign,
        "A",
        color=tan_color,
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=10,
    )
    ax3.text(
        ap_arrow_post[0] - ap_label_offset * pc1[0] * anterior_sign,
        ap_arrow_post[1] - ap_label_offset * pc1[1] * anterior_sign,
        "P",
        color=tan_color,
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=10,
    )

    ax3.set_xlim(-display_radius, display_radius)
    ax3.set_ylim(-display_radius, display_radius)
    ax3.set_aspect("equal")
    ax3.set_xlabel("X", color="white", fontsize=10)
    ax3.set_ylabel(
        "Y", color="white", fontsize=10, rotation=0, ha="right", va="center"
    )
    ax3.set_xticks([])
    ax3.set_yticks([])
    for spine in ax3.spines.values():
        spine.set_color("gray")

    # TILE 4: R×M vs GT Accuracy Scatter Plot
    ax4 = fig.add_subplot(gs[1, 1], facecolor="black")

    # Different marker shapes for each individual
    marker_shapes = ["o", "s", "^", "D", "v", "h", "*", "X", "P", "p"]

    # Check if both metrics and accuracy data are present
    has_data = (
        all_individuals_metrics
        and len(all_individuals_metrics) > 0
        and individual_accuracy
        and len(individual_accuracy) > 0
    )

    if has_data:
        # Collect data points for scatter plot
        rxm_values = []
        accuracy_values = []
        ind_labels = []

        for ind in sorted(all_individuals_metrics.keys()):
            R = all_individuals_metrics[ind]["R"]
            M = all_individuals_metrics[ind]["M"]
            rxm = R * M

            if ind in individual_accuracy:
                acc = individual_accuracy[ind]["accuracy"]
                rxm_values.append(rxm)
                accuracy_values.append(acc)
                ind_labels.append(ind)

        if rxm_values:
            # Use file color if provided, otherwise default to gray
            base_color = file_color[:3] if file_color else (0.5, 0.5, 0.5)

            # Find best individual (highest R×M)
            best_idx = np.argmax(rxm_values)

            # Plot each individual with a different marker shape
            # Best individual (highest R×M) always uses star marker
            legend_handles = []
            data = zip(rxm_values, accuracy_values, ind_labels, strict=True)
            for i, (rxm, acc, label) in enumerate(data):
                is_best = i == best_idx
                marker = (
                    "*" if is_best else marker_shapes[i % len(marker_shapes)]
                )
                size = 200 if is_best else 120

                zord = 6 if is_best else 5
                _scatter = ax4.scatter(
                    [rxm],
                    [acc],
                    s=size,
                    c=[base_color],
                    marker=marker,
                    edgecolors="white",
                    linewidths=1.0,
                    alpha=0.9,
                    zorder=zord,
                )

                # Create legend handle (white edge)
                ms = 12 if is_best else 10
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=marker,
                        color="w",
                        markerfacecolor=base_color,
                        markeredgecolor="white",
                        markeredgewidth=1.0,
                        markersize=ms,
                        linestyle="None",
                        label=label,
                    )
                )

            # Set axis limits with padding
            rxm_range = (
                max(rxm_values) - min(rxm_values)
                if len(rxm_values) > 1
                else 0.1
            )
            # Handle case when all values are identical (range = 0)
            if rxm_range < 0.01:
                rxm_range = 0.1
            x_min = min(rxm_values) - rxm_range * 0.15
            x_max = max(rxm_values) + rxm_range * 0.25
            ax4.set_xlim(x_min, x_max)
            # Y-axis: -20 to 120 range, but only show ticks 0-100
            ax4.set_ylim(-20, 120)
            ax4.set_yticks([0, 20, 40, 60, 80, 100])

            # Axis labels
            ax4.set_xlabel("R×M", color="white", fontsize=10)
            ax4.set_ylabel(
                "Pairwise GT Concordance\n(inferred AP, %)",
                color="white",
                fontsize=10,
            )

            # Add legend for individual markers
            ax4.legend(
                handles=legend_handles,
                loc="lower right",
                fontsize=8,
                facecolor="black",
                edgecolor="gray",
                labelcolor="white",
            )

            # Add grid
            ax4.grid(
                True, color="gray", alpha=0.3, linestyle="-", linewidth=0.5
            )
    else:
        ax4.text(
            0.5,
            0.5,
            "No data",
            color="gray",
            fontsize=12,
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )

    ax4.tick_params(axis="y", colors="gray", labelsize=8)
    ax4.tick_params(axis="x", colors="gray", labelsize=8)
    for spine in ax4.spines.values():
        spine.set_color("gray")

    # NODE LEGEND (Center of figure)
    legend_x = 0.47
    legend_top_y = 0.88
    legend_spacing = 0.045

    fig_w, fig_h = fig.get_size_inches()
    fig_aspect = fig_w / fig_h
    dot_h = 0.012
    dot_w = dot_h / fig_aspect

    legend_entry_count = 0
    for i in valid_idx:
        y_pos = legend_top_y - legend_entry_count * legend_spacing

        ax_dot = fig.add_axes(
            [legend_x - dot_w / 2, y_pos - dot_h / 2, dot_w, dot_h]
        )
        ax_dot.set_xlim(0, 1)
        ax_dot.set_ylim(0, 1)
        ax_dot.scatter(
            0.5, 0.5, s=100, c=[colors[i]], edgecolors="black", linewidths=0.5
        )
        ax_dot.axis("off")

        name = keypoint_names[i] if i < len(keypoint_names) else f"node_{i}"
        fig.text(
            legend_x + 0.015,
            y_pos,
            f"[{i}] {name}",
            color="white",
            fontsize=10,
            va="center",
            ha="left",
        )

        legend_entry_count += 1

    # Centroid entry
    y_pos = legend_top_y - legend_entry_count * legend_spacing
    ax_dot = fig.add_axes(
        [legend_x - dot_w / 2, y_pos - dot_h / 2, dot_w, dot_h]
    )
    ax_dot.set_xlim(0, 1)
    ax_dot.set_ylim(0, 1)
    ax_dot.scatter(
        0.5,
        0.5,
        s=100,
        c=[(0.3, 0.3, 0.3)],
        edgecolors="black",
        linewidths=0.5,
    )
    ax_dot.axis("off")
    fig.text(
        legend_x + 0.015,
        y_pos,
        "bbox centroid",
        color="white",
        fontsize=10,
        va="center",
        ha="left",
    )
    legend_entry_count += 1

    # Histogram bar entries (blue = +vel, red = -vel)
    bar_w = 0.012
    bar_h = 0.008

    # Blue bar (+vel projections)
    y_pos = legend_top_y - legend_entry_count * legend_spacing
    ax_bar = fig.add_axes(
        [legend_x - bar_w / 2, y_pos - bar_h / 2, bar_w, bar_h]
    )
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(0, 1)
    ax_bar.add_patch(
        plt.Rectangle(
            (0.1, 0.1),
            0.8,
            0.8,
            facecolor=(0.4, 0.75, 1.0),
            alpha=0.6,
            edgecolor="none",
        )
    )
    ax_bar.axis("off")
    fig.text(
        legend_x + 0.015,
        y_pos,
        "+PC1 vel proj.",
        color="white",
        fontsize=10,
        va="center",
        ha="left",
    )
    legend_entry_count += 1

    # Red bar (-vel projections)
    y_pos = legend_top_y - legend_entry_count * legend_spacing
    ax_bar = fig.add_axes(
        [legend_x - bar_w / 2, y_pos - bar_h / 2, bar_w, bar_h]
    )
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(0, 1)
    ax_bar.add_patch(
        plt.Rectangle(
            (0.1, 0.1),
            0.8,
            0.8,
            facecolor=(1.0, 0.45, 0.35),
            alpha=0.6,
            edgecolor="none",
        )
    )
    ax_bar.axis("off")
    fig.text(
        legend_x + 0.015,
        y_pos,
        "−PC1 vel proj.",
        color="white",
        fontsize=10,
        va="center",
        ha="left",
    )
    legend_entry_count += 1

    # AP Midline (dotted beige line)
    line_w = 0.025
    line_h = 0.006
    y_pos = legend_top_y - legend_entry_count * legend_spacing
    ax_line = fig.add_axes(
        [legend_x - line_w / 2, y_pos - line_h / 2, line_w, line_h]
    )
    ax_line.set_xlim(0, 1)
    ax_line.set_ylim(0, 1)
    ax_line.plot([0.1, 0.9], [0.5, 0.5], ":", color=tan_color, linewidth=2)
    ax_line.axis("off")
    fig.text(
        legend_x + 0.015,
        y_pos,
        "AP midline",
        color="white",
        fontsize=10,
        va="center",
        ha="left",
    )

    # Save figure
    plt.savefig(
        output_path,
        format="svg",
        facecolor="black",
        edgecolor="none",
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def generate_individual_plots(analysis_data, output_dir, timestamp=None):  # noqa: C901
    """Generate detailed 2x2 plot for each file's best individual."""
    data = analysis_data
    stored_skeletons = data.get("skeletons", {})
    stored_pc1 = data.get("pc1_vectors", {})
    stored_vel_projs = data.get("vel_projs_pc1", {})
    keypoints_by_file = data.get("keypoints", {})
    individual_accuracy_by_file = data.get("individual_accuracy", {})

    # File color mapping (matching ap_validation_results figure)
    # Files sorted alphabetically, colors assigned in order
    file_colors = {
        "free-moving-2flies-ID-13nodes-1024x1024x1-30_3pxmm": (
            0.12,
            0.47,
            0.71,
            1.0,
        ),  # Blue
        "free-moving-2mice-noID-5nodes-1280x1024x1-1_9pxmm": (
            0.17,
            0.63,
            0.17,
            1.0,
        ),  # Green
        "free-moving-4gerbils-ID-14nodes-1024x1280x3-2pxmm": (
            0.80,
            0.70,
            0.10,
            1.0,
        ),  # Gold
        "free-moving-5mice-noID-11nodes-1280x1024x1-1_97pxmm": (
            1.00,
            0.50,
            0.05,
            1.0,
        ),  # Orange
        "freemoving-2bees-noID-21nodes-1535x2048x1-14pxmm": (
            0.58,
            0.40,
            0.74,
            1.0,
        ),  # Purple
    }

    # Collect R, M, and R×M values per file per individual
    def default_metrics():
        return {"R": [], "M": [], "RxM": []}

    file_individual_data = defaultdict(lambda: defaultdict(default_metrics))
    for i in range(len(data["file"])):
        file_stem = data["file"][i]
        individual = data["individual"][i]
        R = data["resultant_length"][i]
        M = data["vote_margin"][i]
        rxm = data["rxm"][i]
        if not np.isnan(rxm):
            file_individual_data[file_stem][individual]["R"].append(R)
            file_individual_data[file_stem][individual]["M"].append(M)
            file_individual_data[file_stem][individual]["RxM"].append(rxm)

    # Compute mean R, M per individual and find best by R×M
    best_individual = {}
    best_metrics = {}
    # {file_stem: {individual: {"R": mean_R, "M": mean_M}}}
    all_individuals_metrics_by_file = {}

    for file_stem, individuals in file_individual_data.items():
        best_rxm = -1
        best_ind = None
        all_individuals_metrics_by_file[file_stem] = {}

        for ind, metrics_dict in individuals.items():
            mean_R = np.mean(metrics_dict["R"])
            mean_M = np.mean(metrics_dict["M"])
            mean_rxm = np.mean(metrics_dict["RxM"])
            all_individuals_metrics_by_file[file_stem][ind] = {
                "R": mean_R,
                "M": mean_M,
            }
            if mean_rxm > best_rxm:
                best_rxm = mean_rxm
                best_ind = ind

        best_individual[file_stem] = best_ind

        # Get metrics for best individual
        for i in range(len(data["file"])):
            is_match = (
                data["file"][i] == file_stem
                and data["individual"][i] == best_ind
            )
            if is_match:
                best_metrics[file_stem] = {
                    "vote_margin": data["vote_margin"][i],
                    "resultant_length": data["resultant_length"][i],
                    "anterior_sign": data["anterior_sign"][i],
                    "circ_mean_dir": data["circ_mean_dir"][i],
                    "num_selected_frames": data["num_selected_frames"][i],
                    "n_frames": data["n_frames"][i],
                }
                break

    print("\nGenerating per-file detail plots (best individual)...")

    for file_stem, best_ind in sorted(best_individual.items()):
        label = FILE_LABELS.get(file_stem, file_stem[:15])
        print(f"  {label}: *{best_ind}*")

        has_skel = (
            file_stem in stored_skeletons
            and best_ind in stored_skeletons[file_stem]
        )
        if not has_skel:
            print("    No skeleton data")
            continue

        avg_skeleton = stored_skeletons[file_stem][best_ind]

        has_pc1 = file_stem in stored_pc1 and best_ind in stored_pc1[file_stem]
        if not has_pc1:
            print("    No PC1 data")
            continue

        pc1 = stored_pc1[file_stem][best_ind]
        keypoint_names = keypoints_by_file.get(file_stem, [])
        metrics = best_metrics.get(file_stem, {})

        # Get velocity projections for histogram
        vel_projs = None
        has_vel = (
            file_stem in stored_vel_projs
            and best_ind in stored_vel_projs[file_stem]
        )
        if has_vel:
            vel_projs = stored_vel_projs[file_stem][best_ind]

        # Get R and M data for all individuals in this file
        all_individuals_metrics = all_individuals_metrics_by_file.get(
            file_stem, {}
        )

        # Get individual accuracy data for this file
        individual_accuracy = individual_accuracy_by_file.get(file_stem, {})

        # Get file color
        file_color = file_colors.get(file_stem, (0.5, 0.5, 0.5, 1.0))

        # Sanitize individual name for filename
        safe_ind = best_ind.replace(" ", "_")
        ts_suffix = f"_{timestamp}" if timestamp else ""
        output_path = output_dir / f"{file_stem}_{safe_ind}{ts_suffix}.svg"
        create_individual_plot(
            file_stem,
            avg_skeleton,
            pc1,
            keypoint_names,
            metrics,
            output_path,
            vel_projs,
            individual_name=best_ind,
            all_individuals_metrics=all_individuals_metrics,
            file_color=file_color,
            individual_accuracy=individual_accuracy,
        )

    print(f"Per-file detail plots saved to: {output_dir}")


def main():  # noqa: C901
    ensure_demo_datasets()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    H5_DIR.mkdir(exist_ok=True)

    # Set up log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"ap_validation_{timestamp}.log"

    with TeeOutput(log_file):
        print(f"AP Validation started at {datetime.now().isoformat()}")
        print(f"Log file: {log_file}")
        print()

        # Check for existing H5 files
        existing_h5 = sorted(H5_DIR.glob("ap_validation_*.h5"))

        if existing_h5:
            # Use most recent existing file
            output_path = existing_h5[-1]
            print(f"Found existing H5 file: {output_path.name}")
            print("Skipping generation, proceeding to analysis...")
        else:
            # Generate new H5 file
            slp_files = sorted(SLP_DIR.glob("*.slp"))
            if not slp_files:
                print("No .slp files found after bootstrap")
                sys.exit(1)

            # PASS 1: R×M Selection — Find best individual per file
            n_files = len(slp_files)
            print(
                f"Pass 1: R×M Selection — finding best individual "
                f"per file from {n_files} files..."
            )
            rxm_tasks, file_metadata = generate_rxm_tasks(slp_files)
            print(f"  Running {len(rxm_tasks)} R×M computations...")

            rxm_results = []
            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                futures = [
                    executor.submit(process_single_validation, task)
                    for task in rxm_tasks
                ]
                for future in as_completed(futures):
                    rxm_results.append(future.result())

            best_individuals, all_rxm, file_individual_data = (
                find_best_individuals(rxm_results)
            )
            best_ind_str = ", ".join(
                f"{FILE_LABELS.get(k, k[:15])}: *{v}*"
                for k, v in best_individuals.items()
            )
            print(f"  Best individuals (max R×M): {{{best_ind_str}}}")

            # Compute PC1-based orderings for all individuals
            # (prerequisite for Pass 2: cross-individual consistency)
            best_pc1_orderings, all_pc1_orderings = compute_pc1_orderings(
                file_individual_data, file_metadata, best_individuals
            )

            # PASS 2: Cross-Individual Ordering Consistency
            print(
                "\nPass 2: Cross-Individual Ordering Consistency — "
                "do individuals from the same video agree on node ordering?\n"
                "  Each individual's raw PC1 ordering of GT nodes is compared "
                "against the best individual's ordering (the 'pseudo GT').\n"
                "  This is a CONSISTENCY check, not a correctness check — "
                "high agreement means the body shape is stable across "
                "individuals, but says nothing about whether the ordering "
                "is anatomically correct."
            )
            ordering_matches = compare_orderings_to_pseudo_gt(
                all_pc1_orderings, best_pc1_orderings
            )
            for file_stem, matches in ordering_matches.items():
                n_match = sum(matches.values())
                n_total = len(matches)
                label = FILE_LABELS.get(file_stem, file_stem[:15])
                print(
                    f"  {label}: {n_match}/{n_total} individuals "
                    f"share the best individual's ordering"
                )

            # PASS 3: Inferred AP Concordance per Individual
            # For each individual, project GT nodes onto the inferred
            # AP axis (anterior_sign × PC1) and compare the ordering
            # against hand-curated GT. This tests the full pipeline
            # (PCA + velocity voting) per individual. Feeds Figure 1 Tile 4.
            print(
                "\nPass 3: Inferred AP Concordance — "
                "does each individual's velocity-inferred AP ordering "
                "match hand-curated GT?\n"
                "  For each individual, GT nodes are projected onto "
                "anterior_sign × PC1 (the inferred AP axis). All C(n,2) "
                "unique pairs are tested against the hand-curated GT "
                "ranking.\n"
                "  This tests the full pipeline (PCA + velocity voting) "
                "per individual."
            )
            individual_accuracy = compute_inferred_ap_concordance(
                file_individual_data
            )

            # Build R, M, R×M lookup from Pass 1 results
            ind_metrics = {}
            for rec in rxm_results:
                if not rec.get("error", False):
                    key = (rec["file"], rec["individual"])
                    ind_metrics[key] = {
                        "R": rec.get("resultant_length", float("nan")),
                        "M": rec.get("vote_margin", float("nan")),
                        "rxm": rec.get("rxm", float("nan")),
                    }

            for file_stem, accuracies in individual_accuracy.items():
                print(f"  {FILE_LABELS.get(file_stem, file_stem[:15])}:")
                best_ind = best_individuals.get(file_stem)
                for ind, acc in sorted(accuracies.items()):
                    c = acc["correct"]
                    t = acc["total"]
                    a = acc["accuracy"]
                    marker = "*" if ind == best_ind else ""
                    m = ind_metrics.get((file_stem, ind), {})
                    R = m.get("R", float("nan"))
                    M = m.get("M", float("nan"))
                    rxm = m.get("rxm", float("nan"))
                    print(
                        f"    {marker}{ind}{marker}: "
                        f"{c}/{t} pairs = {a:.1f}%  "
                        f"(R={R:.2f}, M={M:.2f}, R×M={rxm:.2f})"
                    )

            # Generate validation tasks for H5 storage
            print(
                "\nGenerating H5 validation records "
                "(all GT pair permutations × all individuals)..."
            )
            tasks, keypoints_by_file = generate_gt_validation_tasks(
                file_metadata, best_individuals
            )
            n_tasks = len(tasks)
            print(
                f"  Processing {n_tasks} validation comparisons "
                f"with {N_WORKERS} workers..."
            )

            all_results = []
            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                futures = [
                    executor.submit(process_single_validation, task)
                    for task in tasks
                ]

                for i, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    all_results.append(result)
                    if i % 50 == 0:
                        print(f"    {i}/{len(tasks)} completed...")

            h5_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = H5_DIR / f"ap_validation_{h5_timestamp}.h5"
            # Include Pass 1 results to ensure best individual's data is saved
            combined_results = rxm_results + all_results
            save_to_h5(
                combined_results,
                keypoints_by_file,
                output_path,
                individual_accuracy=individual_accuracy,
                all_rxm=all_rxm,
            )

            print(f"\nSaved {len(all_results)} records to: {output_path}")

        # Reporting: GT coverage, suggested pairs, and Figure 2 data
        analysis_data = analyze_results(output_path)
        plot_validation_results(analysis_data, output_path)

        # Generate per-file 2×2 tile detail plots for each best individual
        # Extract timestamp from H5 filename (ap_validation_YYYYMMDD_HHMMSS.h5)
        h5_stem = output_path.stem
        fig_timestamp = "_".join(h5_stem.split("_")[-2:])
        generate_individual_plots(analysis_data, FIGURES_DIR, fig_timestamp)

        print(f"\nLog saved to: {log_file}")


if __name__ == "__main__":
    main()
