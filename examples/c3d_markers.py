"""Process C3D motion capture data, convert it to CSV, and
visualize it using Napari.

"""

import ast  # To parse JSON-like strings

import kineticstoolkit as ktk
import napari
import numpy as np
import pandas as pd


def convert_c3d_to_csv(
    c3d_file: str, output_csv: str, bbox_width: int = 20, bbox_height: int = 20
) -> None:
    """Convert a C3D file to a CSV format for visualization in Napari.

    Args:
        c3d_file (str): Path to the input C3D file.
        output_csv (str): Path to save the output CSV file.
        bbox_width (int, optional): Width of the bounding box. Default is 20.
        bbox_height (int, optional): Height of the bounding box. Default is 20.

    """
    # Load C3D data
    data = ktk.read_c3d(c3d_file)

    # Extract marker positions
    marker_positions = data["Points"]
    marker_names = list(marker_positions.data.keys())  # List of marker names
    all_positions = np.array(
        [marker_positions.data[key] for key in marker_names]
    )  # Convert to array

    # Prepare CSV output
    csv_data = []

    for frame_idx in range(all_positions.shape[1]):  # Iterate over frames
        for marker_idx, marker_name in enumerate(
            marker_names
        ):  # Iterate markers
            xyz = all_positions[marker_idx, frame_idx, :]

            # Extract x, y (ignoring z for 2D visualization)
            if xyz.shape[0] >= 3:
                x, y, _ = xyz[:3]
            else:
                x, y = np.nan, np.nan

            # Skip invalid points
            if np.isnan(x) or np.isnan(y):
                continue

            # Compute bounding box centered at (x, y)
            bbox_x = x - bbox_width / 2
            bbox_y = y - bbox_height / 2

            # Format row for CSV
            row = {
                "filename": f"frame_{frame_idx}.jpg",
                "file_size": 0,
                "file_attributes": "{}",
                "region_count": 1,
                "region_id": marker_idx,
                "region_shape_attributes": (
                    f'{{"name": "rect", "x": {bbox_x}, "y": {bbox_y}, '
                    f'"width": {bbox_width}, "height": {bbox_height}}}'
                ),
                "region_attributes": f'{{"track": "{marker_name}"}}',
            }
            csv_data.append(row)

    # Save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_csv, index=False)

    print(f"Bounding box CSV file successfully generated: {output_csv}")


def visualize_tracks(csv_file: str, marker_size: float = 1.0) -> None:
    """Load CSV tracking data and visualize it in Napari.

    Args:
        csv_file (str): Path to the CSV file containing tracking data.
        marker_size (float, optional): Size of the markers in Napari.
            Default is 1.0.

    """
    # Load CSV tracking data with selected columns
    df = pd.read_csv(
        csv_file,
        usecols=["filename", "region_shape_attributes", "region_attributes"],
    )

    # Extract frame numbers (assuming filenames follow "frame_0.jpg" format)
    df["frame"] = df["filename"].str.extract(r"(\d+)").astype(int)

    # Parse bounding box JSON
    df["bbox"] = df["region_shape_attributes"].apply(ast.literal_eval)

    # Extract bounding box center points
    df["x_center"] = df["bbox"].apply(lambda b: b["x"] + (b["width"] / 2))
    df["y_center"] = df["bbox"].apply(lambda b: b["y"] + (b["height"] / 2))

    # Extract track names
    df["track"] = (
        df["region_attributes"]
        .apply(ast.literal_eval)
        .apply(lambda r: r.get("track", "Unknown"))
    )

    # Group by track
    track_groups = df.groupby("track")

    # Start Napari viewer
    viewer = napari.Viewer()

    # Define colors for different tracks
    colors = [
        "red",
        "blue",
        "green",
        "yellow",
        "magenta",
        "cyan",
        "orange",
        "purple",
    ]

    for i, (track_name, group) in enumerate(track_groups):
        points = np.column_stack(
            (group["frame"], group["y_center"], group["x_center"])
        )
        color = colors[i % len(colors)]  # Cycle through colors
        viewer.add_points(
            points,
            name=track_name,
            size=marker_size,  # Set smaller size
            face_color=color,
            opacity=0.8,
            symbol="o",  # Prevents auto-scaling issues
        )

    # Run Napari
    napari.run()


if __name__ == "__main__":
    # Convert C3D file to CSV
    convert_c3d_to_csv("sample.c3d", "c3d_bboxes.csv")

    # Visualize the converted tracking data
    visualize_tracks(
        "c3d_bboxes.csv", marker_size=0.5
    )  # Adjust marker_size if needed
