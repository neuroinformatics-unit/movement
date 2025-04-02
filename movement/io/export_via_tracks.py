import json


def export_via_tracks(bboxes, output_file):
    """Exports movement bounding boxes dataset to VIA-tracks format.

    Args:
        bboxes (dict): Dictionary containing bbox data (frame, object_id, x, y, width, height).
        output_file (str): Path to save the VIA-tracks JSON file.

    Returns:
        None

    """
    via_tracks_data = {}

    for frame_id, objects in bboxes.items():
        for obj_id, bbox in objects.items():
            if frame_id not in via_tracks_data:
                via_tracks_data[frame_id] = {}

            via_tracks_data[frame_id][obj_id] = {
                "x": bbox["x"],
                "y": bbox["y"],
                "width": bbox["width"],
                "height": bbox["height"],
            }

    # Save to file
    with open(output_file, "w") as f:
        json.dump(via_tracks_data, f, indent=4)

    print(f"Exported {len(bboxes)} frames to {output_file}")
