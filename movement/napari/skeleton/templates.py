"""Pre-defined skeleton templates for common animals."""

from typing import TypedDict


class ConnectionTemplate(TypedDict):
    """Type definition for a connection template entry."""

    start: str
    end: str
    color: str
    width: float
    segment: str


class SkeletonTemplate(TypedDict):
    """Type definition for a complete skeleton template."""

    keypoints: list[str]
    connections: list[ConnectionTemplate]


# Mouse skeleton based on the GitHub issue #693 example
MOUSE_TEMPLATE: SkeletonTemplate = {
    "keypoints": [
        "nose",
        "ear_left",
        "ear_right",
        "neck",
        "hip_left",
        "hip_right",
        "tail_base",
    ],
    "connections": [
        {
            "start": "nose",
            "end": "ear_left",
            "color": "#0066FF",  # Blue - head
            "width": 2.0,
            "segment": "head",
        },
        {
            "start": "nose",
            "end": "ear_right",
            "color": "#0066FF",  # Blue - head
            "width": 2.0,
            "segment": "head",
        },
        {
            "start": "ear_left",
            "end": "neck",
            "color": "#00CC66",  # Green - body
            "width": 3.0,
            "segment": "body",
        },
        {
            "start": "ear_right",
            "end": "neck",
            "color": "#00CC66",  # Green - body
            "width": 3.0,
            "segment": "body",
        },
        {
            "start": "hip_left",
            "end": "neck",
            "color": "#00CC66",  # Green - body
            "width": 3.0,
            "segment": "body",
        },
        {
            "start": "hip_right",
            "end": "neck",
            "color": "#00CC66",  # Green - body
            "width": 3.0,
            "segment": "body",
        },
        {
            "start": "hip_left",
            "end": "tail_base",
            "color": "#FF6600",  # Orange - tail
            "width": 2.0,
            "segment": "tail",
        },
        {
            "start": "hip_right",
            "end": "tail_base",
            "color": "#FF6600",  # Orange - tail
            "width": 2.0,
            "segment": "tail",
        },
    ],
}

# Rat skeleton - similar to mouse but with additional keypoints
RAT_TEMPLATE: SkeletonTemplate = {
    "keypoints": [
        "nose",
        "ear_left",
        "ear_right",
        "neck",
        "shoulder_left",
        "shoulder_right",
        "hip_left",
        "hip_right",
        "tail_base",
        "tail_mid",
    ],
    "connections": [
        # Head connections
        {
            "start": "nose",
            "end": "ear_left",
            "color": "#0066FF",
            "width": 2.0,
            "segment": "head",
        },
        {
            "start": "nose",
            "end": "ear_right",
            "color": "#0066FF",
            "width": 2.0,
            "segment": "head",
        },
        {
            "start": "ear_left",
            "end": "neck",
            "color": "#0066FF",
            "width": 2.0,
            "segment": "head",
        },
        {
            "start": "ear_right",
            "end": "neck",
            "color": "#0066FF",
            "width": 2.0,
            "segment": "head",
        },
        # Body connections
        {
            "start": "neck",
            "end": "shoulder_left",
            "color": "#00CC66",
            "width": 3.0,
            "segment": "body",
        },
        {
            "start": "neck",
            "end": "shoulder_right",
            "color": "#00CC66",
            "width": 3.0,
            "segment": "body",
        },
        {
            "start": "shoulder_left",
            "end": "hip_left",
            "color": "#00CC66",
            "width": 3.0,
            "segment": "body",
        },
        {
            "start": "shoulder_right",
            "end": "hip_right",
            "color": "#00CC66",
            "width": 3.0,
            "segment": "body",
        },
        # Tail connections
        {
            "start": "hip_left",
            "end": "tail_base",
            "color": "#FF6600",
            "width": 2.0,
            "segment": "tail",
        },
        {
            "start": "hip_right",
            "end": "tail_base",
            "color": "#FF6600",
            "width": 2.0,
            "segment": "tail",
        },
        {
            "start": "tail_base",
            "end": "tail_mid",
            "color": "#FF6600",
            "width": 2.0,
            "segment": "tail",
        },
    ],
}

# Dictionary of all templates
TEMPLATES: dict[str, SkeletonTemplate] = {
    "mouse": MOUSE_TEMPLATE,
    "rat": RAT_TEMPLATE,
}
