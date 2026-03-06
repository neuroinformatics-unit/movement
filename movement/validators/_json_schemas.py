"""JSON schemas for validating movement-compatible files."""

from collections.abc import Mapping
from typing import Any

# Mapping of RoI types to their corresponding valid GeoJSON geometry types.
ROI_TYPE_TO_GEOMETRY: Mapping[str, tuple[str, ...]] = {
    "PolygonOfInterest": ("Polygon",),
    "LineOfInterest": ("LineString", "LinearRing"),
}

# JSON schema for movement-compatible RoI GeoJSON collections.
ROI_COLLECTION_SCHEMA: Mapping[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "RoI Collection GeoJSON",
    "description": (
        "Schema for validating GeoJSON FeatureCollection files containing RoIs"
    ),
    "type": "object",
    "required": ["type", "features"],
    "properties": {
        "type": {"const": "FeatureCollection"},
        "features": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "geometry"],
                "properties": {
                    "type": {"const": "Feature"},
                    "geometry": {
                        "type": "object",
                        "required": ["type", "coordinates"],
                        "properties": {
                            "type": {
                                "enum": [
                                    "Polygon",
                                    "LineString",
                                    "LinearRing",
                                ]
                            },
                            "coordinates": {"type": "array"},
                        },
                    },
                    "properties": {
                        "oneOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "roi_type": {
                                        "enum": [
                                            "PolygonOfInterest",
                                            "LineOfInterest",
                                        ]
                                    },
                                },
                            },
                        ]
                    },
                },
            },
        },
    },
}

# JSON schema for validating MMPose output JSON files.
MMPOSE_SCHEMA: Mapping[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "MMPose Output",
    "description": "Schema for validating MMPose output JSON files",
    "oneOf": [
        # Single frame object
        {
            "type": "object",
            "required": ["frame_id", "instances"],
            "properties": {
                "frame_id": {"type": "integer"},
                "instances": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["keypoints", "keypoint_scores"],
                        "properties": {
                            "keypoints": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 2,
                                    "maxItems": 2,
                                },
                            },
                            "keypoint_scores": {
                                "type": "array",
                                "items": {"type": "number"},
                            },
                            "bbox": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4,
                            },
                            "bbox_score": {"type": "number"},
                            "track_id": {"type": "integer"},
                        },
                    },
                },
            },
        },
        # Array of frame objects
        {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["frame_id", "instances"],
                "properties": {
                    "frame_id": {"type": "integer"},
                    "instances": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["keypoints", "keypoint_scores"],
                            "properties": {
                                "keypoints": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "minItems": 2,
                                        "maxItems": 2,
                                    },
                                },
                                "keypoint_scores": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                },
                                "bbox": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 4,
                                    "maxItems": 4,
                                },
                                "bbox_score": {"type": "number"},
                                "track_id": {"type": "integer"},
                            },
                        },
                    },
                },
            },
        },
    ],
}
