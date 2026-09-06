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

COCO_RESULTS_SCHEMA: Mapping[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "COCO Keypoint Results",
    "description": ("Schema for validating COCO keypoint results JSON files"),
    "type": "array",
    "items": {
        "type": "object",
        "required": [
            "image_id",
            "category_id",
            "keypoints",
            "score",
        ],
        "properties": {
            "image_id": {
                "type": "integer",
            },
            "category_id": {
                "type": "integer",
            },
            "keypoints": {
                "type": "array",
                "items": {
                    "type": "number",
                },
            },
            "score": {
                "type": "number",
            },
        },
    },
}

COCO_ANNOTATIONS_SCHEMA: Mapping[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "COCO Keypoint Annotations",
    "description": (
        "Schema for validating COCO keypoint annotations JSON files"
    ),
    "type": "object",
    "required": ["categories"],
    "properties": {
        "categories": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "keypoints"],
                "properties": {
                    "id": {
                        "type": "integer",
                    },
                    "name": {
                        "type": "string",
                    },
                    "keypoints": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                },
            },
        },
    },
}
