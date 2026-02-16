"""Utility functions for JSON schema validation."""

import json
from pathlib import Path

import jsonschema

from movement.utils.logging import logger


def get_schema(schema_name: str) -> dict:
    """Load a JSON schema from the schemas directory.

    Parameters
    ----------
    schema_name : str
        Name of the schema file (without the .json extension).

    Returns
    -------
    dict
        The JSON schema as a dictionary.

    """
    schema_path = Path(__file__).parent / "schemas" / f"{schema_name}.json"
    with open(schema_path) as file:
        return json.load(file)


def check_file_is_json(filepath: Path) -> dict:
    """Check that the file contains valid JSON and return the parsed data.

    Parameters
    ----------
    filepath : pathlib.Path
        Path to the JSON file to validate.

    Returns
    -------
    dict
        The parsed JSON data.

    Raises
    ------
    ValueError
        If the file cannot be parsed as JSON.

    """
    try:
        with open(filepath) as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        raise logger.error(
            ValueError(f"File {filepath} is not valid JSON: {e}")
        ) from e


def check_file_matches_schema(filepath: Path, schema: dict) -> dict:
    """Check that a JSON file matches the given schema.

    Parameters
    ----------
    filepath : pathlib.Path
        Path to the JSON file to validate.
    schema : dict
        The JSON schema to validate against.

    Returns
    -------
    dict
        The parsed JSON data if validation succeeds.

    Raises
    ------
    ValueError
        If the file does not match the schema.

    """
    data = check_file_is_json(filepath)
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as e:
        raise logger.error(
            ValueError(f"File {filepath} does not match schema: {e.message}")
        ) from e
    return data
