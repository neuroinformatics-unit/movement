(target-implementing-loaders)=
# Implementing new loaders

This guide explains how to implement a new loader to support additional [file formats](target-supported-formats) in `movement`.

## Overview
The `movement` package uses a loader-registry system to support multiple data formats.
Implementing a new loader generally involves the following steps:

1. Creating validator classes for the file format (optional but recommended)
2. Implementing the loader function
3. Updating the `SourceSoftware` type alias

## Create file validators
File validators live in {mod}`movement.validators.files` and ensure that an input file meets the required file access permissions and expected format requirements.
These checks may include verifying the file extension, confirming that the file exists and is readable, or performing format-specific validation (e.g. checking for required datasets in an HDF5 file).
This step is optional, but we strongly recommend implementing at least basic validation so that input files can be accessed reliably.

All validators must implement the {class}`ValidFile<movement.validators.files.ValidFile>` protocol.
At minimum, this requires defining:

- `suffixes`: the expected file extensions for the format
- `file`: the path to the file or an {class}`NWBFile<pynwb.file.NWBFile>` object (if applicable) to validate

Validator classes in `movement` are implemented using the [`attrs`](attrs:) library.
A common pattern is to use an `attrs` {ref}`converter<attrs:converters>` to normalise the input file to a {class}`Path<pathlib.Path>` object, along with one or more validators to ensure the file meets the expected criteria.

In addition to the built-in `attrs` {mod}`validators<attrs.validators>`, `movement` provides several reusable file-specific validators (as callables) in {mod}`movement.validators.files`, including:

- `_file_validator`: a composite validator ensuring that `file` is a {class}`Path<pathlib.Path>`, is not a directory, is accessible with the required permission, and (optionally) has one of the expected `suffixes`
- `_hdf5_validator`: checks that an HDF5 `file` contains the expected dataset(s)
- `_if_instance_of`: conditionally applies a validator only when `file` is an instance of a given class

These validators can be combined using either {func}`attrs.validators.and_` or by passing a list of validators to the `validator` parameter of {func}`field()<attrs.field>`.

For example, the {class}`ValidDeepLabCutH5<movement.validators.files.ValidDeepLabCutH5>` validator—which checks that the input file is a readable DeepLabCut .h5 file containing the `df_with_missing` dataset—is implemented as follows:
```python
@define
class ValidDeepLabCutH5:
    """Class for validating DeepLabCut-style .h5 files."""

    suffixes: ClassVar[set[str]] = {".h5"}
    file: Path = field(
        converter=Path,
        validator=validators.and_(
            _file_validator(permission="r", suffixes=suffixes),
            _hdf5_validator(datasets={"df_with_missing"}),
        ),
    )
```

:::{seealso}
- {external+attrs:std:doc}`examples`: Overview of writing `attrs` classes
- {ref}`attrs Validators<attrs:validators>`: Details on writing custom validators for attributes
:::

## Implement the loader function

Loader functions live in {mod}`movement.io.load_poses` or {mod}`movement.io.load_bboxes`, depending on the data type.

A loader function must:

- accept `file` as its first parameter (either a `str`, a {class}`Path<pathlib.Path>`, or an {class}`NWBFile<pynwb.file.NWBFile>` object)
- return an {class}`xarray.Dataset<xarray.Dataset>` object containing the [movement dataset](target-poses-and-bboxes-dataset)

The {func}`@register_loader()<movement.io.load.register_loader>` decorator registers the loader function so it can be invoked via the unified {func}`from_file()<movement.io.load.from_file>` interface based on the `source_software` argument.
If a file validator is provided, the decorator validates the input `file` before calling the loader.
In this case, the loader receives the validated file object rather than the raw input.
If no validator is provided, the loader receives the raw `file` argument.

Many software packages produce multiple file formats (e.g. CSV and HDF5).
In such cases, we recommend implementing a single loader function that dispatches internally based on file extension or content, ensuring a consistent entry point for each supported source software.
If the formats require very different validation logic, you may pass multiple validators to the `file_validators` parameter.
The decorator will select the appropriate validator based on the input file's suffix and the validator's declared `suffixes`.

Within the loader function, parse the file, extract the relevant arrays, and construct the movement dataset via {func}`movement.io.load_poses.from_numpy` or {func}`movement.io.load_bboxes.from_numpy`.

A simplified example for a hypothetical format "MySoftware" is shown below:
```python
@register_loader(
    source_software="MySoftware",
    file_validators=ValidMySoftwareCSV,
)
def from_mysoftware_file(file: str | Path) -> xr.Dataset:
    """Load data from MySoftware files."""
    # The decorator returns an instance of ValidMySoftwareCSV
    # so we need to let the type checker know this
    valid_file = cast("ValidMySoftwareCSV", file)
    file_path = valid_file.file  # Path
    # The _parse_* functions are pseudocode
    ds = load_poses.from_numpy(
        position_array= _parse_positions(file_path),
        confidence_array=_parse_confidences(file_path),
        individual_names=_parse_individual_names(file_path),
        keypoint_names=_parse_keypoint_names(file_path),
        fps=_parse_fps(file_path),
        source_software="MySoftware",
    )
    logger.info(f"Loaded poses from {file_path.name}")
    return ds
```

## Update SourceSoftware type alias
The `SourceSoftware` type alias is defined in {mod}`movement.io.load` as a `Literal` containing all supported source software names.
When adding a new loader, update this type alias to include the new software name to maintain type safety across the codebase.
