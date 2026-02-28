(target-implementing-loaders)=
# Implementing new loaders

Implementing a new loader to support additional [file formats](target-supported-formats) in `movement` involves the following steps:

1. Create validator classes for the file format (recommended).
2. Implement the loader function.
3. Update the `SourceSoftware` type alias.

## Create file validators
`movement` enforces separation of concerns by decoupling file validation from data loading, so that loaders can focus solely on reading and parsing data, while validation logic is encapsulated in dedicated file validator classes.
Besides allowing users to get early feedback on file issues, this also makes it easier to reuse validation logic across different loaders that may support the same file format.

All file validators are [`attrs`](attrs:)-based classes and live in {mod}`movement.validators.files`.
They define the rules an input file must satisfy before it can be loaded, and they conform to the {class}`ValidFile<movement.validators.files.ValidFile>` protocol.
At minimum, this requires defining:

- `suffixes`: The expected file extensions for the format.
- `file`: The path to the file or an {class}`NWBFile<pynwb.file.NWBFile>` object, depending on the loader.

Additional attributes can also be defined to store pre-parsed information that the loader may need later.

Using a hypothetical format "MySoftware" that produces CSV files containing the columns `scorer`, `bodyparts`, and `coords`, we illustrate the full pattern file validators follow:

- Declare expected file suffixes.
- Normalise the input file and apply reusable validators.
- Implement custom, format-specific validation.

```python
@define
class ValidMySoftwareCSV:
    """Validator for MySoftware .csv output files."""
    suffixes: ClassVar[set[str]] = {".csv"}
    file: Path = field(
        converter=Path,
        validator=_file_validator(permission="r", suffixes=suffixes),
    )
    col_names: list[str] = field(init=False, factory=list)

    @file.validator
    def _file_contains_expected_header(self, attribute, value):
        """Ensure that the .csv file contains the expected header row.
        """
        expected_cols = ["scorer", "bodyparts", "coords"]
        with open(value) as f:
            col_names = f.readline().split(",")[:3]
            if col_names != expected_cols:
                raise logger.error(
                    ValueError(
                        ".csv header row does not match the known format for "
                        "MySoftware output files."
                    )
                )
            self.col_names = col_names
```

### Declare expected file suffixes
The `suffixes` class variable restricts the validator to only accept files with the specified extensions.
If a suffix check is not required, this can be set to an empty set (`set()`).
In the `ValidMySoftwareCSV` example, only files with a `.csv` extension are accepted.

### Normalise input file and apply reusable validators
An `attrs` {ref}`converter<attrs:converters>` is typically used to normalise input files into {class}`Path<pathlib.Path>` objects, along with one or more validators to ensure the file meets the expected criteria.

In addition to the built-in `attrs` {mod}`validators<attrs.validators>`, `movement` provides several reusable file-specific validators (as callables) in {mod}`movement.validators.files`:

- `_file_validator`: A composite validator that ensures `file` is a {class}`Path<pathlib.Path>`, is not a directory, is accessible with the required permission, and has one of the expected `suffixes` (if any).
- `_hdf5_validator`: Checks that an HDF5 `file` contains the expected dataset(s).
- `_if_instance_of`: Conditionally applies a validator only when `file` is an instance of a given class.

In the current example, the `_file_validator` is used to ensure that the input `file` is a readable CSV file.

:::{dropdown} Combining reusable validators
:color: success
:icon: light-bulb

Reusable validators can be combined using either {func}`attrs.validators.and_` or by passing a list of validators to the `validator` parameter of {func}`field()<attrs.field>`.
The `file` attribute in {class}`ValidDeepLabCutH5<movement.validators.files.ValidDeepLabCutH5>` combines both `_file_validator` and `_hdf5_validator` to ensure the input file is a readable HDF5 file containing the expected dataset `df_with_missing`:

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
:::

### Implement format-specific validation
Most formats often require custom validation logic beyond basic file checks.
In the current example, the `_file_contains_expected_header` method uses the `file` attribute's validator method as a decorator (`@file.validator`) to check that the first line of the CSV file matches the expected header row for MySoftware output files.

:::{seealso}
- {external+attrs:std:doc}`examples`: Overview of writing `attrs` classes.
- {ref}`attrs Validators<attrs:validators>`: Details on writing custom validators for attributes.
:::

## Implement loader function
Once the file validator is defined, the next step is to implement the loader function that reads the validated file and constructs the movement dataset.
Continuing from the hypothetical "MySoftware" example, the loader function `from_mysoftware_file` would look like this:

```python
@register_loader(
    source_software="MySoftware",
    file_validators=ValidMySoftwareCSV,
)
def from_mysoftware_file(file: str | Path) -> xr.Dataset:
    """Load data from MySoftware files."""
    # The decorator returns an instance of ValidMySoftwareCSV
    # which conforms to the ValidFile protocol
    # so we need to let the type checker know this
    valid_file = cast("ValidFile", file)
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

Loader functions live in {mod}`movement.io.load_poses` or {mod}`movement.io.load_bboxes`, depending on the data type (poses or bounding boxes).

A loader function must conform to the {class}`LoaderProtocol<movement.io.load.LoaderProtocol>`, which requires the loader to:

- Accept `file` as its first parameter, which may be:
    - A `str` or a {class}`Path<pathlib.Path>`.
    - An {class}`NWBFile<pynwb.file.NWBFile>` object (for NWB-based formats).
- Return an {class}`xarray.Dataset<xarray.Dataset>` object containing the [movement dataset](target-poses-and-bboxes-dataset).

### Decorate the loader with `@register_loader`
The {func}`@register_loader()<movement.io.load.register_loader>` decorator associates a loader function with a `source_software` name so that users can load files from that software via the unified {func}`load_dataset()<movement.io.load.load_dataset>` interface:
```python
from movement.io import load_dataset
ds = load_dataset("path/to/mysoftware_output.csv", source_software="MySoftware")
```

which is equivalent to calling the loader function directly:
```python
from movement.io.load_poses import from_mysoftware_file
ds = from_mysoftware_file("path/to/mysoftware_output.csv")
```

If a `file_validators` argument is supplied to the {func}`@register_loader()<movement.io.load.register_loader>` decorator, the decorator selects the appropriate validator&mdash;based on its declared `suffixes`&mdash;and uses it to normalise and validate the input `file` before invoking the loader.
As a result, the loader receives the validated file object instead of the raw path or handle.

If no validator is provided, the loader is passed the raw `file` argument as-is.

:::{dropdown} Handling multiple file formats for the same software
:color: success
:icon: light-bulb

Many software packages produce multiple file formats (e.g. DeepLabCut outputs both CSV and HDF5).
In that case, we recommend **one loader per source software**, which internally dispatches to per-format parsing functions, to ensure a consistent entry point for each supported source software.
If formats require very different validation logic, you may pass multiple validators to `file_validators=[...]`.
The decorator will select the appropriate validator based on file suffix and the validator's `suffixes` attribute.

```python
@register_loader(
    source_software="MySoftware",
    file_validators=[ValidMySoftwareCSV, ValidMySoftwareH5],
)
def from_mysoftware_file(file: str | Path) -> xr.Dataset:
    """Load data from MySoftware files (CSV or HDF5)."""
    ...
```
:::

### Construct the dataset
After parsing the input file, the loader function should construct the movement dataset using:

- {func}`movement.io.load_poses.from_numpy` for pose tracks.
- {func}`movement.io.load_bboxes.from_numpy` for bounding box tracks.

These helper functions create the {class}`xarray.Dataset<xarray.Dataset>` object from numpy arrays and metadata, ensuring that the dataset conforms to the [movement dataset specification](target-poses-and-bboxes-dataset).

## Update SourceSoftware type alias
The `SourceSoftware` type alias is defined in {mod}`movement.io.load` as a `Literal` containing all supported source software names.
When adding a new loader, update this type alias to include the new software name to maintain type safety across the codebase:

```python
SourceSoftware: TypeAlias = Literal[
    "DeepLabCut",
    "SLEAP",
    ...,
    "MySoftware",  # Newly added software
]
```
