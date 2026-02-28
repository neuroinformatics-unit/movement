(target-text-formatting-guide)=
# Text formatting guide

When writing or updating documentation for movement (including Python docstrings), please adhere to the following conventions for text formatting. These guidelines help maintain a consistent and professional appearance across the project.

## Monospace formatting

We closely follow the [NumPy documentation style](https://numpydoc.readthedocs.io/en/latest/format.html) regarding monospace formatting.

**ALWAYS use monospace (backticks) for:**
*   Package names when referring to the code: movement, `numpy`, `xarray`
*   Module names: `movement.kinematics`
*   Function names: `compute_speed()`
*   Class names: `PosesDataset`
*   Method names: `.diff()`
*   Parameter names: `in_degrees`
*   File extensions: `.csv`, `.h5`, `.slp`, `.dlc`
*   File paths: `/path/to/file.csv`, `~/.movement`
*   Command-line commands: `pip install movement`
*   Literal values: `None`, `True`, `False`

**Do NOT use monospace for:**
*   General prose references to the software product name. For example, "movement is a toolbox" (plain) vs "import `movement`" (monospace).

### Examples

**Correct:**
> To compute velocity, use `compute_velocity()`. The results will be saved to a `.csv` file in `~/.movement`.
>
> A movement dataset represents tracks...

**Incorrect:**
> To compute velocity, use compute_velocity(). The results will be saved to a .csv file in ~/.movement.
>
> A movement dataset represents tracks...
