(target-testing)=
# Testing
We use [pytest](https://docs.pytest.org/en/latest/) for testing, aiming for ~100% test coverage
where feasible. All new features and bug fixes should be accompanied by tests.

## Running tests

All commands below should be run from the root of the repository, inside your
[development environment](target-creating-a-development-environment).

To run the full test suite with coverage:

```sh
pytest
```

Some useful options for local development:

```sh
pytest --no-cov                          # skip coverage report for a faster run
pytest --cov-report=term-missing         # show which lines are not covered
pytest -v                                # verbose output
pytest tests/test_unit/test_filtering.py # run a specific file
pytest tests/test_unit/test_logging.py::test_log_to_file  # run a specific test
pytest -k "dlc"                          # run tests whose name matches a pattern
```

## Test organisation

Tests live in the `tests/` directory, organised as follows:

- `test_unit/`: Unit tests to verify that individual modules or functions
  work correctly in isolation. Their structure mirrors that of the `movement`
  package: each module typically has a corresponding test file
  (e.g. `movement/filtering.py` → `tests/test_unit/test_filtering.py`).
- `test_integration/`: Integration tests to verify that different modules
  or functions work properly when combined or chained together in a workflow.
- `fixtures/`: Reusable fixtures auto-loaded by pytest via `conftest.py`. Check here before
  adding new fixtures to avoid duplication.

### Fixtures and test data

Pytest fixtures are used to set up test state, such as setup and teardown, and provide frequently test data.
All files in `tests/fixtures/` are automatically discovered and loaded by pytest via
`conftest.py` — no imports needed in test files. The key fixture files are:

- `datasets.py`: Synthetic {class}`xarray.Dataset` objects with known trajectories (e.g. linear
  motion with predictable velocities), useful for testing numerical correctness without real
  tracking data.
- `files.py`: File path fixtures for sample data, accessed via `pytest.DATA_PATHS`. Use
  `tmp_path` (a built-in pytest fixture) for any temporary files your test needs to create.
- `helpers.py`: A `Helpers` class fixture with custom assertion methods such as
  `assert_valid_dataset()` and `count_nans()`.

For tests requiring real experimental data, use the [sample data](target-contributing-sample-data)
accessed through `pytest.DATA_PATHS`:

```python
def test_load_dlc_file():
    file_path = pytest.DATA_PATHS.get("DLC_single-wasp.predictions.h5")
    ds = load_poses.from_dlc_file(file_path)
    ...
```

Avoid including large data files directly in the GitHub repository.

## Writing tests

### Naming conventions

- Test files: `test_<module>.py`
- Test functions: `test_<what_is_being_tested>()`
- Test classes: `Test<Feature>` (sometimes used to group related tests,
  e.g. all tests for a particular function or workflow)

### Parametrise

Use the `@pytest.mark.parametrize` decorator to run the same test logic
across multiple inputs, rather than writing separate test functions.
A key benefit is that when a test fails, pytest reports exactly which parameter
combination caused the failure — making it much easier to diagnose the problem.
Adding a human-readable `id` to each case via `pytest.param` makes that output
even clearer:

```python
@pytest.mark.parametrize(
    "window, expected_nans",
    [
        pytest.param(3, 0, id="window-3"),
        pytest.param(5, 2, id="window-5"),
    ],
)
def test_rolling_filter(window, expected_nans, valid_poses_dataset):
    ...
```


### Parametrising across fixtures

When you want to run the same test against multiple fixtures (e.g. both
`valid_poses_dataset` and `valid_bboxes_dataset`), pass the fixture names as string
parameters and retrieve them dynamically using `request.getfixturevalue()`:

```python
@pytest.mark.parametrize(
    "dataset_fixture",
    [
        pytest.param("valid_poses_dataset", id="poses"),
        pytest.param("valid_bboxes_dataset", id="bboxes"),
    ],
)
def test_filter_returns_dataset(dataset_fixture, request):
    dataset = request.getfixturevalue(dataset_fixture)
    result = filter_by_confidence(dataset, threshold=0.5)
    assert isinstance(result, xr.Dataset)
```

The built-in `request` fixture provides access to the test context;
`getfixturevalue()` looks up and evaluates the named fixture at runtime.

### Testing valid and invalid inputs together

Use {func}`contextlib.nullcontext` (imported as `does_not_raise`) alongside
{func}`pytest.raises` to parametrise both valid and invalid inputs in a single test:

```python
from contextlib import nullcontext as does_not_raise

@pytest.mark.parametrize(
    "kwargs, expected_exception",
    [
        pytest.param({"window": 3}, does_not_raise(), id="valid"),
        pytest.param({"window": -1}, pytest.raises(ValueError), id="negative-window"),
    ],
)
def test_rolling_filter_kwargs(kwargs, expected_exception, valid_poses_dataset):
    with expected_exception:
        rolling_filter(valid_poses_dataset, **kwargs)
```

Where an exception is expected, use the `match` argument to assert on a substring of
the error message. This both documents the expected message and prevents the test from
passing silently if a different `ValueError` is raised for an unrelated reason:

```python
pytest.raises(ValueError, match="window must be a positive integer")
```

### Asserting on xarray outputs

Use {func}`xarray.testing.assert_allclose` to compare datasets or data arrays:

```python
xr.testing.assert_allclose(result, expected)
```

For checking that a dataset conforms to the movement dataset specification, use the
`assert_valid_dataset()` method from the `helpers` fixture:

```python
def test_load_produces_valid_dataset(helpers):
    ds = load_poses.from_dlc_file(pytest.DATA_PATHS.get("DLC_single-wasp.predictions.h5"))
    helpers.assert_valid_dataset(ds, {
        "source_software": "DeepLabCut",
        "fps": 30,
    })
```

## Napari plugin tests

Tests for the napari plugin live in `tests/test_unit/test_napari_plugin/` and follow the
general patterns described above, with a few napari-specific additions. See also the
[napari plugin testing guide](https://napari.org/stable/plugins/testing_and_publishing/test.html#plugin-test)
for broader context.

### Viewer fixture

Use `make_napari_viewer_proxy` from napari's built-in test support to create a headless
viewer instance:

```python
def test_something(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    # interact with the viewer...
```

The viewer is torn down automatically after each test.
Napari-specific fixtures are defined in `tests/fixtures/napari.py` and auto-loaded like all other fixtures.

### Mocking widget interactions

Use `mocker` (from [pytest-mock](https://pytest-mock.readthedocs.io/)) to patch file dialogs
and widget methods at I/O boundaries, keeping the rest of the interaction real:

```python
def test_load_button(mocker, make_napari_viewer_proxy):
    mocker.patch(
        "movement.napari.loader_widgets.QFileDialog.getOpenFileName",
        return_value=(str(file_path), None),
    )
    mocker.patch("movement.napari.loader_widgets.show_error")
    widget = DataLoader(make_napari_viewer_proxy())
    widget._on_load_clicked()
    mock_show_error.assert_not_called()
```

### Separating connection tests from method tests

Keep tests that verify a UI action is wired up correctly separate from tests that verify
what the connected method actually does.

A connection test mocks the handler, triggers the action,
and asserts the mock was called — it says nothing about the handler's behaviour:

```python
def test_load_button_calls_handler(mocker, make_napari_viewer_proxy):
    mock_handler = mocker.patch.object(DataLoader, "_on_load_clicked")
    widget = DataLoader(make_napari_viewer_proxy())  # constructed after patching
    widget.load_button.click()
    mock_handler.assert_called_once()
```

A method test calls the handler directly with controlled inputs and checks the resulting
state — no button clicking needed:

```python
def test_on_load_clicked_adds_layer(make_napari_viewer_proxy, valid_poses_path_and_ds):
    file_path, _ = valid_poses_path_and_ds
    widget = DataLoader(make_napari_viewer_proxy())
    widget._on_load_clicked(file_path)
    assert len(widget.viewer.layers) == 1
```

This separation keeps each test focused and makes failures easier to interpret: a broken
connection test points to a missing signal hookup, while a broken method test points to
the logic inside the handler.

:::{important}
Instantiate widgets inside each test function rather than via a shared fixture.
Mocking a method that has already been connected to a signal at widget construction time
will not intercept calls to it — the signal holds a reference to the original, unmocked
method. Constructing the widget after patching ensures the mock is in place when the
connection is made.
:::

## Running benchmark tests
Some tests are marked as `benchmark` because we use them along with [pytest-benchmark](pytest-benchmark:) to measure the performance of a section of the code. These tests are excluded from the default test run to keep CI and local test running fast.
This applies to all ways of running `pytest` (via command line, IDE, tox or CI).

To run only the benchmark tests locally:

```sh
pytest -m benchmark
```

To run all tests, including those marked as `benchmark`:

```sh
pytest -m ""
```

## Comparing benchmark runs across branches

To compare performance between branches (e.g., `main` and a PR branch), we use [pytest-benchmark](pytest-benchmark:)'s save and compare functionality:

1. Run benchmarks on the `main` branch and save the results:

    ```sh
    git checkout main
    pytest -m benchmark --benchmark-save=main
    ```
    By default the results are saved to `.benchmarks/` (a directory ignored by git) as JSON files with the format `<machine-identifier>/0001_main.json`, where `<machine-identifier>` is a directory whose name relates to the machine specifications, `0001` is a counter for the benchmark run, and `main` corresponds to the string passed in the `--benchmark-save` option.

2. Switch to your PR branch and run the benchmarks again:

    ```sh
    git checkout pr-branch
    pytest -m benchmark --benchmark-save=pr
    ```

3. Show the results from both runs together:

    ```sh
    pytest-benchmark compare <path-to-main-result.json> <path-to-pr-result.json> --group-by=name
    ```
    Instead of providing the paths to the results, you can also provide the identifiers of the runs (e.g. `0001_main` and `0002_pr`), or use glob patterns to match the results (e.g. `*main*` and `*pr*`).

    You can sort the results by the name of the run using the `--sort='name'`, or group them with the `--group-by=<label>` option (e.g. `group-by=name` to group by the name of the run, `group-by=func` to group by the name of the test function, or `group-by=param` to group by the parameters used to test the function). For further options, check the [comparison CLI documentation](pytest-benchmark:usage.html#comparison-cli).

We recommend reading the [pytest-benchmark documentation](pytest-benchmark:) for more information on the available [CLI arguments](pytest-benchmark:usage.html#commandline-options). Some useful options are:
- `--benchmark-warmup=on`: to enable warmup to prime caches and reduce variability between runs. This is recommended for tests involving I/O or external resources.
- `--benchmark-warmup-iterations=N`: to set the number of warmup iterations.
- `--benchmark-compare`: to run benchmarks and compare against the last saved run.
- `--benchmark-min-rounds=10`: to run more rounds for stable results.

:::{note}
High standard deviation in benchmark results often indicates bad isolation or non-deterministic behaviour (I/O, side-effects, garbage collection overhead). Before comparing past runs, it is advisable to make the benchmark runs as consistent as possible. See the [pytest-benchmark guidance on comparing runs](pytest-benchmark:comparing.html) and the [pytest-benchmark FAQ](pytest-benchmark:faq.html) for troubleshooting tips.
:::
