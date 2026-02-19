(target-testing)=
# Testing
We use [pytest](https://docs.pytest.org/en/latest/) for testing, aiming for ~100% test coverage where feasible. All new features should be accompanied by tests.

Tests are stored in the `tests` directory, structured as follows:

- `test_unit/`: Contains unit tests that closely follow the `movement` package structure.
- `test_integration/`: Includes tests for interactions between different modules.
- `fixtures/`: Holds reusable test data fixtures, automatically imported via `conftest.py`. Check for existing fixtures before adding new ones, to avoid duplication.

For tests requiring experimental data, you can use [sample data](target-contributing-sample-data) from our external data repository.
These datasets are accessible through the `pytest.DATA_PATHS` dictionary, populated in `conftest.py`.
Avoid including large data files directly in the GitHub repository.

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
