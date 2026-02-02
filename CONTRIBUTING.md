# How to Contribute

## Before you start

Before starting work on a contribution, please check the [issue tracker](movement-github:issues) to see if there's already an issue describing what you have in mind.

- If there is, add a comment to let others know you're willing to work on it.
- If there isn't, please create a new issue to describe your idea.

We strongly encourage discussing your plans before you start coding—either in the issue itself or on our [Zulip chat](movement-zulip:).
This helps avoid duplicated effort and ensures your work aligns with the project's [scope](target-mission) and [roadmap](target-roadmaps).

Keep in mind that we use issues liberally to track development.
Some may be vague or aspirational, serving as reminders for future work rather than tasks ready to be tackled.
There are a few reasons an issue might not be actionable yet:

- It depends on other issues being resolved first.
- It hasn't been clearly scoped. In such cases, helping to clarify the scope or breaking the issue into smaller parts can be a valuable contribution. Maintainers typically lead this process, but you're welcome to participate in the discussion.
- It doesn't currently fit into the roadmap or the maintainers' priorities, meaning we may be unable to commit to timely guidance and prompt code reviews.

If you're unsure whether an issue is ready to work on, just ask!

Some issues are labelled as ``good first issue``.
These are especially suitable if you're new to the project, and we recommend starting there.

## Contributing code

### Creating a development environment

In order to make changes to `movement`, you will need to fork the [repository](movement-github:).
If you are not familiar with `git`, we recommend reading up on [this guide](https://docs.github.com/en/get-started/using-git/about-git#basic-git-commands).

1. Clone the forked repository to your local machine and change directory:

    ```sh
    git clone https://github.com/<your-github-username>/movement.git
    cd movement
    ```

2. Set the upstream remote to the base `movement` repository:

    ```sh
    git remote add upstream https://github.com/neuroinformatics-unit/movement.git
    ```

3. Create an environment using [conda](conda:) or [uv](uv:getting-started/installation/) and install `movement` in editable mode, including development dependencies.

    ::::{tab-set}

    :::{tab-item} conda
    First, create and activate a `conda` environment:

    ```sh
    conda create -n movement-dev -c conda-forge python=3.13
    conda activate movement-dev
    ```

    Then, install the package in editable mode with development dependencies:

    ```sh
    pip install -e ".[dev]"
    ```
    :::

    :::{tab-item} uv

    First, create and activate a [virtual environment](uv:pip/environments/):

    ```sh
    uv venv --python=3.13
    source .venv/bin/activate  # On macOS and Linux
    .venv\Scripts\activate     # On Windows PowerShell
    ```

    Then, install the package in editable mode with development dependencies:

    ```sh
    uv pip install -e ".[dev]"
    ```
    :::

    ::::
    If you also want to edit the documentation and preview the changes locally, you will additionally need the `docs` extra dependencies. See [Editing the documentation](#editing-the-documentation) for more details.

4. Finally, initialise the [pre-commit hooks](#formatting-and-pre-commit-hooks):

    ```sh
    pre-commit install
    ```

### Pull requests
In all cases, please submit code to the main repository via a pull request (PR).
We recommend, and adhere, to the following conventions:

- Please submit _draft_ PRs as early as possible to allow for discussion.
- The PR title should be descriptive e.g. "Add new function to do X" or "Fix bug in Y".
- The PR description should be used to provide context and motivation for the changes.
  - If the PR is solving an issue, please add the issue number to the PR description, e.g. "Fixes #123" or "Closes #123".
  - Make sure to include cross-links to other relevant issues, PRs and Zulip threads, for context.
- The maintainers triage PRs and assign suitable reviewers using the GitHub review system.
- One approval of a PR (by a maintainer) is enough for it to be merged.
- Unless someone approves the PR with optional comments, the PR is immediately merged by the approving reviewer.
- PRs are preferably merged via the ["squash and merge"](github-docs:pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges#squash-and-merge-your-commits) option, to keep a clean commit history on the _main_ branch.

A typical PR workflow would be:
* Create a new branch, make your changes, and stage them.
* When you try to commit, the [pre-commit hooks](#formatting-and-pre-commit-hooks) will be triggered.
* Stage any changes made by the hooks, and commit.
* You may also run the pre-commit hooks manually, at any time, with `pre-commit run -a`.
* Make sure to write tests for any new features or bug fixes. See [testing](#testing) below.
* Don't forget to update the documentation, if necessary. See [contributing documentation](#contributing-documentation) below.
* Push your changes to GitHub and open a draft pull request, with a meaningful title and a thorough description of the changes.
* If all checks (e.g. linting, type checking, testing) run successfully, you may mark the pull request as ready for review.
* Respond to review comments and implement any requested changes.
* One of the maintainers will approve the PR and add it to the [merge queue](https://github.blog/changelog/2023-02-08-pull-request-merge-queue-public-beta/).
* Success 🎉 !! Your PR will be (squash-)merged into the _main_ branch.

## Development guidelines

### Formatting and pre-commit hooks
Running `pre-commit install` will set up [pre-commit hooks](https://pre-commit.com/) to ensure a consistent formatting style. Currently, these include:
* [ruff](https://github.com/astral-sh/ruff) does a number of jobs, including code linting and auto-formatting.
* [mypy](https://mypy.readthedocs.io/en/stable/index.html) as a static type checker.
* [check-manifest](https://github.com/mgedmin/check-manifest) to ensure that the right files are included in the pip package.
* [codespell](https://github.com/codespell-project/codespell) to check for common misspellings.

These will prevent code from being committed if any of these hooks fail.
To run all the hooks before committing:

```sh
pre-commit run  # for staged files
pre-commit run -a  # for all files in the repository
```

Some problems will be automatically fixed by the hooks. In this case, you should
stage the auto-fixed changes and run the hooks again:

```sh
git add .
pre-commit run
```

If a problem cannot be auto-fixed, the corresponding tool will provide
information on what the issue is and how to fix it. For example, `ruff` might
output something like:

```sh
movement/io/load_poses.py:551:80: E501 Line too long (90 > 79)
```

This pinpoints the problem to a single code line and a specific [ruff rule](https://docs.astral.sh/ruff/rules/) violation.
Sometimes you may have good reasons to ignore a particular rule for a specific line of code. You can do this by adding an inline comment, e.g. `# noqa: E501`. Replace `E501` with the code of the rule you want to ignore.

For docstrings, we adhere to the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style.
Make sure to provide docstrings for all public functions, classes, and methods.
This is important as it allows for [automatic generation of the API reference](#updating-the-api-reference).

### Testing
We use [pytest](https://docs.pytest.org/en/latest/) for testing, aiming for ~100% test coverage where feasible. All new features should be accompanied by tests.

Tests are stored in the `tests` directory, structured as follows:

- `test_unit/`: Contains unit tests that closely follow the `movement` package structure.
- `test_integration/`: Includes tests for interactions between different modules.
- `fixtures/`: Holds reusable test data fixtures, automatically imported via `conftest.py`. Check for existing fixtures before adding new ones, to avoid duplication.

For tests requiring experimental data, you can use [sample data](#sample-data) from our external data repository.
These datasets are accessible through the `pytest.DATA_PATHS` dictionary, populated in `conftest.py`.
Avoid including large data files directly in the GitHub repository.

#### Running benchmark tests
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

#### Comparing benchmark runs across branches

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

### Logging
We use the {mod}`loguru<loguru._logger>`-based {class}`MovementLogger<movement.utils.logging.MovementLogger>` for logging.
The logger is configured to write logs to a rotating log file at the `DEBUG` level and to {obj}`sys.stderr` at the `WARNING` level.

To import the logger:
```python
from movement.utils.logging import logger
```

Once the logger is imported, you can log messages with the appropriate [severity levels](inv:loguru#levels) using the same syntax as {mod}`loguru<loguru._logger>` (e.g. `logger.debug("Debug message")`, `logger.warning("Warning message")`).

#### Logging and raising exceptions
Both {meth}`logger.error()<movement.utils.logging.MovementLogger.error>` and {meth}`logger.exception()<movement.utils.logging.MovementLogger.exception>` can be used to log [](inv:python#tut-errors), with the difference that the latter will include the traceback in the log message.
As these methods will return the logged Exception, you can log and raise the Exception in a single line:
```python
raise logger.error(ValueError("message"))
raise logger.exception(ValueError("message")) # with traceback
```

#### When to use `print`, `warnings.warn`, `logger.warning` and `logger.info`
We aim to adhere to the [When to use logging guide](inv:python#logging-basic-tutorial) to ensure consistency in our logging practices.
In general:
* Use {func}`print` for simple, non-critical messages that do not need to be logged.
* Use {func}`warnings.warn` for user input issues that are non-critical and can be addressed within `movement`, e.g. deprecated function calls that are redirected, invalid `fps` number in {class}`ValidPosesInputs<movement.validators.datasets.ValidPosesInputs>` that is implicitly set to `None`; or when processing data containing excessive NaNs, which the user can potentially address using appropriate methods, e.g. {func}`interpolate_over_time()<movement.filtering.interpolate_over_time>`
* Use {meth}`logger.info()<loguru._logger.Logger.info>` for informational messages about expected behaviours that do not indicate problems, e.g. where default values are assigned to optional parameters.

### Continuous integration
All pushes and pull requests will be built by [GitHub actions](github-docs:actions).
This will usually include linting, testing and deployment.

A GitHub actions workflow (`.github/workflows/test_and_deploy.yml`) has been set up to run (on each push/PR):
* Linting checks (pre-commit).
* Testing (only if linting checks pass)
* Release to PyPI (only if a git tag is present and if tests pass).

### Versioning and releases
We use [semantic versioning](https://semver.org/), which includes `MAJOR`.`MINOR`.`PATCH` version numbers:

* PATCH = small bugfix
* MINOR = new feature
* MAJOR = breaking change

We use [setuptools_scm](setuptools-scm:) to automatically version `movement`.
It has been pre-configured in the `pyproject.toml` file.
`setuptools_scm` will automatically [infer the version using git](setuptools-scm:usage#default-versioning-scheme).
To manually set a new semantic version, create a tag and make sure the tag is pushed to GitHub.
Make sure you commit any changes you wish to be included in this version. E.g. to bump the version to `1.0.0`:

```sh
git add .
git commit -m "Add new changes"
git tag -a v1.0.0 -m "Bump to version 1.0.0"
git push --follow-tags
```
Alternatively, you can also use the GitHub web interface to create a new release and tag.

The addition of a GitHub tag triggers the package's deployment to PyPI.
The version number is automatically determined from the latest tag on the _main_ branch.

## Contributing documentation
The documentation is hosted via [GitHub pages](https://pages.github.com/) at
[movement.neuroinformatics.dev](target-movement).
Its source files are located in the `docs` folder of this repository.
They are written in either [Markdown](myst-parser:syntax/typography.html)
or [reStructuredText](https://docutils.sourceforge.io/rst.html).
The `index.md` file corresponds to the homepage of the documentation website.
Other `.md`  or `.rst` files are linked to the homepage via the `toctree` directive.

We use [Sphinx](sphinx-doc:) and the [PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html)
to build the source files into HTML output.
This is handled by a GitHub actions workflow (`.github/workflows/docs_build_and_deploy.yml`).
The build job runs on each PR, ensuring that the documentation build is not broken by new changes.
The deployment job runs on tag pushes (for PyPI releases) or manual triggers on the _main_ branch.
This keeps the documentation aligned with releases, while allowing manual redeployment when necessary.

### Editing the documentation
To edit the documentation, ensure you have already set up a [development environment](#creating-a-development-environment).

To build the documentation locally, install the extra dependencies by running the following command from the repository root:
```sh
pip install -e ".[docs]"      # conda env
uv pip install -e ".[docs]"   # uv env
```

Now create a new branch, edit the documentation source files (`.md` or `.rst` in the `docs` folder),
and commit your changes. Submit your documentation changes via a pull request,
following the [same guidelines as for code changes](#pull-requests).
Make sure that the header levels in your `.md` or `.rst` files are incremented
consistently (H1 > H2 > H3, etc.) without skipping any levels.

#### Adding new pages
If you create a new documentation source file (e.g. `my_new_file.md` or `my_new_file.rst`),
you will need to add it to the `toctree` directive in `index.md`
for it to be included in the documentation website:

```rst
:maxdepth: 2
:hidden:

existing_file
my_new_file
```

#### Linking to external URLs
If you are adding references to an external URL (e.g. `https://github.com/neuroinformatics-unit/movement/issues/1`) in a `.md` file, you will need to check if a matching URL scheme (e.g. `https://github.com/neuroinformatics-unit/movement/`) is defined in `myst_url_schemes` in `docs/source/conf.py`. If it is, the following `[](scheme:loc)` syntax will be converted to the [full URL](movement-github:issues/1) during the build process:
```markdown
[link text](movement-github:issues/1)
```

If it is not yet defined and you have multiple external URLs pointing to the same base URL, you will need to [add the URL scheme](myst-parser:syntax/cross-referencing.html#customising-external-url-resolution) to `myst_url_schemes` in `docs/source/conf.py`.

### Updating the API reference
The [API reference](target-api) is auto-generated by the `docs/make_api.py` script, and the [sphinx-autodoc](sphinx-doc:extensions/autodoc.html) and [sphinx-autosummary](sphinx-doc:extensions/autosummary.html) extensions.
The script inspects the source tree and generates the `docs/source/api_index.rst` file, which lists the modules to be included in the [API reference](target-api), skipping those listed in `EXCLUDE_MODULES`.

For each _package module_ listed in `PACKAGE_MODULES`&mdash;a module that re-exports selected classes and functions from its submodules via `__init__.py` (e.g. {mod}`movement.kinematics`)&mdash;the script also generates a `.rst` file in `docs/source/api/` with autosummary entries for the top-level objects exposed by the module.

The Sphinx extensions then generate the API reference pages for each module listed in `api_index.rst`, based on their docstrings.
So make sure that all your public functions/classes/methods have valid docstrings following the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style.
Our `pre-commit` hooks include some checks (`ruff` rules) that ensure the docstrings are formatted consistently.

If your PR introduces new modules that should *not* be documented in the [API reference](target-api), or if there are changes to existing modules that necessitate their removal from the documentation, make sure to update `EXCLUDE_MODULES` in `docs/make_api.py` accordingly.

Likewise, if you want to document a module that exposes its public API via its `__init__.py`, rather than through its submodules individually, make sure to add it to `PACKAGE_MODULES` in `docs/make_api.py`.

### Updating the examples
We use [sphinx-gallery](sphinx-gallery:)
to create the [examples](target-examples).
To add new examples, you will need to create a new `.py` file in `examples/`,
or in `examples/advanced/` if your example targets experienced users.
The file should be structured as specified in the relevant
[sphinx-gallery documentation](sphinx-gallery:syntax).

We are using sphinx-gallery's [integration with binder](sphinx-gallery:configuration#binder-links), to provide interactive versions of the examples.
This is configured in `docs/source/conf.py` under the `sphinx_gallery_conf` variable,
and further customised for our repository by the `.binder/postBuild` script.
If your examples rely on packages that are not among `movement`'s dependencies,
you will need to add them to the `.binder/requirements.txt` file.

### Cross-referencing Python objects
:::{note}
Docstrings in the `.py` files for the [API reference](target-api) and the [examples](target-examples) are converted into `.rst` files, so these should use reStructuredText syntax.
:::

#### Internal references
::::{tab-set}
:::{tab-item} Markdown
For referencing `movement` objects in `.md` files, use the `` {role}`target` `` syntax with the appropriate [Python object role](sphinx-doc:domains/python.html#cross-referencing-python-objects).

For example, to reference the {mod}`movement.io.load_poses` module, use:
```markdown
{mod}`movement.io.load_poses`
```
:::
:::{tab-item} RestructuredText
For referencing `movement` objects in `.rst` files, use the `` :role:`target` `` syntax with the appropriate [Python object role](sphinx-doc:domains/python.html#cross-referencing-python-objects).

For example, to reference the {mod}`movement.io.load_poses` module, use:
```rst
:mod:`movement.io.load_poses`
```
:::
::::

#### External references
For referencing external Python objects using [intersphinx](sphinx-doc:extensions/intersphinx.html),
ensure the mapping between module names and their documentation URLs is defined in [`intersphinx_mapping`](sphinx-doc:extensions/intersphinx.html#confval-intersphinx_mapping) in `docs/source/conf.py`.
Once the module is included in the mapping, use the same syntax as for [internal references](#internal-references).

::::{tab-set}
:::{tab-item} Markdown
For example, to reference the {meth}`xarray.Dataset.update` method, use:
```markdown
{meth}`xarray.Dataset.update`
```
:::

:::{tab-item} RestructuredText
For example, to reference the {meth}`xarray.Dataset.update` method, use:
```rst
:meth:`xarray.Dataset.update`
```
:::
::::

### Updating the contributors list
The [contributors list](target-contributors) is automatically updated on the first day of each month by a GitHub actions workflow (`.github/workflows/update_contributors_list.yml`).
It uses the [Contributors-Readme-Action](https://github.com/marketplace/actions/contribute-list) to generate the list of contributors based on the commits to the repository.

It is also possible to manually add other contributors who have not contributed code to the repository, but have contributed in other ways (e.g. by providing sample data, or by actively participating in discussions).
The way to add them differs depending on whether they are GitHub users or not.

::::{tab-set}
:::{tab-item} GitHub users
To add a contributor who has a GitHub account, locate the section marked with `MANUAL: OTHER GITHUB CONTRIBUTORS` in `docs/source/community/people.md`.

Next, add their GitHub username (e.g. `newcontributor`) to the `<!-- readme: -start -->` and `<!-- readme: -end -->` lines as follows:
```html
<!-- readme: githubUser1,githubUser2,newcontributor -start -->
existing content...
<!-- readme: githubUser1,githubUser2,newcontributor -end -->
```

The aforementioned GitHub actions workflow will then automatically update the contributors list with `newcontributor`'s GitHub profile picture, name, and link to their GitHub profile.
:::

:::{tab-item} Non-GitHub users
To add a contributor who does not have a GitHub account, locate the section marked with `MANUAL: OTHER NON-GITHUB CONTRIBUTORS` in `docs/source/community/people.md`.

Next, add a row containing the contributor's image, name, and link to their website to the existing `list-table` as follows:
```markdown
*   - existing content...
*   - [![newcontributor](https://newcontributor.image.jpg) <br /> <sub><b>New Contributor</b></sub>](https://newcontributor.website.com)
```
:::
::::

### Building the documentation locally
We recommend that you build and view the documentation website locally, before you push your proposed changes.

First, ensure your development environment with the required dependencies is active (see [Editing the documentation](#editing-the-documentation) for details on how to create it). Then, navigate to the `docs/` directory:
```sh
cd docs
```
All subsequent commands should be run from this directory.

:::{note}
Windows PowerShell users should prepend `make` commands with `.\` (e.g. `.\make html`).
:::

To build the documentation, run:

```sh
make html
```
The local build can be viewed by opening `docs/build/html/index.html` in a browser.

To re-build the documentation after making changes,
we recommend removing existing build files first.
The following command will remove all generated files in `docs/`,
including the auto-generated files `source/api_index.rst` and
`source/snippets/admonitions.md`, as well as all files in
 `build/`, `source/api/`, and `source/examples/`.
 It will then re-build the documentation:

```sh
make clean html
```

To check that external links are correctly resolved, run:

```sh
make linkcheck
```

If the linkcheck step incorrectly marks links with valid anchors as broken, you can skip checking the anchors in specific links by adding the URLs to `linkcheck_anchors_ignore_for_url` in `docs/source/conf.py`, e.g.:

```python
# The linkcheck builder will skip verifying that anchors exist when checking
# these URLs
linkcheck_anchors_ignore_for_url = [
    "https://gin.g-node.org/G-Node/Info/wiki/",
    "https://neuroinformatics.zulipchat.com/",
]
```

:::{tip}
The `make` commands can be combined to run multiple tasks sequentially.
For example, to re-build the documentation and check the links, run:
```sh
make clean html linkcheck
```
:::

### Previewing the documentation in continuous integration
We use [artifact.ci](https://www.artifact.ci/) to preview the documentation that is built as part of our GitHub Actions workflow. To do so:
1. Go to the "Checks" tab in the GitHub PR.
2. Click on the "Docs" section on the left.
3. If the "Build Sphinx Docs" action is successful, a summary section will appear under the block diagram with a link to preview the built documentation.
4. Click on the link and wait for the files to be uploaded (it may take a while the first time). You may be asked to sign in to GitHub.
5. Once the upload is complete, look for `docs/build/html/index.html` under the "Detected Entrypoints" section.


## Sample data
We maintain some sample datasets to be used for testing, examples and tutorials on an
[external data repository](gin:neuroinformatics/movement-test-data).
Our hosting platform of choice is called [GIN](gin:) and is maintained
by the [German Neuroinformatics Node](https://www.g-node.org/).
GIN has a GitHub-like interface and git-like
[CLI](gin:G-Node/Info/wiki/GIN+CLI+Setup#quickstart) functionalities.

Currently, the data repository contains sample pose estimation data files
stored in the `poses` folder, and tracked bounding boxes data files under the `bboxes` folder. For some of these files, we also host
the associated video file (in the `videos` folder) and/or a single
video frame (in the `frames`) folder. These can be used to develop and
test visualisations, e.g. to overlay the data on video frames.
The `metadata.yaml` file holds metadata for each sample dataset,
including information on data provenance as well as the mapping between data files and related
video/frame files.

For most sample datasets, the tracking data lives in a single file under `poses` or `bboxes`.
However, some tools—like [TRex](TRex:)—may split their tracking outputs across multiple files.
In those cases, the dataset is distributed as a ZIP archive containing every relevant file, and is automatically extracted when fetched.

### Fetching data
To fetch the data from GIN, we use the [pooch](https://www.fatiando.org/pooch/latest/index.html)
Python package, which can download data from pre-specified URLs and store them
locally for all subsequent uses. It also provides some nice utilities,
like verification of sha256 hashes and decompression of archives.

The relevant functionality is implemented in the {mod}`movement.sample_data` module.
The most important parts of this module are:

1. The `SAMPLE_DATA` download manager object.
2. The {func}`list_datasets()<movement.sample_data.list_datasets>` function, which returns a list of the available poses and bounding boxes datasets (file names of the data files).
3. The {func}`fetch_dataset_paths()<movement.sample_data.fetch_dataset_paths>` function, which returns a dictionary containing local paths to the files associated with a particular sample dataset: `poses` or `bboxes`, `frame`, `video`. If the relevant files are not already cached locally, they will be downloaded.
4. The {func}`fetch_dataset()<movement.sample_data.fetch_dataset>` function, which downloads the files associated with a given sample dataset (same as `fetch_dataset_paths()`) and additionally loads the pose or bounding box data into `movement`, returning an `xarray.Dataset` object. If available, the local paths to the associated video and frame files are stored as dataset attributes, with names `video_path` and `frame_path`, respectively.

By default, the downloaded files are stored in the `~/.movement/data` folder.
This can be changed by setting the `DATA_DIR` variable in the `sample_data.py` file.

### Adding new data
Only core `movement` developers may add new files to the external data repository.
Make sure to run the following procedure on a UNIX-like system, as we have observed some weird behaviour on Windows (some sha256sums may end up being different).
To add a new file, you will need to:

1. Create a [GIN](gin:) account.
2. Request collaborator access to the [movement data repository](gin:neuroinformatics/movement-test-data) if you don't already have it.
3. Install and configure the [GIN CLI](gin:G-Node/Info/wiki/GIN+CLI+Setup#quickstart) by running `gin login` in a terminal with your GIN credentials.
4. Clone the `movement` data repository to your local machine using `gin get neuroinformatics/movement-test-data`, then run `gin download --content` to download all the files.
5. Add your new files to the appropriate folders (`poses`, `bboxes`, `videos`, and/or `frames`) following the existing file naming conventions.
6. Add metadata for your new files to `metadata.yaml` using the example entry below as a template. You can leave all `sha256sum` values as `null` for now.
7. Update file hashes in `metadata.yaml` by running `python update_hashes.py` from the root of the [movement data repository](gin:neuroinformatics/movement-test-data). This script computes SHA256 hashes for all data files and updates the corresponding `sha256sum` values in the metadata file. Make sure you're in a [Python environment with `movement` installed](#creating-a-development-environment).
8. Commit your changes using `gin commit -m <message> <filename>` for specific files or `gin commit -m <message> .` for all changes.
9. Upload your committed changes to the GIN repository with `gin upload`. Use `gin download` to pull the latest changes or `gin sync` to synchronise changes bidirectionally.

### `metadata.yaml` example entry
```yaml
SLEAP_three-mice_Aeon_proofread.analysis.h5:
  sha256sum: null
  source_software: SLEAP
  type: poses
  fps: 50
  species: mouse
  number_of_individuals: 3
  shared_by:
    name: Chang Huan Lo
    affiliation: Sainsbury Wellcome Centre, UCL
  frame:
    file_name: three-mice_Aeon_frame-5sec.png
    sha256sum: null
  video:
    file_name: three-mice_Aeon_video.avi
    sha256sum: null
  note: All labels were proofread (user-defined) and can be considered ground truth.
    It was exported from the .slp file with the same prefix.
```
