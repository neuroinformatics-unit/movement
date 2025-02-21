# How to Contribute

## Contributing code

### Creating a development environment

It is recommended to use [conda](conda:)
or [mamba](mamba:) to create a
development environment for movement. In the following we assume you have
`conda` installed, but the same commands will also work with `mamba`/`micromamba`.

First, create and activate a `conda` environment with some prerequisites:

```sh
conda create -n movement-dev -c conda-forge python=3.11 pytables
conda activate movement-dev
```

To install movement for development, clone the [GitHub repository](movement-github:),
and then run from within the repository:

```sh
pip install -e .[dev]  # works on most shells
pip install -e '.[dev]'  # works on zsh (the default shell on macOS)
```

This will install the package in editable mode, including all dependencies
required for development.

Finally, initialise the [pre-commit hooks](#formatting-and-pre-commit-hooks):

```bash
pre-commit install
```

### Pull requests

In all cases, please submit code to the main repository via a pull request (PR).
We recommend, and adhere, to the following conventions:

- Please submit _draft_ PRs as early as possible to allow for discussion.
- The PR title should be descriptive e.g. "Add new function to do X" or "Fix bug in Y".
- The PR description should be used to provide context and motivation for the changes.
- One approval of a PR (by a repo owner) is enough for it to be merged.
- Unless someone approves the PR with optional comments, the PR is immediately merged by the approving reviewer.
- Ask for a review from someone specific if you think they would be a particularly suited reviewer.
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

These will prevent code from being committed if any of these hooks fail. To run them individually (from the root of the repository), you can use:

```sh
ruff .
mypy -p movement
check-manifest
codespell
```

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

We use [pytest](https://docs.pytest.org/en/latest/) for testing and aim for
~100% test coverage (as far as is reasonable).
All new features should be tested.
Write your test methods and classes in the _tests_ folder.

For some tests, you will need to use real experimental data.
Do not include these data in the repository, especially if they are large.
We store several sample datasets in an external data repository.
See [sample data](#sample-data) for more information.


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

We use [setuptools_scm](setuptools-scm:) to automatically version movement.
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
The build job is triggered on each PR, ensuring that the documentation build is not broken by new changes.
The deployment job is only triggered whenever a tag is pushed to the _main_ branch,
ensuring that the documentation is published in sync with each PyPI release.


### Editing the documentation

To edit the documentation, first clone the repository, and install `movement` in a
[development environment](#creating-a-development-environment).

Then, install a few additional dependencies in your development environment to be able to build the documentation locally. To do this, run the following command from the root of the repository:
```sh
pip install -r ./docs/requirements.txt
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
The [API reference](target-api) is auto-generated by the `docs/make_api_index.py` script, and the [sphinx-autodoc](sphinx-doc:extensions/autodoc.html) and [sphinx-autosummary](sphinx-doc:extensions/autosummary.html) extensions.
The script generates the `docs/source/api_index.rst` file containing the list of modules to be included in the [API reference](target-api).
The plugins then generate the API reference pages for each module listed in `api_index.rst`, based on the docstrings in the source code.
So make sure that all your public functions/classes/methods have valid docstrings following the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style.
Our `pre-commit` hooks include some checks (`ruff` rules) that ensure the docstrings are formatted consistently.

If your PR introduces new modules that should *not* be documented in the [API reference](target-api), or if there are changes to existing modules that necessitate their removal from the documentation, make sure to update the `exclude_modules` list within the `docs/make_api_index.py` script to reflect these exclusions.

### Updating the examples
We use [sphinx-gallery](sphinx-gallery:)
to create the [examples](target-examples).
To add new examples, you will need to create a new `.py` file in `examples/`.
The file should be structured as specified in the relevant
[sphinx-gallery documentation](sphinx-gallery:syntax).

We are using sphinx-gallery's [integration with binder](sphinx-gallery:configuration#binder-links)
to provide interactive versions of the examples.
If your examples rely on packages that are not among movement's dependencies,
you will need to add them to the `docs/source/environment.yml` file.
That file is used by binder to create the conda environment in which the
examples are run. See the relevant section of the
[binder documentation](https://mybinder.readthedocs.io/en/latest/using/config_files.html).

### Cross-referencing Python objects
:::{note}
Docstrings in the `.py` files for the [API reference](target-api) and the [examples](target-examples) are converted into `.rst` files, so these should use reStructuredText syntax.
:::

#### Internal references
::::{tab-set}
:::{tab-item} Markdown
For referencing movement objects in `.md` files, use the `` {role}`target` `` syntax with the appropriate [Python object role](sphinx-doc:domains/python.html#cross-referencing-python-objects).

For example, to reference the {mod}`movement.io.load_poses` module, use:
```markdown
{mod}`movement.io.load_poses`
```
:::
:::{tab-item} RestructuredText
For referencing movement objects in `.rst` files, use the `` :role:`target` `` syntax with the appropriate [Python object role](sphinx-doc:domains/python.html#cross-referencing-python-objects).

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

### Fetching data
To fetch the data from GIN, we use the [pooch](https://www.fatiando.org/pooch/latest/index.html)
Python package, which can download data from pre-specified URLs and store them
locally for all subsequent uses. It also provides some nice utilities,
like verification of sha256 hashes and decompression of archives.

The relevant functionality is implemented in the `movement.sample_data.py` module.
The most important parts of this module are:

1. The `SAMPLE_DATA` download manager object.
2. The `list_datasets()` function, which returns a list of the available poses and bounding boxes datasets (file names of the data files).
3. The `fetch_dataset_paths()` function, which returns a dictionary containing local paths to the files associated with a particular sample dataset: `poses` or `bboxes`, `frame`, `video`. If the relevant files are not already cached locally, they will be downloaded.
4. The `fetch_dataset()` function, which downloads the files associated with a given sample dataset (same as `fetch_dataset_paths()`) and additionally loads the pose or bounding box data into movement, returning an `xarray.Dataset` object. If available, the local paths to the associated video and frame files are stored as dataset attributes, with names `video_path` and `frame_path`, respectively.

By default, the downloaded files are stored in the `~/.movement/data` folder.
This can be changed by setting the `DATA_DIR` variable in the `movement.sample_data.py` module.

### Adding new data
Only core movement developers may add new files to the external data repository.
To add a new file, you will need to:

1. Create a [GIN](gin:) account
2. Ask to be added as a collaborator on the [movement data repository](gin:neuroinformatics/movement-test-data) (if not already)
3. Download the [GIN CLI](gin:G-Node/Info/wiki/GIN+CLI+Setup#quickstart) and set it up with your GIN credentials, by running `gin login` in a terminal.
4. Clone the movement data repository to your local machine, by running `gin get neuroinformatics/movement-test-data` in a terminal.
5. Add your new files to the `poses`, `bboxes`, `videos` and/or `frames` folders as appropriate. Follow the existing file naming conventions as closely as possible.
6. Determine the sha256 checksum hash of each new file. You can do this in a terminal by running:
    ::::{tab-set}
    :::{tab-item} Ubuntu
    ```bash
    sha256sum <filename>
    ```
    :::

    :::{tab-item} MacOS
    ```bash
    shasum -a 256 <filename>
    ```
    :::

    :::{tab-item} Windows
    ```bash
    certutil -hashfile <filename> SHA256
    ```
    :::
    ::::
    For convenience, we've included a `get_sha256_hashes.py` script in the [movement data repository](gin:neuroinformatics/movement-test-data). If you run this from the root of the data repository, within a Python environment with movement installed, it will calculate the sha256 hashes for all files in the `poses`, `bboxes`, `videos` and `frames` folders and write them to files named `poses_hashes.txt`, `bboxes_hashes.txt`, `videos_hashes.txt`, and `frames_hashes.txt` respectively.

7. Add metadata for your new files to `metadata.yaml`, including their sha256 hashes you've calculated. See the example entry below for guidance.

8. Commit a specific file with `gin commit -m <message> <filename>`, or `gin commit -m <message> .` to commit all changes.

9. Upload the committed changes to the GIN repository by running `gin upload`. Latest changes to the repository can be pulled via `gin download`. `gin sync` will synchronise the latest changes bidirectionally.



### `metadata.yaml` example entry
```yaml
"SLEAP_three-mice_Aeon_proofread.analysis.h5":
  sha256sum: "82ebd281c406a61536092863bc51d1a5c7c10316275119f7daf01c1ff33eac2a"
  source_software: "SLEAP"
  type: "poses"  # "poses" or "bboxes" depending on the type of tracked data
  fps: 50
  species: "mouse"
  number_of_individuals: 3
  shared_by:
    name: "Chang Huan Lo"
    affiliation: "Sainsbury Wellcome Centre, UCL"
  frame:
    file_name: "three-mice_Aeon_frame-5sec.png"
    sha256sum: "889e1bbee6cb23eb6d52820748123579acbd0b2a7265cf72a903dabb7fcc3d1a"
  video:
    file_name: "three-mice_Aeon_video.avi"
    sha256sum: "bc7406442c90467f11a982fd6efd85258ec5ec7748228b245caf0358934f0e7d"
  note: "All labels were proofread (user-defined) and can be considered ground truth. It was exported from the .slp file with the same prefix."
```
