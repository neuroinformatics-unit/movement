# How to Contribute

## Contributing code

### Creating a development environment

It is recommended to use [conda](conda:)
or [mamba](mamba:) to create a
development environment for movement. In the following we assume you have
`conda` installed, but the same commands will also work with `mamba`/`micromamba`.

First, create and activate a `conda` environment with some prerequisites:

```sh
conda create -n movement-dev -c conda-forge python=3.10 pytables
conda activate movement-dev
```

The above method ensures that you will get packages that often can't be
installed via `pip`, including [hdf5](https://www.hdfgroup.org/solutions/hdf5/).

To install movement for development, clone the GitHub repository,
and then run from inside the repository:

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
* Success 🎉 !! Your PR will be (squash-)merged into the _main_ branch.

## Development guidelines

### Formatting and pre-commit hooks

Running `pre-commit install` will set up [pre-commit hooks](https://pre-commit.com/) to ensure a consistent formatting style. Currently, these include:
* [ruff](https://github.com/astral-sh/ruff) does a number of jobs, including enforcing PEP8 and sorting imports
* [black](https://black.readthedocs.io/en/stable/) for auto-formatting
* [mypy](https://mypy.readthedocs.io/en/stable/index.html) as a static type checker
* [check-manifest](https://github.com/mgedmin/check-manifest) to ensure that the right files are included in the pip package.

These will prevent code from being committed if any of these hooks fail. To run them individually (from the root of the repository), you can use:

```sh
ruff .
black ./
mypy -p movement
check-manifest
```

To run all the hooks before committing:

```sh
pre-commit run  # for staged files
pre-commit run -a  # for all files in the repository
```

For docstrings, we adhere to the  [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style.

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
They are written in either [reStructuredText](https://docutils.sourceforge.io/rst.html) or
[markdown](myst-parser:syntax/typography.html).
The `index.md` file corresponds to the homepage of the documentation website.
Other `.rst`  or `.md` files are linked to the homepage via the `toctree` directive.

We use [Sphinx](https://www.sphinx-doc.org/en/master/) and the
[PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html)
to build the source files into HTML output.
This is handled by a GitHub actions workflow (`.github/workflows/docs_build_and_deploy.yml`).
The build job is triggered on each PR, ensuring that the documentation build is not broken by new changes.
The deployment job is only triggerred whenever a tag is pushed to the _main_ branch,
ensuring that the documentation is published in sync with each PyPI release.

### Editing the documentation

To edit the documentation, first clone the repository, and install movement in a
[development environment](#creating-a-development-environment).

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

#### Adding external links
If you are adding references to an external link (e.g. `https://github.com/neuroinformatics-unit/movement/issues/1`) in a `.md` file, you will need to check if a matching URL scheme (e.g. `https://github.com/neuroinformatics-unit/movement/`) is defined in `myst_url_schemes` in `docs/source/conf.py`. If it is, the following `[](scheme:loc)` syntax will be converted to the [full URL](movement-github:issues/1) during the build process:
```markdown
[link text](movement-github:issues/1)
```

If it is not yet defined and you have multiple external links pointing to the same base URL, you will need to [add the URL scheme](myst-parser:syntax/cross-referencing.html#customising-external-url-resolution) to `myst_url_schemes` in `docs/source/conf.py`.


### Updating the API reference
If your PR introduces new public-facing functions, classes, or methods,
make sure to add them to the `docs/source/api_index.rst` page, so that they are
included in the [API reference](target-api),
e.g.:

```rst
My new module
--------------
.. currentmodule:: movement.new_module
.. autosummary::
    :toctree: api

    new_function
    NewClass
```

For this to work, your functions/classes/methods will need to have docstrings
that follow the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style.

### Updating the examples
We use [sphinx-gallery](sphinx-gallery:)
to create the [examples](target-examples).
To add new examples, you will need to create a new `.py` file in `examples/`.
The file should be structured as specified in the relevant
[sphinx-gallery documentation](sphinx-gallery:syntax).


### Building the documentation locally
We recommend that you build and view the documentation website locally, before you push it.
To do so, first install the requirements for building the documentation:
```sh
pip install -r docs/requirements.txt
```

Then, from the root of the repository, run:
```sh
sphinx-build docs/source docs/build
```

You can view the local build by opening `docs/build/index.html` in a browser.
To refresh the documentation, after making changes, remove the `docs/build` folder and re-run the above command:

```sh
rm -rf docs/build && sphinx-build docs/source docs/build
```

To check that external links are correctly resolved, run:

```sh
sphinx-build docs/source docs/build -b linkcheck
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

## Sample data

We maintain some sample data to be used for testing, examples and tutorials on an
[external data repository](gin:neuroinformatics/movement-test-data).
Our hosting platform of choice is called [GIN](gin:) and is maintained
by the [German Neuroinformatics Node](https://www.g-node.org/).
GIN has a GitHub-like interface and git-like
[CLI](gin:G-Node/Info/wiki/GIN+CLI+Setup#quickstart) functionalities.

Currently, the data repository contains sample pose estimation data files
stored in the `poses` folder. Metadata for these files, including information
about their provenance, is stored in the `poses_files_metadata.yaml` file.

### Fetching data
To fetch the data from GIN, we use the [pooch](https://www.fatiando.org/pooch/latest/index.html)
Python package, which can download data from pre-specified URLs and store them
locally for all subsequent uses. It also provides some nice utilities,
like verification of sha256 hashes and decompression of archives.

The relevant functionality is implemented in the `movement.sample_data.py` module.
The most important parts of this module are:

1. The `SAMPLE_DATA` download manager object.
2. The `list_sample_data()` function, which returns a list of the available files in the data repository.
3. The `fetch_sample_data_path()` function, which downloads a file (if not already cached locally) and returns the local path to it.
4. The `fetch_sample_data()` function, which downloads a file and loads it into movement directly, returning an `xarray.Dataset` object.

By default, the downloaded files are stored in the `~/.movement/data` folder.
This can be changed by setting the `DATA_DIR` variable in the `movement.sample_data.py` module.

### Adding new data
Only core movement developers may add new files to the external data repository.
To add a new file, you will need to:

1. Create a [GIN](gin:) account
2. Ask to be added as a collaborator on the [movement data repository](gin:neuroinformatics/movement-test-data) (if not already)
3. Download the [GIN CLI](gin:G-Node/Info/wiki/GIN+CLI+Setup#quickstart) and set it up with your GIN credentials, by running `gin login` in a terminal.
4. Clone the movement data repository to your local machine, by running `gin get neuroinformatics/movement-test-data` in a terminal.
5. Add your new files to `/movement-test-data/poses/`.
6. Determine the sha256 checksum hash of each new file by running `sha256sum <filename>` in a terminal. Alternatively, you can use `pooch` to do this for you: `python -c "import pooch; hash = pooch.file_hash('/path/to/file'); print(hash)"`. If you wish to generate a text file containing the hashes of all the files in a given folder, you can use `python -c "import pooch; pooch.make_registry('/path/to/folder', 'sha256_registry.txt')`.
7. Add metadata for your new files to `poses_files_metadata.yaml`, including their sha256 hashes.
8. Commit your changes using `gin commit -m <message> <filename>`.
9. Upload the committed changes to the GIN repository by running `gin upload`. Latest changes to the repository can be pulled via `gin download`. `gin sync` will synchronise the latest changes bidirectionally.
