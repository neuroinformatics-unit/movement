(target-code-style)=
# Code style

This page describes the conventions we follow to keep `movement`'s codebase consistent, covering automated formatting tools and logging practices.

(target-formatting-and-pre-commit-hooks)=
## Formatting and pre-commit hooks
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
This is important as it allows for [automatic generation of the API reference](target-updating-the-api-reference).

## Logging
We use the {mod}`loguru<loguru._logger>`-based {class}`MovementLogger<movement.utils.logging.MovementLogger>` for logging.
The logger is configured to write logs to a rotating log file at the `DEBUG` level and to {obj}`sys.stderr` at the `WARNING` level.

To import the logger:
```python
from movement.utils.logging import logger
```

Once the logger is imported, you can log messages with the appropriate [severity levels](inv:loguru#levels) using the same syntax as {mod}`loguru<loguru._logger>` (e.g. `logger.debug("Debug message")`, `logger.warning("Warning message")`).

### Logging and raising exceptions
Both {meth}`logger.error()<movement.utils.logging.MovementLogger.error>` and {meth}`logger.exception()<movement.utils.logging.MovementLogger.exception>` can be used to log [](inv:python#tut-errors), with the difference that the latter will include the traceback in the log message.
As these methods will return the logged Exception, you can log and raise the Exception in a single line:
```python
raise logger.error(ValueError("message"))
raise logger.exception(ValueError("message")) # with traceback
```

### When to use `print`, `warnings.warn`, `logger.warning` and `logger.info`
We aim to adhere to the [When to use logging guide](inv:python#logging-basic-tutorial) to ensure consistency in our logging practices.
In general:
* Use {func}`print` for simple, non-critical messages that do not need to be logged.
* Use {func}`warnings.warn` for user input issues that are non-critical and can be addressed within `movement`, e.g. deprecated function calls that are redirected, invalid `fps` number in {class}`ValidPosesInputs<movement.validators.datasets.ValidPosesInputs>` that is implicitly set to `None`; or when processing data containing excessive NaNs, which the user can potentially address using appropriate methods, e.g. {func}`interpolate_over_time()<movement.filtering.interpolate_over_time>`
* Use {meth}`logger.info()<loguru._logger.Logger.info>` for informational messages about expected behaviours that do not indicate problems, e.g. where default values are assigned to optional parameters.
