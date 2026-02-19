(target-contribution-workflow)=
# Contribution workflow

If you want to contribute to `movement` and don't have permission to make changes directly, you can create your own copy of the project, make updates, and then suggest those updates for inclusion in the main project. This process is often called a "fork and pull request" workflow.

When you create your own copy (or "fork") of a project, it's like making a new workspace that shares code with the original project.
Once you've made your changes in your copy, you can submit them as a pull request, which is a way to propose changes back to the main project.

If you are not familiar with `git`, we recommend reading up on [this guide](https://docs.github.com/en/get-started/using-git/about-git#basic-git-commands).

## Forking the repository

1. Fork the [repository](movement-github:) on GitHub.
   You can read more about [forking in the GitHub docs](https://docs.github.com/en/get-started/quickstart/fork-a-repo).

2. Clone your fork to your local machine and navigate to the repository folder:

    ```sh
    git clone [https://github.com/](https://github.com/)<your-github-username>/movement.git
    cd movement
    ```

3. Set the upstream remote to the base `movement` repository:
   This links your local copy to the original project so you can pull the latest changes.

    ```sh
    git remote add upstream [https://github.com/neuroinformatics-unit/movement.git](https://github.com/neuroinformatics-unit/movement.git)
    ```

    :::{note}
    Your repository now has two remotes: `origin` (your fork, where you push changes) and `upstream` (the main repository, where you pull updates from)

(target-creating-a-development-environment)=
## Creating a development environment

Now that you have the repository locally, you need to set up a Python environment and install the project dependencies.

1. Create an environment using [conda](conda:) or [uv](uv:getting-started/installation/) and install `movement` in editable mode, including development dependencies.

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
    If you also want to edit the documentation and preview the changes locally, you will additionally need the `docs` extra dependencies. See [Editing the documentation](target-editing-the-documentation) for more details.

2. Finally, initialise the [pre-commit hooks](target-formatting-and-pre-commit-hooks):

    ```sh
    pre-commit install
    ```

(target-pull-requests)=
## Pull requests
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
* When you try to commit, the [pre-commit hooks](target-formatting-and-pre-commit-hooks) will be triggered.
* Stage any changes made by the hooks, and commit.
* You may also run the pre-commit hooks manually, at any time, with `pre-commit run -a`.
* Make sure to write tests for any new features or bug fixes. See [testing](target-testing) below.
* Don't forget to [update the documentation](target-updating-docs), if necessary.
* Push your changes to your fork on GitHub(`git push origin <branch-name>`).
* Open a draft pull request from your fork to the upstream `movement` repository, with a meaningful title and a thorough description of the changes.
  :::{note}
  When creating the PR, ensure the base repository is `neuroinformatics-unit/movement` (the `upstream`) and the head repository is your fork. GitHub sometimes defaults to comparing against your own fork. Also make sure to tick the "Allow edits by maintainers" checkbox, so that maintainers can make small fixes directly to your branch.
  :::
* If all checks (e.g. linting, type checking, testing) run successfully, you may mark the pull request as ready for review.
* Respond to review comments and implement any requested changes.
* One of the maintainers will approve the PR and add it to the [merge queue](https://github.blog/changelog/2023-02-08-pull-request-merge-queue-public-beta/).
* Success 🎉 !! Your PR will be (squash-)merged into the _main_ branch.
