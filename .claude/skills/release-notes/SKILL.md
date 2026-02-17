---
name: release-notes
description: Organise auto-generated GitHub release notes into structured sections
allowed-tools: Read, Edit, Write, Bash(gh *), Grep, Glob
disable-model-invocation: true
---

# Organise release notes for a movement release

You are helping prepare release notes for the `movement` Python package.
The user will paste GitHub's auto-generated release notes (a flat list of PR bullet points under `## What's Changed`).
Your job is to organise these into meaningful sections.

## Workflow

1. **Read the raw notes** the user provides (either pasted directly or in a file).
2. **Investigate highlight PRs.** The user will tell you which PR(s) to highlight. Use `gh pr view <number>` and read relevant source files to understand what changed — especially new APIs, deprecated functions, and breaking changes.
3. **Organise PRs into sections** using the format below. Keep the original PR bullet points intact (do not rewrite them).
4. **Write section descriptions** for highlights. Include:
   - A shoutout to the main contributor(s) of the feature — especially if they are new or external contributors.
   - A brief explanation of the change and why it matters.
   - Code snippets showing the new API/syntax.
   - For deprecations: list deprecated functions in a `> [!WARNING]` block, and provide before/after migration examples.
   - For breaking changes: show old vs new syntax.
5. **Acknowledge new contributors** — keep the `## New Contributors` section as-is.
6. **Verify completeness** — confirm every PR from the original list appears exactly once in the final notes.

## Section format

Use these sections as a starting point. Each section header uses an emoji prefix. Use your discretion — not every release will need all sections, and some releases may warrant additional or different ones depending on the PRs involved.

Commonly used sections (in typical order):

```
To update movement to the latest version, see the [update guide](https://movement.neuroinformatics.dev/latest/user_guide/installation.html#update-the-package).

## What's Changed

### ⚡️ Highlight: <short description>
(descriptive text, code snippets, deprecation warnings, migration guide)
* PR bullet ...

### 🐛 Bug fixes
* PR bullet ...

### 🤝 Improving the contributor experience
* PR bullet ...

### 📚 Documentation
* PR bullet ...

### 🧹 Housekeeping
* PR bullet ...

## New Contributors
* ...

**Full Changelog**: ...
```

Other sections you might use depending on the release:
- `🛠️ Refactoring` — for significant internal restructuring
- `✨ New features` — when there are multiple new features beyond the highlight
- `⚠️ Breaking changes` — when API changes require user action

## Categorisation guidelines

- **Highlight**: Flagship features or breaking changes the user wants to showcase. Ask the user which PR(s) to highlight if they don't specify.
- **Bug fixes**: PRs that fix incorrect behaviour (code bugs, not doc typos).
- **Improving the contributor experience**: Contributing guide updates, benchmarks, dev tooling improvements (e.g. tox-uv, moving dev config to pyproject.toml), restructured community docs.
- **Documentation**: Doc fixes, new examples, link fixes, doc page additions.
- **Housekeeping**: Bot PRs (pre-commit, dependabot, contributors-readme), CI/CD changes, dependency pinning, linkcheck fixes, license metadata.

Use your judgement for edge cases. When a PR could fit multiple sections, prefer the more specific one.

## Example

For the v0.14.0 release, the raw auto-generated notes were:

```
## What's Changed
* Restrict sphinx version to < 9 by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/768
* Use `tox-uv` in test action by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/766
* Fix docstring Raises formatting in LineOfInterest.normal by @AlgoFoe in https://github.com/neuroinformatics-unit/movement/pull/771
* Move docs dependencies to pyproject.toml by @AlgoFoe in https://github.com/neuroinformatics-unit/movement/pull/774
* Fix deprecated license syntax by @sfmig in https://github.com/neuroinformatics-unit/movement/pull/549
* Add preliminary benchmarks by @sfmig in https://github.com/neuroinformatics-unit/movement/pull/772
* Ignore Docutils URL in linkcheck by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/793
* Hide _factorized properties from Points layer tooltips by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/781
* Added link to FOSDEM 2026 talk by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/797
* [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci[bot] in https://github.com/neuroinformatics-unit/movement/pull/795
* Fix broken 404 page by @AlgoFoe in https://github.com/neuroinformatics-unit/movement/pull/785
* Contributors-Readme-Action: Update contributors list by @github-actions[bot] in https://github.com/neuroinformatics-unit/movement/pull/787
* Bump conda-incubator/setup-miniconda from 3.2.0 to 3.3.0 by @dependabot[bot] in https://github.com/neuroinformatics-unit/movement/pull/788
* Adapt `savgol_filter` for compatibility with Scipy >= 1.17 by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/761
* Fix coordinate assignment for `elem2` in `_cdist` by @HARSHDIPSAHA in https://github.com/neuroinformatics-unit/movement/pull/776
* Docs: restructure contributing guide and add community cards by @ishan372or in https://github.com/neuroinformatics-unit/movement/pull/764
* Authenticate with GitHub token during linkcheck by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/800
* Fix links to SLEAP analysis h5 docs by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/807
* Add code snippets for verifying sample data loading by @Edu92337 in https://github.com/neuroinformatics-unit/movement/pull/759
* Ignore ISO URL in linkcheck by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/815
* Unpin sphinx and pin ablog>=0.11.13 by @niksirbi in https://github.com/neuroinformatics-unit/movement/pull/811
* Unify loaders by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/722

## New Contributors
* @AlgoFoe made their first contribution in ...
* @HARSHDIPSAHA made their first contribution in ...
* @ishan372or made their first contribution in ...
* @Edu92337 made their first contribution in ...

**Full Changelog**: https://github.com/neuroinformatics-unit/movement/compare/v0.13.0...v0.14.0
```

With "Unify loaders" (#722) as the highlight, the organised result was:

```
To update movement to the latest version, see the [update guide](https://movement.neuroinformatics.dev/latest/user_guide/installation.html#update-the-package).

## What's Changed

### ⚡️ Highlight: unified data loading interface

* Unify loaders by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/722

Thanks to @lochhh's tireless efforts, loading poses and bounding boxes is now handled by a single entry point, `movement.io.load_dataset`.

The new `load_dataset` function works for all our supported third-party formats. For example:

\```python
from movement.io import load_dataset

# DeepLabCut -> poses dataset
ds = load_dataset("path/to/file.h5", source_software="DeepLabCut", fps=30)

# SLEAP -> poses dataset
ds = load_dataset("path/to/file.slp", source_software="SLEAP")

# VGG Image Annotator tracks -> bounding boxes dataset
ds = load_dataset("path/to/file.csv", source_software="VIA-tracks")
\```

Similarly, `movement.io.load_multiview_dataset` replaces the old `movement.io.load_poses.from_multiview_files`, with added support for bounding boxes.

Software-specific loaders (e.g. `load_poses.from_dlc_file`, `load_bboxes.from_via_tracks_file`) remain available for users who want full control over the loading process.

> [!WARNING]
> The following functions are deprecated and will be removed in a future release:
> - `load_poses.from_file` → use `load_dataset` instead
> - `load_bboxes.from_file` → use `load_dataset` instead
> - `load_poses.from_multiview_files` → use `load_multiview_dataset` instead

**Migrating from deprecated functions:**

\```python
# Before
from movement.io import load_poses, load_bboxes
ds = load_poses.from_file("file.h5", source_software="DeepLabCut", fps=30)

# After
from movement.io import load_dataset
ds = load_dataset("file.h5", source_software="DeepLabCut", fps=30)
\```

### 🐛 Bug fixes

* Adapt `savgol_filter` for compatibility with Scipy >= 1.17 by @niksirbi in .../pull/761
* Fix coordinate assignment for `elem2` in `_cdist` by @HARSHDIPSAHA in .../pull/776
* Hide _factorized properties from Points layer tooltips by @niksirbi in .../pull/781

### 🤝 Improving the contributor experience

* Docs: restructure contributing guide and add community cards by @ishan372or in .../pull/764
* Add preliminary benchmarks by @sfmig in .../pull/772
* Use `tox-uv` in test action by @niksirbi in .../pull/766
* Move docs dependencies to pyproject.toml by @AlgoFoe in .../pull/774

### 📚 Documentation

* Fix docstring Raises formatting in LineOfInterest.normal by @AlgoFoe in .../pull/771
* Added link to FOSDEM 2026 talk by @niksirbi in .../pull/797
* Fix broken 404 page by @AlgoFoe in .../pull/785
* Fix links to SLEAP analysis h5 docs by @niksirbi in .../pull/807
* Add code snippets for verifying sample data loading by @Edu92337 in .../pull/759

### 🧹 Housekeeping

* Restrict sphinx version to < 9 by @niksirbi in .../pull/768
* Unpin sphinx and pin ablog>=0.11.13 by @niksirbi in .../pull/811
* Fix deprecated license syntax by @sfmig in .../pull/549
* (... remaining bot/CI PRs ...)

## New Contributors
* @AlgoFoe made their first contribution in ...
* @HARSHDIPSAHA made their first contribution in ...
* @ishan372or made their first contribution in ...
* @Edu92337 made their first contribution in ...

**Full Changelog**: https://github.com/neuroinformatics-unit/movement/compare/v0.13.0...v0.14.0
```

## Important reminders

- Always keep the original PR bullet text intact — do not reword them.
- Every PR from the input must appear exactly once in the output.
- Ask the user which PR(s) to highlight if they don't specify.
- Use `gh pr view <number>` and read source code to understand highlight PRs before writing about them.
- Present a draft to the user for review; iterate based on their feedback.
