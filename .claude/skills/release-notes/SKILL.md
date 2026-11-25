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
2. **Investigate highlight PRs and check the whole release for API changes.** The user will tell you which PR(s) to highlight. Use `gh pr view <number>` and read relevant source files to understand what changed — especially new APIs, deprecated functions, and breaking changes. Then check the rest of the release too, not just the highlights:
   - Diff the previous tag against the release target (`git diff v<PREV>..origin/main -- movement`) for new `DeprecationWarning`s, removed or renamed parameters, changed defaults and changed `Literal` options. PRs described as refactors, simplifications or small fixes can still change behaviour.
   - If a PR description and the merged code disagree (e.g. the design changed during review), trust the code.
   - Check `pyproject.toml` for changes to runtime dependencies and tell the user: the conda-forge feedstock recipe will need the same changes.
   - If the user plans to merge more PRs before releasing, investigate those too and add bullets for them in GitHub's format (`* <title> by @<author> in <PR URL>`).
3. **Organise PRs into sections** using the format below. Keep the original PR bullet points intact (do not rewrite them). In any section that has prose (highlights, but also e.g. new features or community sections), lead with the prose and put the PR bullets at the end of the section, so the description isn't buried behind a list of commits. Introduce the PR bullets with a `**Pull requests:**` line in every section that has prose, to mark where the description ends; omit it in list-only sections (e.g. housekeeping), where the heading already says what follows.
4. **Write section descriptions** for highlights. Include:
   - A shoutout to the main contributor(s) of the feature — especially if they are new or external contributors.
   - A brief explanation of the change and why it matters. When describing new functions or APIs, follow the docstrings closely — use their wording for what the function does and base code examples on the docstring examples (if any exist).
   - Where possible, link to relevant docs pages using the pattern `[`movement.module.function()`](https://movement.neuroinformatics.dev/latest/api/movement.module.function.html)`. The URL path must use the function's full module path as it appears in the docs (e.g. `movement.io.load.load_dataset`, not `movement.io.load_dataset`). When in doubt, check the actual docs URL. For new gallery examples, link to the rendered page.
   - Code snippets showing the new API/syntax.
   - For deprecations: list deprecated functions in a `> [!WARNING]` block, and provide before/after migration examples.
   - For breaking changes (including changed defaults): show old vs new syntax, and list them in an `> [!important]` banner at the top of the notes.
   - Keep warnings concise: at most one warning box per section, combining related changes into a short list and a single before/after snippet. Don't repeat what the prose above already explains.
5. **Acknowledge new contributors** — keep the `## New Contributors` section as-is.
6. **Verify completeness** — confirm every PR from the original list appears exactly once in the final notes.
7. **Test code snippets and claims** — run all code snippets included in the release notes to verify they execute correctly. Run them against the code being released (e.g. a temporary `git worktree` of `origin/main`, put first on `sys.path`), not the local branch, which may be behind. When the notes say behaviour is unchanged (or changed) compared with the previous release, verify it by running both versions (a second worktree of the previous tag). Check that new docs links resolve: before the release, new pages only exist under `/dev/`, so check the `/dev/` URL while keeping `/latest/` in the notes.
8. **Save the GitHub release notes** to `release_notes_v<VERSION>.md` in the repo root — the full notes (all sections, PR bullets, New Contributors, Full Changelog link). Present this draft to the user and iterate on it until they explicitly approve it. Do not write the Zulip variant yet.
9. **Write the Zulip announcement only after the GitHub notes are approved.** Derive it once from the approved GitHub notes and save it to `release_zulip_v<VERSION>.md` in the repo root. It is a **short, skimmable announcement** (roughly 30–40 lines): the full details live in the GitHub notes, which it links to. Keep the same section headings as the GitHub notes for the highlights and any other sections with prose (e.g. notable new features or examples), but condense each one. Specifically:
     - Starts with `## Release [v<VERSION>](https://github.com/neuroinformatics-unit/movement/releases/tag/v<VERSION>) is out`, followed by the intro sentence (if any) and the update-guide link from the GitHub notes.
     - **Breaking changes and deprecations go in one line at the top**, e.g. `:warning: **This release contains minor breaking changes** to <areas>. See the [release notes](<GitHub release URL>) for how to update your code.` Do not repeat per-section warnings or migration code.
     - **Each section is 1–3 short sentences**: what's new, who to thank, and the key link(s) (user guide section, API page or example). Use plain descriptive link text (e.g. `[angular velocity](...)`) rather than full function signatures.
     - **Keep contributor credits** (@mentions) for the work described in each section, including thanks to mentors or co-authors named in the GitHub notes. Credits for work only mentioned in migration notes or PR lists can be left to the GitHub notes.
     - **No code snippets** by default. Include one only if the user asks; if so, use `Python` (capitalised) as the code fence language tag.
     - Omits feature bullet lists, "What's next?" paragraphs, implementation details, and minor improvements (e.g. better error messages, internal refactoring, bug fixes folded into a section).
     - **Uses Zulip mentions, not GitHub handles.** Zulip display names differ from GitHub handles. Look up each mentioned contributor in [`zulip_handles.md`](zulip_handles.md) and use the `@**Display Name**` form listed there. For anyone not listed, suggest a candidate from their GitHub profile name (`gh api users/<handle> -q .name`) and ask the user to confirm; if unconfirmed or not on Zulip, keep the plain GitHub handle. If the user is unsure, help them match identities: the user can see Zulip emails, which can be compared with commit author emails (`git log --format='%an <%ae>'`) or the public GitHub profile email; if those are no-reply addresses, the website linked from the GitHub profile may show the full name. Ask the user to check the exact display name via Zulip autocomplete. Add newly confirmed mappings to `zulip_handles.md`.
     - Uses Zulip emoji syntax (`:high_voltage:` instead of ⚡️).
     - **No blockquote admonitions.** Zulip has no native admonitions (`> [!WARNING]` etc. won't render). Use inline emoji + bold text instead (`:warning:` for warnings, `:pushpin:` for notes, `:rotating_light:` for critical alerts).
     - Omits all PR bullet points and `**Pull requests:**` labels, list-only sections (bug fixes, housekeeping, etc.), and the New Contributors / Full Changelog sections. A community/contributor experience section is included (condensed) only if its GitHub version has prose about notable changes, such as new governance or contribution policies.
     - Ends with `See the [release notes](https://github.com/neuroinformatics-unit/movement/releases/tag/v<VERSION>) for more info and a full list of changes.`

   For example, a condensed section from the v0.18.0 announcement:

   ```
   ### :triangular_ruler: New kinematic measures: angular derivatives and path metrics

   You can now compute the [angular velocity](...) and [angular acceleration](...) of 2D orientations, such as head direction. See the new [example](...) for a walkthrough.

   Thanks to @**Sanjana** and @**Vybhav**, there are also two new path metrics: [sinuosity](...) and [maximum expected displacement](...).
   ```

## Section format

Use these sections as a starting point. Each section header uses an emoji prefix. Use your discretion — not every release will need all sections, and some releases may warrant additional or different ones depending on the PRs involved.

Commonly used sections (in typical order):

```
(optional one-sentence summary of the release)

> [!important]
> This release contains <minor> breaking changes to <areas>. Please read the notes below before updating.
(only if there are breaking changes or changed defaults)

To update movement to the latest version, see the [update guide](https://movement.neuroinformatics.dev/latest/user_guide/installation.html#update-the-package).

## What's Changed

### ⚡️ Highlight: <short description>
(descriptive text, code snippets, deprecation warnings, migration guide)

**Pull requests:**

* PR bullet ...

### <emoji> <thematic section, titled for this release>
(short prose, code snippets, warnings)

**Pull requests:**

* PR bullet ...

### 🐛 Bug fixes
* PR bullet ...

### 🤝 Community and contributor experience
(optional prose, e.g. on governance or contribution policies)

**Pull requests:**

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
- `🔌 Interoperability` — for changes that expand format support or help downstream tools integrate
- `🗑️ Removed functions` — when deprecated functions are removed (with links to replacements)
- `⚠️ Breaking changes` — when API changes require user action. Be careful not to over-categorise: additive changes (new functions, expanded validation, new warnings) are not breaking even if they touch core APIs.

## Categorisation guidelines

The structure is decided anew for each release, together with the user. Section titles, emojis, and whether a group of PRs becomes a highlight or a thematically named section all vary from release to release, so treat the following as defaults to propose, not fixed rules.

- **Highlights vs thematic sections**: Ask the user which PR(s) to highlight if they don't specify. Notable work can be presented either as a highlight (`⚡️ Highlight: ...`, numbered only if there are several) or as its own section with a descriptive heading and a fitting emoji. This is a judgement call to settle with the user each time; propose options and expect it to change while iterating. For example, v0.18.0 ended up with one highlight and thematic sections such as `💾 Unified save_dataset entry point and other I/O improvements`, `📐 New kinematic measures: ...` and `🦫 New example: ...`.
- **Group by feature area where possible.** Bug fixes, refactors, docs and tests that relate to a highlight or thematic section usually read better in that section than under a generic heading. Fixes to a feature introduced in the same release fit naturally with that feature.
- **Bug fixes**: Fixes to incorrect behaviour that don't fit a thematic section. If only one or two remain, consider moving them elsewhere (e.g. a low-impact edge case to housekeeping) and dropping the section.
- **Community and contributor experience**: Governance, contributing guide updates, team changes, benchmarks, dev tooling, tests, and CI reliability (e.g. more reliable sample data hosting or fetching). Add prose if there are notable changes such as new governance or contribution policies.
- **Documentation**: Link, typo and docs build fixes can go in housekeeping. A notable new example or guide may deserve its own section; new docs for a feature can go with that feature.
- **Housekeeping**: Bot PRs (pre-commit, dependabot, contributors-readme), CI/CD changes, dependency pinning, linkcheck fixes, license metadata, minor docs fixes.

Sections with only one or two PRs are a hint that they might fit better elsewhere. Use your judgement for edge cases, and when a PR could fit multiple sections, prefer the more specific one.

## Example

This example predates some of the conventions above (e.g. "Pull requests:" labels and grouping by feature area). It illustrates the general approach, not a fixed structure to copy.

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

**Pull requests:**

* Unify loaders by @lochhh in https://github.com/neuroinformatics-unit/movement/pull/722

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
- Use `gh pr view <number> --json title,body,author 2>/dev/null` to fetch PR details — bare `gh pr view` throws a GraphQL deprecation error. Run these calls one at a time, not in parallel (parallel calls cause cancellations). Read relevant source files alongside the PR body to understand the new APIs.
- Present a draft of the GitHub notes to the user for review; iterate based on their feedback. Only write the Zulip announcement once the user has approved the GitHub notes, so there is no need to keep two files in sync while iterating.
- The user may edit the GitHub notes file directly between turns. Re-read it before making further changes, and build on their edits rather than reverting them.
- If the GitHub notes change after the Zulip announcement has been written, update the Zulip file without waiting to be asked wherever the change affects what it covers (section headings, credits, key links, the list of breaking changes). Changes to details the Zulip announcement omits need no sync.
