(target-text-formatting)=
# Text Formatting Guide

This page documents the text formatting conventions used across `movement`'s
documentation, to help contributors produce consistent and readable content.

## Monospace (backtick) formatting

Use backtick (`` ` ``) monospace formatting for anything that is code or a
code-adjacent term. The sections below describe what should and should not be
wrapped in backticks.

### File extensions

Always wrap file extensions in backticks.

| ❌ Wrong | ✅ Correct |
|---------|-----------|
| `Use a py file or md file` | `` Use a `.py` file or `.md` file `` |
| `Edit the rst file` | `` Edit the `.rst` file `` |

### File paths and file names

Always wrap file paths and file names in backticks.

| ❌ Wrong | ✅ Correct |
|---------|-----------|
| `Edit docs/source/community/contributing.md` | `` Edit `docs/source/community/contributing.md` `` |
| `The file is named conftest.py` | `` The file is named `conftest.py` `` |

### Function names and method names

Always wrap function and method names in backticks. Include empty parentheses
to make it clear you are referring to a callable.

| ❌ Wrong | ✅ Correct |
|---------|-----------|
| `Call the fetch_dataset function` | `` Call the `fetch_dataset()` function `` |

### Variable names, parameter names, and CLI arguments

Always wrap variable names, parameter names, and CLI argument names in
backticks.

| ❌ Wrong | ✅ Correct |
|---------|-----------|
| `Set the fps parameter` | `` Set the `fps` parameter `` |
| `The source_software argument` | `` The `source_software` argument `` |

### Configuration keys, environment variables, and class names

Any configuration key, environment variable, or class name should be wrapped
in backticks.

| ❌ Wrong | ✅ Correct |
|---------|-----------|
| `Set EXCLUDE_MODULES in the script` | `` Set `EXCLUDE_MODULES` in the script `` |
| `The DATA_DIR variable` | `` The `DATA_DIR` variable `` |

---

## The `movement` package name

> `` `movement` `` is always written in monospace when referring to the
> software package or project, to distinguish it from the common English
> word "movement". **This is intentional.**

When used in prose to mean "animal movement" or "bodily motion" (i.e. as a
plain English word and *not* as the name of the software), it should _not_
be wrapped in backticks.

| Context | Rendering |
|---------|-----------|
| Referring to the software package | `` `movement` aims to facilitate... `` |
| Referring to animal motion (biology) | `studying the movement of mice` |

---

## What not to format as monospace

- General English words, even if they share a name with a code concept.
- Proper nouns that are not code terms (e.g. software brands like "DeepLabCut"
  or "SLEAP" unless in a code context).
- URLs or hyperlink anchor text (use Markdown link syntax instead).
