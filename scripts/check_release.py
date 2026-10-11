"""Refuse to publish a release whose tag, version, CHANGELOG.md and CITATION.cff disagree.

wheels.yml runs this before it uploads anything:

    python scripts/check_release.py --tag v0.1.0   # a pushed tag, before PyPI
    python scripts/check_release.py                # the TestPyPI dry run (no tag)

It checks that

* the tag is ``v<version>``, where ``<version>`` is ``pymcpu.__version__``
  (read from pymcpu/__init__.py, so nothing has to be built or imported);
* CHANGELOG.md has a ``## [<version>] - YYYY-MM-DD`` heading;
* CITATION.cff has ``version: <version>`` and a ``date-released``;
* both dates are real dates, not the ``YYYY-MM-DD`` placeholder, and the same.

Exit status 0 means ready to publish; 1 means not, with every problem listed.
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _version(root: Path) -> str:
    text = (root / "pymcpu" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r"^__version__\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
    if not match:
        raise SystemExit("pymcpu/__init__.py sets no __version__")
    return match.group(1)


def _cff_value(text: str, key: str) -> str | None:
    """A top-level ``key: value`` of CITATION.cff, without quotes or comment."""
    match = re.search(rf"^{key}:[ \t]*([^#\n]*)", text, re.MULTILINE)
    return match.group(1).strip().strip("\"'") if match else None


def _is_date(value: str | None) -> bool:
    try:
        datetime.date.fromisoformat(value or "")
    except ValueError:
        return False
    return True


def problems(root: Path, tag: str | None) -> list[str]:
    """Every reason not to publish the tree at ``root`` as ``tag``."""
    version = _version(root)
    found = []
    if tag is not None and tag != f"v{version}":
        found.append(f"the tag is {tag!r}, but pymcpu.__version__ is {version!r}: "
                     f"the tag must be 'v{version}'")

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.search(rf"^## \[{re.escape(version)}\] - (\S+)[ \t]*$", changelog,
                        re.MULTILINE)
    changelog_date = heading.group(1) if heading else None
    if not heading:
        found.append(f"CHANGELOG.md has no '## [{version}] - YYYY-MM-DD' heading")
    elif not _is_date(changelog_date):
        found.append(f"CHANGELOG.md gives {version} the date {changelog_date!r}; "
                     f"set the release date")

    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    cff_version = _cff_value(citation, "version")
    cff_date = _cff_value(citation, "date-released")
    if cff_version != version:
        found.append(f"CITATION.cff has version {cff_version!r}, not {version!r}")
    if not _is_date(cff_date):
        found.append(f"CITATION.cff has date-released {cff_date!r}; set the release date")

    if _is_date(changelog_date) and _is_date(cff_date) and changelog_date != cff_date:
        found.append(f"the release dates differ: CHANGELOG.md {changelog_date}, "
                     f"CITATION.cff {cff_date}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", help="the git tag being published, e.g. v0.1.0")
    parser.add_argument("--root", type=Path, default=REPO,
                        help="the source tree to check (default: this checkout)")
    args = parser.parse_args(argv)

    found = problems(args.root, args.tag)
    for problem in found:
        print(f"not ready: {problem}")
    if found:
        return 1
    print(f"ready to publish pymcpu {_version(args.root)}"
          + (f" as {args.tag}" if args.tag else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
