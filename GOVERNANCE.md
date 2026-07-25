# Project Governance

This document describes how the jaxfolio project is run: who makes decisions, how
they are made, and what users and contributors can expect regarding maintenance.

## Roles

### Users
Anyone who uses jaxfolio. Users contribute by filing issues, joining
discussions, and giving feedback. No formal responsibilities.

### Contributors
Anyone who submits a pull request, issue, documentation change, or review.
Contributions are governed by [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md).

### Maintainers
Maintainers have write access and are responsible for reviewing and merging
changes, triaging issues, cutting releases, and stewarding the project's
direction. Current maintainers are listed in [`.github/CODEOWNERS`](.github/CODEOWNERS).

## Decision-making

jaxfolio uses a **maintainer-led, lazy-consensus** model:

- Routine changes (bug fixes, docs, tests, small features) are merged by any
  maintainer after review.
- Substantial or cross-cutting changes (new public API, breaking changes,
  dependency policy, architectural shifts) should start as an issue or discussion
  so they can be reviewed openly. If no maintainer objects within a reasonable
  window, the proposal proceeds ("lazy consensus"). Disagreements are resolved by
  discussion; if consensus can't be reached, the maintainers decide.
- Anyone may propose changes; you do not need to be a maintainer to have
  influence — well-argued proposals and quality PRs carry weight.

## Becoming a maintainer

Contributors who show sustained, high-quality involvement — good PRs, helpful
reviews, constructive issue triage — may be invited to become maintainers by the
existing maintainers. There is no fixed quota. If you are interested, the best
path is simply to keep contributing; you may also express interest via a
discussion.

## Releases

- The project follows [Semantic Versioning](https://semver.org) and documents API
  stability and deprecation in [docs/reference/stability.md](docs/reference/stability.md).
- Releases are cut from `main` by tagging `vX.Y.Z`, which triggers the automated
  PyPI publish and GitHub release workflow (`.github/workflows/release.yml`).
- The `CHANGELOG.md` is generated from Conventional Commits and curated before
  release.
- Cadence is **as-needed**: patch releases when fixes accumulate, minor releases
  for new features. We do not commit to a fixed calendar schedule pre-1.0.

## Maintenance commitment

jaxfolio is actively maintained on a best-effort basis by its maintainers. In
practice this means:

- Security reports are handled per [SECURITY.md](SECURITY.md).
- Issues and PRs are triaged as maintainer time allows; we aim to respond to new
  issues within roughly two weeks.
- CI keeps the supported OS/Python matrix green (Ubuntu, macOS, Windows across
  Python 3.11–3.13).

This is an open-source project offered under the [MIT License](LICENSE) with **no
warranty**; the commitment above describes intent and best effort, not a
contractual service-level guarantee. Organizations needing formal support
guarantees should plan accordingly (e.g. pinning versions via `uv.lock` and
budgeting for in-house maintenance).

## Changing this document

Changes to governance follow the same process as substantial changes: propose via
issue/PR, allow for maintainer review and lazy consensus.
