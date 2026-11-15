# Contributing to Ray360

Ray360 focuses on geometry for computer vision on already-stitched 360 media.
Contributions should make that workflow easier to use, more correct, or measurably
faster while keeping the NumPy-only core small.

## Choose the right place

| What you need | Where to start |
| --- | --- |
| Help using the API or an integration idea | [Discussions](https://github.com/aasmanbashyal/ray360/discussions) |
| Reproducible incorrect behavior | [Bug report](https://github.com/aasmanbashyal/ray360/issues/new?template=bug_report.yml) |
| A concrete missing capability | [Feature request](https://github.com/aasmanbashyal/ray360/issues/new?template=feature_request.yml) |
| A small documentation or code correction | Open a pull request; an issue is optional |
| A potential vulnerability | Follow [SECURITY.md](SECURITY.md) |

For new backends, annotation types, dependencies, or changes to the public API,
discuss the use case before implementing it. Include a consuming application and
the limitation you encountered. Read [the scope](docs/scope.md) first.

## Set up a development checkout

Fork the repository on GitHub, then clone your fork. Replace `YOUR-USERNAME` below:

```bash
git clone https://github.com/YOUR-USERNAME/ray360.git
cd ray360
git remote add upstream https://github.com/aasmanbashyal/ray360.git
git switch -c fix/short-description
uv sync --locked --group dev --extra image
uv run --no-sync python examples/basic.py
```

Use Python 3.11 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Branch names should describe the change, such as `fix/pole-boundary`,
`feat/region-export`, or `docs/video-example`. Do not use a `codex/` prefix.

## Make and check a change

Keep each pull request focused. Preserve the package boundary: inference and
decoding belong in applications and examples; runtime dependencies belong in the
core only when the geometry requires them.

Run these checks from the checkout:

```bash
uv run --no-sync ruff check src tests scripts examples benchmarks
uv run --no-sync ruff format --check src tests scripts examples benchmarks
uv run --no-sync mypy
uv run --no-sync python -m pytest
```

Use `uv run --no-sync ruff format <changed-files>` when formatting is needed.
For packaging or dependency changes, also run:

```bash
uv build --no-sources
uv run --no-sync python scripts/check_wheel.py
uv run --no-sync python scripts/check_wheel.py --image
```

Add a regression test for a bug fix. For geometry, prefer an independent expected
result, such as known directions, analytic regions, or carefully chosen seam and
pole cases. A round trip alone can hide matching errors in two transforms.
Documentation-only corrections need no new unit tests; run any examples you change.

The default suite uses synthetic inputs. Photo, video, and model checks are
optional; follow [the validation guide](docs/validation.md) when those integrations
are relevant. Do not commit downloaded media, model weights, virtual environments,
generated reports, or private recordings. Public fixtures need provenance,
redistribution information, and checksums in the existing manifest.

## Open a pull request

Push your branch to your fork and open a pull request against `main`. Draft pull
requests are welcome when you want feedback before finishing.

Explain the problem, the resulting behavior, and how you checked it. Include a
small before/after example when it makes a geometry change easier to review. Link
any related issue, document API changes, and add a migration example for breaking
changes. Update [the changelog](docs/changelog.md) for user-visible behavior.

CI checks Linux, macOS, Windows, multiple Python versions, minimum dependencies,
and installed distributions. A maintainer reviews correctness, scope, and the
public API before merging. Aasman Bashyal currently maintains the project and makes
release decisions. No response-time guarantee is implied.

## Working together

Describe the behavior you observed and the result you expected. Be specific in
reviews, welcome questions from newcomers, and keep criticism about the work.
Harassment, discriminatory language, and disclosure of someone else's private
information are not acceptable. Maintainers may remove disruptive content and
use GitHub moderation tools when needed.

Share the smallest reproduction you can. A synthetic array is often enough for a
geometry issue and avoids exposing private scenes, people, or credentials.
Contributions are distributed under the repository's [MIT license](LICENSE).
