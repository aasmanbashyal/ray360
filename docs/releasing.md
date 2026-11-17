# Publishing a release

Releases use `uv` and PyPI trusted publishing from
[`publish.yml`](../.github/workflows/publish.yml). No long-lived API token is needed.

## First-time PyPI setup

The Ray360 publisher is already configured. This section records the setup for
maintainer reference; do not add a duplicate pending publisher for an existing project.

Create a PyPI account, verify its email, and enable two-factor authentication.
In [account publishing settings](https://pypi.org/manage/account/publishing/),
add a pending GitHub publisher with these values:

| Field | Value |
| --- | --- |
| PyPI project name | `ray360` |
| Owner | `aasmanbashyal` |
| Repository | `ray360` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

The first successful workflow upload creates the PyPI project under the account
that registered the pending publisher. See the
[PyPI documentation](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## Release steps

1. Choose a new version, update it in `pyproject.toml` and `src/ray360/__init__.py`,
   then run `uv lock`. Move the relevant Unreleased entries in [the changelog](changelog.md)
   under that version. Version `0.1.0` has already been published and must not be reused.
2. Commit the release changes, push `main`, and wait for the tests workflow.
3. Create and push an annotated tag matching the package version. For example,
   after preparing version `0.1.1`:

   ```bash
   git tag -a v0.1.1 -m "Ray360 0.1.1"
   git push origin v0.1.1
   ```

4. The publishing workflow checks the version, runs tests, builds the sdist and
   wheel, validates their metadata, and checks both isolated wheel installations.
   A separate job uploads those same artifacts to PyPI using a short-lived token.
5. Verify installation from the index outside the source checkout:

   ```bash
   uv run --no-project --with "ray360[image]==0.1.1" --refresh-package ray360 python -I -c "import ray360; print(ray360.__version__)"
   ```

6. Create a GitHub release for that existing tag, summarizing changes, installation,
   and any migration notes. Attach the exact distributions from the successful
   publish workflow or PyPI, with their SHA-256 checksums. Do not rebuild the
   same version to populate GitHub release assets.

README changes on GitHub appear immediately after merging. PyPI retains the
README packaged with each release; a newer README reaches PyPI with the next
version. A GitHub release does not create a GitHub Packages entry. Python
distributions are hosted on PyPI.

If an upload is interrupted, rerun the failed workflow. For an unchanged build,
`uv publish` skips identical files that already exist. Published files cannot be
replaced with a different build under the same version.

## Publish the optional Rust extension

`ray360-native` has its own releases. Its first PyPI upload needs a separate
[pending Trusted Publisher](https://pypi.org/manage/account/publishing/) for
project `ray360-native`, GitHub owner `aasmanbashyal`, repository `ray360`,
workflow `publish-native.yml`, and environment `pypi`.

1. Set the same native version in `native/pyproject.toml`, `native/Cargo.toml`,
   and `native/Cargo.lock`. Push the changes to `main`.
2. Run [the native workflow](../.github/workflows/publish-native.yml) manually
   on `main` to build and test the wheels without publishing.
3. Create and push an annotated tag in the form `native-v<VERSION>` on the
   tested commit. The workflow builds, tests, and publishes wheels for CPython
   3.11–3.14 on Linux x86-64 and ARM64 with glibc 2.28 or later, macOS Intel and
   Apple Silicon, and Windows x86-64.
4. Install the released version outside the checkout with
   `uv pip install --only-binary :all: 'ray360-native[ray360]==<VERSION>'` and
   verify that Ray360 can load the Rust sampler.
