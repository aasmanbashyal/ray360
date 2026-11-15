# Publishing a release

Releases use `uv` and PyPI trusted publishing from
[`publish.yml`](../.github/workflows/publish.yml). No long-lived API token is needed.

## First-time PyPI setup

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

1. Update the version in `pyproject.toml` and `src/ray360/__init__.py`, then run
   `uv lock`. Keep the existing `0.1.0` for the first release.
2. Commit the release changes, push `main`, and wait for the tests workflow.
3. Create and push an annotated tag matching the package version:

   ```bash
   git tag -a v0.1.0 -m "Ray360 0.1.0"
   git push origin v0.1.0
   ```

4. The publishing workflow checks the version, runs tests, builds the sdist and
   wheel, validates their metadata, and checks both isolated wheel installations.
   A separate job uploads those same artifacts to PyPI using a short-lived token.
5. Verify installation from the index outside the source checkout:

   ```bash
   uv run --no-project --with "ray360[image]==0.1.0" --refresh-package ray360 python -I -c "import ray360; print(ray360.__version__)"
   ```

If an upload is interrupted, rerun the failed workflow. For an unchanged build,
`uv publish` skips identical files that already exist. Published files cannot be
replaced with a different build under the same version.
