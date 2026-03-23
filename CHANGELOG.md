# Changelog

## [0.1.7] - 2026-03-23

### Fixed
- Bumped numpy minimum version from >=1.24 to >=1.25 to ensure compatibility with scipy's use of numpy.exceptions (added in numpy 1.25)


## v0.1.5 (2026-03-22) [unreleased]
- fix: skip torch-dependent tests gracefully; drop unused packaging dep
- fix: use plain string license field for universal setuptools compatibility
- fix: use license text instead of file reference in pyproject.toml
- fix: use importlib.metadata for __version__ (prevents drift from pyproject.toml)

## v0.1.5 (2026-03-21)
- docs: replace pip install with uv add in README
- Add 50-feature benchmark demonstrating CANN+NID at scale
- Add community CTA to README
- docs: clarify torch extra is required for InteractionDetector quickstart
- Add real benchmark numbers to Performance section (Databricks 2026-03-16)
- fix: QA audit fixes — v0.1.3
- Fix P0/P1/P2 bugs: correct cat×cat interactions, rename AIC/BIC columns, fix SHAP crash, fix mixed-order NID dataframe (v0.1.4)
- Add standalone benchmark script
- fix: bump scipy to >=1.10 — drop upper cap that blocked Python 3.12 wheels
- Add Databricks Notebook section to README
- Fix: relax scipy/numpy constraints for Databricks serverless compat
- Add Related Libraries section to README
- docs: add Performance section with benchmark summary
- Add Databricks benchmark notebook: library vs main-effects GLM
- fix: add self-contained data generation to quickstart
- fix: update polars floor to >=1.0 and fix project URLs
- Fix eager torch import causing 0/26 tests to collect

## v0.1.0 (2026-03-09)
- test: mark NID integration test xfail (flaky with small CANN); use larger network
- polish README: add PyPI badge, blog link, related libraries cross-refs
- Add GitHub Actions CI workflow and test badge
- fix: update URLs to burning-cost org
- Replace uv pip install with uv add in error messages
- Update error message to reference CatBoost instead of LightGBM
- docs: switch examples to CatBoost/polars/uv, fix tone
- fix: all 46 tests passing on Databricks
- fix: standardise on CatBoost, uv, clean up style
- fix: uv references
- Add insurance-interactions library: CANN + NID automated GLM interaction detection

