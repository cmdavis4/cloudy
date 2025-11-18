# Installation Issue Fix Summary

## Problem Identified

The package couldn't be imported after installation because of a **name collision** with an existing PyPI package called `common` (https://pypi.org/project/common/).

### Symptoms:
- `pip install ~/projects/common` ran without errors
- `python -c "import common"` failed with `ModuleNotFoundError: No module named 'common'`
- Import only worked when running Python from within the source directory

## Root Cause

There's an existing package on PyPI named `common`, and when pip tried to install your local package with the same name, it caused conflicts in the package registry.

## Solution

**Renamed the package from `common` to `clouded`**

### Key Points:
- **Installation name**: `clouded` (what you use with pip)
- **Import name**: `clouded` (what you use in Python code)
- **Recommended alias**: `import clouded as cd`
- Both package name and import name are now `clouded`

## Changes Made

1. **Directory renamed**: `common/` → `clouded/`
2. **pyproject.toml**: Changed `name = "common"` to `name = "clouded"` and updated package discovery
3. **README.md**: Updated all documentation
4. **Import name**: Changed from `import common` to `import clouded`

## Installation Instructions

### On your pixi machine:

```bash
# 1. Uninstall any conflicting packages
pixi run pip uninstall -y common clouded

# 2. Install your packages
pixi run pip install --config-settings editable-mode=strict -e ~/projects/ps/ ~/projects/common

# 3. Verify installation (run from ANY directory, not from the source!)
cd /tmp
pixi run python -c "import clouded as cd; print('✓ Success!'); print('Version:', cd.__version__)"
```

### Verification checklist:

✓ Run the test from `/tmp` or another directory (NOT from `~/projects/common`)
✓ Should see "Success!" message
✓ Package should be located in pixi's site-packages, not the source directory

## Why The Test Location Matters

When you run Python from `~/projects/common`, it can import from the local `./common/` subdirectory even if the package isn't properly installed. Testing from `/tmp` ensures you're importing the installed package, not the source.

## Debugging

If issues persist, run the diagnostic script:

```bash
cd /tmp  # Important!
pixi run python ~/projects/common/debug_install.py
```

Look for:
- ✓ Package should be in site-packages (not source directory)
- ✓ Package name should be `clouded`
- ✓ Import location should be in `.../site-packages/clouded/__init__.py`
