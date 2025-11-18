# Installation Guide for Common Package

## Overview

The `common` package supports **selective installation** of submodules. This allows you to install only the dependencies you need for your specific use case.

## Quick Start

### For Scientific Manuscript Analysis (Recommended)

If you're using this package for RAMS data analysis in a scientific manuscript:

```bash
pip install -e /path/to/common[rams,trajectories,plotting]
```

This installs:
- ✓ RAMS atmospheric model utilities
- ✓ Trajectory analysis tools
- ✓ Matplotlib plotting utilities
- ✗ NOT PyVista (large dependency, not needed for manuscript work)
- ✗ NOT Blender integration
- ✗ NOT VDB tools

### For 3D Visualization Work

```bash
pip install -e /path/to/common[rams,pvplotting]
```

### For Everything

```bash
pip install -e /path/to/common[all]
```

## Available Installation Extras

| Extra          | What it includes | Use case |
|----------------|------------------|----------|
| `rams`         | pandas, xarray, metpy, pint, h5netcdf | RAMS atmospheric model analysis |
| `trajectories` | pandas, xarray, scipy, jinja2 | Trajectory calculations |
| `plotting`     | matplotlib, pandas | Basic plotting |
| `pvplotting`   | pyvista | 3D visualization |
| `blender`      | (minimal deps) | Blender rendering |
| `vdb`          | pyopenvdb | VDB volume conversion |
| `dev`          | pytest, jupyter | Development tools |
| `all`          | All of the above | Everything |

## Combining Multiple Extras

You can combine extras by separating them with commas:

```bash
# Common combination for analysis work
pip install -e .[rams,trajectories,plotting]

# For visualization + analysis
pip install -e .[rams,pvplotting]

# With development tools
pip install -e .[rams,dev]
```

## What Gets Imported?

The package automatically imports only the submodules you have installed:

```python
import common

# Always available (no dependencies)
common.utils

# Available if you installed [rams]
common.rams.eread_rams_output(...)
common.rams.calculate_thermodynamic_variables(...)

# Available if you installed [trajectories]
common.trajectories.calculate_trajectories(...)

# Available if you installed [plotting]
common.plotting.start_figure(...)

# Available if you installed [pvplotting]
common.pvplotting.create_plotter(...)
```

## Helpful Error Messages

If you try to use a submodule you haven't installed, the package will tell you at import time:

```python
import common
# Output:
# Loaded submodules: rams, plotting
# To install missing submodules (trajectories, pvplotting, vdb), use:
#     pip install -e .[trajectories,pvplotting,vdb]
```

## Testing Your Installation

Run the test script to see what's installed:

```bash
python test_installation.py
```

This will show you which submodules are available and which are missing.

## Examples

### Example 1: Lightweight Environment for Manuscript Analysis

Your manuscript project might have its own `environment.yml`:

```yaml
name: my-manuscript
dependencies:
  - python=3.10
  - pip
  - pip:
    - -e /path/to/common[rams,plotting]
```

This keeps your environment lean - no PyVista, Blender, or other heavy dependencies.

### Example 2: Full-Featured Visualization Environment

For interactive 3D visualization work:

```yaml
name: viz-work
dependencies:
  - python=3.10
  - pip
  - pip:
    - -e /path/to/common[rams,pvplotting,plotting]
```

### Example 3: Core Only (Minimal)

If you only need the utilities:

```bash
pip install -e /path/to/common
```

This installs only numpy + the core utils module.

## Upgrading

To add more submodules to an existing installation:

```bash
# You have [rams], want to add [plotting]
pip install -e .[rams,plotting]
```

## Special Environments

The package automatically detects special environments:

- **Blender**: Only imports blender module
- **VDB conda env**: Only imports vdb module
- **Regular Python**: Imports all installed submodules

## Troubleshooting

### "Module not found" errors

If you see:
```
AttributeError: module 'common' has no attribute 'pvplotting'
```

This means you haven't installed that submodule. Check the import message or run `test_installation.py` to see what's available.

### Installing new dependencies

After adding new extras, you may need to reinstall:

```bash
pip install -e .[rams,pvplotting] --force-reinstall --no-deps
```

## More Information

See [README.md](README.md) for full package documentation.
