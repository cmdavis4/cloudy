# Clouded

Cloud and atmospheric data utilities for atmospheric modeling and visualization, with a focus on RAMS (Regional Atmospheric Modeling System) data processing and analysis.

## Features

This package provides modular tools for:

- **RAMS** - RAMS atmospheric model data processing, analysis, and unit handling
- **Trajectories** - Trajectory analysis and calculation
- **Plotting** - Matplotlib-based plotting utilities
- **PyVista Plotting** - 3D visualization with PyVista
- **Blender** - Integration with Blender for high-quality rendering
- **VDB** - Volume data conversion to OpenVDB format
- **Utils** - Core utilities (always available)

## Installation

**Package name:** `clouded`

**Import:** `import clouded as cd`

The package supports **selective installation** - install only the submodules you need!

### Basic Installation (Core utilities only)

```bash
pip install -e .
# Or from path: pip install ~/projects/common
```

This installs the `clouded` package with core `rams`, `plotting`, and `utils` modules.

**Import usage:**
```python
import clouded as cd  # Recommended
# Or: import clouded
```

### Install Specific Submodules

For scientific manuscript analysis (recommended for most users):
```bash
pip install -e .[rams,trajectories,plotting]
```

For 3D visualization work:
```bash
pip install -e .[pvplotting]
```

For Blender rendering:
```bash
pip install -e .[blender]
```

For VDB conversion:
```bash
pip install -e .[vdb]
```

### Install Everything

```bash
pip install -e .[all]
```

### Development Installation

```bash
pip install -e .[dev]
```

This includes pytest, jupyter, and other development tools.

## Available Installation Extras

| Extra | Description | Key Dependencies |
|-------|-------------|------------------|
| `rams` | RAMS atmospheric modeling utilities | pandas, xarray, metpy, pint |
| `trajectories` | Trajectory analysis | pandas, xarray, scipy |
| `plotting` | Matplotlib plotting | matplotlib, pandas |
| `pvplotting` | PyVista 3D visualization | pyvista |
| `blender` | Blender integration | (minimal) |
| `vdb` | VDB volume conversion | pyopenvdb |
| `dev` | Development tools | pytest, jupyter |
| `all` | All submodules | (all of the above) |

## Usage

The package automatically imports only the submodules you have installed:

```python
import common

# Always available
common.utils

# Available if installed with [rams]
common.rams

# Available if installed with [trajectories]
common.trajectories

# Available if installed with [plotting]
common.plotting

# Available if installed with [pvplotting]
common.pvplotting
```

If you try to import a submodule you haven't installed, the package will tell you which extra to install.

## Example Workflows

### For a scientific manuscript using RAMS data

```bash
# In your manuscript project environment
pip install -e /path/to/common[rams,plotting]
```

```python
import common
import common.rams as rams

# Use RAMS utilities for analysis
ds = rams.eread_rams_output(...)
ds = rams.calculate_thermodynamic_variables(ds)
```

### For 3D visualization projects

```bash
pip install -e /path/to/common[rams,pvplotting]
```

```python
import common.pvplotting as pvp

# Use PyVista plotting
plotter = pvp.create_plotter(...)
```

## Environment Detection

The package automatically detects special environments:

- **Blender environment** - Only imports blender module
- **VDB environment** - Only imports vdb module (to avoid dependency conflicts)
- **Regular Python** - Imports all installed submodules

## Contributing

This is a personal utilities package but contributions are welcome. Make sure to run tests:

```bash
pip install -e .[dev,all]
pytest tests/
```

## License

MIT
