"""Clouded: Cloud and atmospheric data utilities for atmospheric modeling and visualization.

This package provides tools for:
- RAMS (Regional Atmospheric Modeling System) data processing
- 3D visualization with PyVista and Blender
- Trajectory analysis and calculation
- VDB volume data conversion
- General plotting and data utilities

Default installation includes:
    pip install -e .                 # Includes: rams, plotting, utils, types_core

Optional extras:
    pip install -e .[trajectories]   # Trajectory analysis
    pip install -e .[pvplotting]     # PyVista 3D visualization
    pip install -e .[blender]        # Blender integration
    pip install -e .[vdb]            # VDB volume conversion
    pip install -e .[all]            # Everything

The package automatically detects the environment and imports appropriate modules.
In VDB environments, only VDB-related functionality is imported to avoid conflicts.
"""

__version__ = "1.0.0"

# Environment detection and conditional imports
import os
import sys
from typing import TYPE_CHECKING

# Always import core modules (utils, types_core)
from . import utils
from . import types_core

__all__ = ["utils", "types_core"]


def _is_running_in_blender():
    """Detect if we're running from the Blender Python executable."""
    try:
        import bpy
        return True
    except ImportError:
        return False


def _is_running_in_vdb():
    """Detect if we're running in the VDB conda environment."""
    return os.environ.get("CONDA_DEFAULT_ENV") == "vdb" or "envs/vdb" in sys.executable


# Conditional imports based on environment detection
_is_blender = _is_running_in_blender()
_is_vdb = _is_running_in_vdb()
_failed_imports = {}  # Track failed imports for all environments

if _is_blender:
    # Blender environment - only import blender module
    try:
        from . import blender
        __all__ += ["blender"]
        print("Detected Blender environment, imported blender module")
    except ImportError as e:
        print(f"Warning: Could not import blender module: {e}")

elif _is_vdb:
    # VDB environment - only import vdb module (very touchy about dependencies)
    try:
        from . import vdb
        __all__ += ["vdb"]
    except ImportError as e:
        pass

else:
    # Regular Python environment - import default modules and try optional ones
    _available_modules = []

    # Import default modules (should always be available)
    try:
        from . import rams
        _available_modules.append("rams")
    except ImportError as e:
        _failed_imports["rams"] = str(e)

    try:
        from . import plotting
        _available_modules.append("plotting")
    except ImportError as e:
        _failed_imports["plotting"] = str(e)

    # Try importing optional submodules
    try:
        from . import trajectories
        _available_modules.append("trajectories")
    except ImportError as e:
        _failed_imports["trajectories"] = str(e)

    try:
        from . import pvplotting
        _available_modules.append("pvplotting")
    except ImportError as e:
        _failed_imports["pvplotting"] = str(e)

    try:
        from .vdb import vdb
        _available_modules.append("vdb")
    except ImportError as e:
        _failed_imports["vdb"] = str(e)

    __all__ += _available_modules

# Export main types for convenience
if TYPE_CHECKING:
    from .types_core import BlenderObject, BlenderCollection, PathLike, ConfigDict


def __getattr__(name: str):
    """
    Handle access to missing submodules with informative error messages.

    This is called when an attribute is not found through normal lookup.
    If the user tries to access an uninstalled optional module, we provide
    helpful installation instructions.
    """
    # Define which modules are optional vs core
    optional_install_commands = {
        "trajectories": "pip install -e .[trajectories]",
        "pvplotting": "pip install -e .[pvplotting]",
        "vdb": "pip install -e .[vdb]",
    }

    core_modules = ["rams", "plotting", "utils", "types_core"]

    # Check if this is a known submodule that failed to import
    if name in _failed_imports:
        original_error = _failed_imports[name]

        if name in optional_install_commands:
            # This is an optional module
            raise ImportError(
                f"The '{name}' submodule is not installed. "
                f"To install it, run:\n    {optional_install_commands[name]}\n\n"
                f"Original error: {original_error}"
            )
        elif name in core_modules:
            # This is a core module that should have been installed
            raise ImportError(
                f"The '{name}' submodule failed to import. This is a core module that "
                f"should be installed by default. Try reinstalling the package:\n"
                f"    pip install -e .\n\n"
                f"Original error: {original_error}"
            )

    # If we're here, it's truly an unknown attribute
    raise AttributeError(f"module 'clouded' has no attribute '{name}'")
