"""PyVista plotting module for atmospheric data visualization.

This module provides comprehensive 3D visualization capabilities using PyVista
for atmospheric modeling data including trajectories, isosurfaces, and animations.
Re-exports all public functions for backwards compatibility.
"""

# Core plotting functions
from .plotter import initialize_plotter
from .core_pvplotting import (
    get_subplot_keys,
    sanitize_inputs,
    plot_rams_and_trajectories,
    plot_trajectory_frame,
    animate_trajectories,
    rectangle_mesh,
    screenshot_render,
)

from .types_pvplotting import (
    PVConfig,
    PVRamsData,
    PVTrajectoryData,
    PV2DSpec,
    PVContourSpec,
    PVVectorSpec,
    PVTrajectorySpec,
)

# Plotter utilities
from .core_pvplotting import (
    add_mesh_to_subplots,
)

# Trajectory functions
from .trajectories_pvplotting import (
    create_trajectory_polydata,
    create_tetrahedron_head,
    create_trajectory_mesh,
    generate_trajectory_mesh,
)

# Camera functions
from .camera import (
    calculate_camera_offset,
    get_trajectory_camera,
    camera_follow_callback,
)

# Blender export functions
from .pv_to_blender import (
    export_meshes_to_blender,
)

# Main plotting function alias (for common usage)
plot_trajectories = plot_rams_and_trajectories
