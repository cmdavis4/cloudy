"""
Matplotlib function hooks for capturing plotting data.

This module provides a system to hook into matplotlib functions and store
the data passed to them in a global dictionary.
"""

import matplotlib.pyplot as plt
import matplotlib.axes
import xarray as xr
import functools
import inspect
from typing import (
    Dict,
    Any,
    Callable,
    List,
    Tuple,
    Optional,
    TypeAlias,
    Union,
    Iterable,
)
from dataclasses import dataclass, field
from pathlib import Path
import pickle as pkl

from .core_plotting import fig_multisave

PathLike: TypeAlias = Union[str, Path]

HOOKED_PYPLOT_FUNCTIONS = [
    "plot",
    "scatter",
    "bar",
    "hist",
    "imshow",
    "contour",
    "contourf",
    "pcolormesh",
    "fill_between",
    "errorbar",
    "boxplot",
    "violin",
    "pie",
    "polar",
    "loglog",
    "semilogx",
    "semilogy",
]

HOOKED_AXES_METHODS = [
    "plot",
    "scatter",
    "bar",
    "hist",
    "imshow",
    "contour",
    "contourf",
    "pcolormesh",
    "fill_between",
    "errorbar",
    "boxplot",
    "violin",
]


@dataclass
class FigureData:
    figure_name: str
    figure_data: Dict = field(default_factory=dict)
    verbose: Optional[bool] = False

    @property
    def is_active(self):
        global ACTIVE_FIGUREDATA
        return ACTIVE_FIGUREDATA is self

    def save_figuredata(self, output_path):
        with Path(output_path).open("wb") as f:
            pkl.dump(self.figure_data, f)

    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"FigureData(figure_name={self.figure_name}, artist names="
            f"{list(self.figure_data.keys())})"
        )


HOOKS_INSTALLED = False
ACTIVE_FIGUREDATA = None


def set_active_figuredata(figuredata):
    global ACTIVE_FIGUREDATA
    ACTIVE_FIGUREDATA = figuredata


def start_figure(figure_name, verbose=False):
    if not HOOKS_INSTALLED:
        install_all_hooks()
    figuredata = FigureData(figure_name=figure_name, verbose=verbose)
    set_active_figuredata(figuredata=figuredata)
    return figuredata


def save_figuredata(output_path, figuredata: Optional[FigureData] = None):
    return (figuredata or ACTIVE_FIGUREDATA).save_figuredata(output_path=output_path)


def store_call_data(
    *args, name: str, is_method: bool, convert_xarray_to_numpy: bool = True
) -> None:
    """Store function call data in the global dictionary."""
    if name in ACTIVE_FIGUREDATA.figure_data:
        raise ValueError(f"Artist name {name} is already present in this figure's data")
    # If this is a method, drop the first arg
    if is_method:
        args = args[1:]
    if convert_xarray_to_numpy:
        args = [
            (
                x.values
                if (isinstance(x, xr.DataArray) or isinstance(x, xr.Dataset))
                else x
            )
            for x in args
        ]
    ACTIVE_FIGUREDATA.figure_data[name] = tuple(args)
    ACTIVE_FIGUREDATA.print(f"Stored data for artist {name}")


def save_figure_and_data(
    figure_output_dirs: Union[Iterable[PathLike], PathLike],
    data_output_dirs: Union[Iterable[PathLike], PathLike],
    figuredata: Optional[FigureData] = None,
    clear_active_figuredata=True,
    **fig_multisave_kwargs,
):
    figuredata = ACTIVE_FIGUREDATA or figuredata

    # Get the figure
    fig = plt.gcf()

    # Save it
    fig_multisave(
        fig=fig,
        name=figuredata.figure_name,
        dirs=figure_output_dirs,
        **fig_multisave_kwargs,
    )

    # Save the data too
    if not isinstance(data_output_dirs, list):
        data_output_dirs = [
            data_output_dirs,
        ]
    for output_dir in data_output_dirs:
        figuredata.save_figuredata(
            (Path(output_dir) / figuredata.figure_name).with_suffix(".pkl")
        )

    if clear_active_figuredata:
        set_active_figuredata(None)

    # This will still return the figuredata object we used for the call, even if
    # we cleared it
    return figuredata


def create_hooked_fn(original_func: Callable, is_method: bool) -> Callable:
    """Create a hook wrapper around a matplotlib function."""

    # Get the names of keyword arguments to store_call_data programatically
    # so we can remove them
    hook_kwargs = [
        x for x in inspect.signature(store_call_data).parameters.keys() if x != "args"
    ]

    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        original_func_kwargs = {k: v for k, v in kwargs.items() if k not in hook_kwargs}
        store_call_data_kwargs = {k: v for k, v in kwargs.items() if k in hook_kwargs}
        if "name" in store_call_data_kwargs:
            # If they don't pass artist_name then we don't save the data
            store_call_data(*args, **store_call_data_kwargs, is_method=is_method)

        # Call the original function
        return original_func(*args, **original_func_kwargs)

    return wrapper


def install_pyplot_hooks() -> None:
    """
    Install hooks on specified pyplot functions.

    Args:
        functions: List of function names to hook. If None, hooks common plotting functions.
    """
    for func_name in HOOKED_PYPLOT_FUNCTIONS:
        if hasattr(plt, func_name):
            original_func = getattr(plt, func_name)
            hooked_func = create_hooked_fn(original_func, is_method=False)
            setattr(plt, func_name, hooked_func)


def install_axes_hooks() -> None:
    """
    Install hooks on Axes methods.

    Args:
        functions: List of method names to hook. If None, hooks common plotting methods.
    """
    for func_name in HOOKED_AXES_METHODS:
        if hasattr(matplotlib.axes.Axes, func_name):
            original_func = getattr(matplotlib.axes.Axes, func_name)
            hooked_func = create_hooked_fn(original_func, is_method=True)
            setattr(matplotlib.axes.Axes, func_name, hooked_func)


def install_all_hooks():
    """Install hooks on both pyplot and axes functions."""
    print("Installing matplotlib hooks...")
    install_pyplot_hooks()
    install_axes_hooks()
    print("All hooks installed successfully!")
    global HOOKS_INSTALLED
    HOOKS_INSTALLED = True


# Example usage functions
def example_usage():
    """Demonstrate how to use the hooks."""
    # Install hooks
    install_all_hooks()

    # Create some plots
    import numpy as np

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.figure()
    plt.plot(x, y, label="sin(x)")
    plt.scatter([1, 2, 3], [0.5, 0.8, 1.2])

    # Check captured data
    print_captured_summary()

    # Get specific data
    plot_data = get_figure_data("plt.plot")
    print(f"Plot calls: {len(plot_data)}")
    if plot_data:
        print(f"First plot call args: {plot_data[0]['args']}")
        print(f"First plot call kwargs: {plot_data[0]['kwargs']}")
