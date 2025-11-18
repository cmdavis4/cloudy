"""Plotting utilities for atmospheric data visualization and analysis.

This module provides functions for creating faceted plots, animations, legends,
and specialized atmospheric plots like soundings and hodographs.
"""

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.figure as mplfig
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
from tqdm.notebook import tqdm
import matplotlib.animation as mplanim
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import metpy.calc as mpc
from metpy.units import units
from metpy.plots import SkewT, Hodograph
import matplotlib as mpl
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Iterable, Callable
import matplotlib.figure
import matplotlib.axes

from ..types_core import PathLike


def format_t_str(t: Any, strftime: str = "%Y-%m-%d %H:%M:%S") -> Any:
    """
    Format timestamp to string, handling numpy datetime64 objects.

    Args:
        t: Timestamp object (numpy datetime64, datetime, etc.)
        strftime: Format string for datetime formatting

    Returns:
        Formatted string or original object if formatting fails
    """
    try:
        return t.astype("datetime64[s]").item().strftime(strftime)
    except:
        # If it's not coerceable to a datetime, just leave it as is
        return t


def clean_legend(
    ax: matplotlib.axes.Axes,
    include_artists: Optional[List[Any]] = None,
    sort: bool = False,
    use_alphas: bool = False,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """
    Create a clean legend with customizable styling and sorting.

    Args:
        ax: Matplotlib axes to add legend to
        include_artists: List of specific artists to include in legend
        sort: Whether to sort legend entries by maximum values
        use_alphas: Whether to apply alpha values to legend text colors
        **kwargs: Additional keyword arguments passed to legend()

    Returns:
        The modified axes object
    """
    if include_artists is None:
        include_artists = []

    # Create a function to filter out artists that aren't in include_artists
    def filter_artists(artists):
        if not include_artists:
            return artists
        else:
            return [artist for artist in artists if artist in include_artists]

    # Want to handle lines, collections (i.e. scatter plots/points), and patches (i.e. histograms/polygons)
    # Get the max values of each line
    all_maxes = {
        line.get_label(): (
            {
                "max": max(line.get_ydata()),
                "color": line.get_color(),
                "alpha": line.get_alpha(),
            }
            if sort
            else {"color": line.get_color(), "alpha": line.get_alpha()}
        )
        for line in filter_artists(ax.get_lines())
    }
    collection_maxes = {}
    has_printed_not_implemented = False
    collections_to_iterate = []
    # for collection in filter_artists(ax.collections):
    #     if collection in collections_to_iterate:
    #         continue
    #     if isinstance(collection, mpl.contour.QuadContourSet):
    #         collections_to_iterate += [x for x in collection.collections]
    #     else:
    #         collections_to_iterate.append(collection)
    # for collection in collections_to_iterate:
    for collection in filter_artists(ax.collections):
        # Get the color; assume we use the edge color if the face is transparent
        has_facecolor = collection.get_facecolor() and collection.get_facecolor()[3]
        color = (
            collection.get_edgecolor() if has_facecolor else collection.get_facecolor()
        )
        try:
            collection_maxes[collection.get_label()] = (
                {
                    "max": max([x[1] for x in collection.get_offsets().data]),
                    "color": color,
                }
                if sort
                else {"color": color}
            )
        except NotImplementedError:
            if not has_printed_not_implemented:
                print(
                    "clean_legend not implemented for some elements of figure, skipping"
                )
                has_printed_not_implemented = True
    all_maxes.update(collection_maxes)
    # Need to iterate through patches to handle the alpha
    patches_dict = {}
    for patch in filter_artists(ax.patches):
        this_color = patch.get_facecolor()  # Works for a normal histogram
        if this_color[3] == 0:  # I.e. if the face is transparent, so a step histogram
            this_color = patch.get_edgecolor()
        patches_dict[patch.get_label()] = (
            {
                "max": max([xy[1] for xy in patch.get_xy()]),
                "color": this_color,
            }
            if sort
            else {"color": this_color}
        )
    all_maxes.update(patches_dict)
    handles, labels = ax.get_legend_handles_labels()
    if sort:
        # Get the right order of line names
        label_order_desc = [
            k
            for k, v in sorted(
                all_maxes.items(), key=lambda item: item[1]["max"], reverse=True
            )
        ]
        # Now get the existing legend
        # Get the indexes that the order we want corresponds to in the existing labels/handles
    else:
        label_order_desc = labels
    order = [labels.index(x) for x in label_order_desc]
    # Need to order the *handles* correctly so matplotlib can connect them to the actual lines,
    # even though we hide them
    labelcolors = [all_maxes[k]["color"] for k in label_order_desc]
    if use_alphas:
        new_labelcolors = []
        for k_ix, k in enumerate(label_order_desc):
            this_color = list(labelcolors[k_ix])
            this_color[3] = all_maxes[k]["alpha"]
            new_labelcolors.append(tuple(this_color))
        labelcolors = new_labelcolors
    legend = ax.legend(
        [handles[ix] for ix in order],
        [labels[ix] for ix in order],  # This is the same as line_order_desc
        # Hide the handles and make the text color match the line color
        handletextpad=0.0,
        handlelength=0.0,
        handleheight=0.0,
        markerscale=0.0,
        labelcolor=labelcolors,
        #         scatterpoints=0,
        **kwargs,
    )
    # Get rid of any remaining little rectangular blips
    for item in legend.legend_handles:
        item.set_visible(False)
    return ax


def contour_legend(contour_set, **kwargs):
    handles = [x for x in contour_set.legend_elements()[0]]
    return contour_set.figure.legend(
        handles=handles,
        # Hide the handles and make the text color match the line color
        handletextpad=0.0,
        handlelength=0.0,
        handleheight=0.0,
        markerscale=0.0,
        # labelcolor='black',
        # facecolor=contour_edgecolors,
        ncol=len(handles),
        loc="upper left",
        **kwargs,
    )


def get_nth_color(n):
    return plt.rcParams["axes.prop_cycle"].by_key()["color"][n]


def get_next_color(ax):
    return next(ax._get_lines.prop_cycler)["color"]


def get_cmap(name):
    """
    Get a matplotlib colormap object for a given colormap name.
    This is admittedly silly to make a function for, but I can never remember
    how to do it from matplotlib directly.

    Args:
        name (str): Name of the colormap

    Returns:
        matplotlib.colors.Colormap: The colormap object
    """
    return mpl.colormaps[name]


def add_row_header(ax, text, pad=None, **kwargs):
    """
    Add a row header to the left of an axes with automatic spacing.

    Args:
        ax: Matplotlib axes object
        text: Text for the row header (can be multi-line)
        pad: Manual padding offset. If None, calculated automatically
        **kwargs: Additional arguments passed to annotate()

    Returns:
        The annotation object
    """
    # Split text into lines to handle multi-line text
    lines = text.split("\n") if isinstance(text, str) else [str(text)]
    n_lines = len(lines)

    # Calculate automatic padding based on various factors
    if pad is None:
        # Base padding from ylabel
        base_pad = ax.yaxis.labelpad if hasattr(ax.yaxis, "labelpad") else 5

        # Additional padding for tick labels
        tick_width = 0
        if ax.yaxis.get_ticklabels():
            # Estimate width of tick labels
            tick_texts = [label.get_text() for label in ax.yaxis.get_ticklabels()]
            if tick_texts:
                max_tick_len = max(len(str(t)) for t in tick_texts if t)
                tick_width = max_tick_len * 8  # Approximate character width in points

        # Additional padding for ylabel if present
        ylabel_width = 0
        if ax.yaxis.get_label().get_text():
            ylabel_width = 20  # Approximate width for rotated ylabel

        # Extra padding for multi-line text
        multiline_pad = (n_lines - 1) * 10 if n_lines > 1 else 0

        # Calculate total padding
        pad = base_pad + tick_width + ylabel_width + multiline_pad + 15

    # Handle multi-line text positioning
    if n_lines > 1:
        # For multi-line text, adjust vertical alignment
        va = kwargs.get("va", "center")
        # Join lines back together for display
        display_text = "\n".join(lines)
    else:
        va = kwargs.get("va", "center")
        display_text = text

    # Set up annotation arguments with automatic spacing
    annotation_kwargs = {
        "xy": (0, 0.5),
        "xytext": (-pad, 0),
        "xycoords": "axes fraction",  # More reliable than ax.yaxis.label
        "textcoords": "offset points",
        "rotation": 90,
        "fontsize": kwargs.get("fontsize", 16),  # Slightly smaller default
        "ha": "center",
        "va": va,
        "fontweight": "bold",
    }

    # Update with user-provided kwargs
    annotation_kwargs.update(kwargs)

    return ax.annotate(display_text, **annotation_kwargs)


def shifted_colormap(cmap, new_range, n=256):
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    colors_list = cmap(np.linspace(new_range[0], new_range[1], n))
    return colors.LinearSegmentedColormap.from_list("new", colors_list)


# Define some shifted colormaps
shifted_blues = shifted_colormap("Blues", (0.2, 1.0))
shifted_greens = shifted_colormap("Greens", (0.3, 1.0))
shifted_oranges = shifted_colormap("Oranges", (0.3, 1.0))


def plot_sounding(ds: xr.Dataset) -> matplotlib.figure.Figure:

    # Exclude the fake level if present
    if ds["z"].values[0] < 0:
        ds = ds.isel(z=slice(1, len(ds["z"])))

    this_ds_mean = ds.squeeze().mean(["x", "y"])

    fig = plt.figure(figsize=(9, 9))
    skewt = SkewT(fig, rotation=30)
    skewt.plot(
        this_ds_mean["P"].values,
        (this_ds_mean["T"].values * units("K")).to("degC").magnitude,
        "r",
    )
    skewt.plot(
        this_ds_mean["P"].values,
        (this_ds_mean["dewpoint"].values * units("K")).to("degC"),
        "blue",
    )
    # fig.suptitle(sounding_time)

    # Calculate and plot parcel profile
    parcel_path = mpc.parcel_profile(
        this_ds_mean["P"].values * units.hPa,
        this_ds_mean["T"].values[0] * units.K,
        this_ds_mean["dewpoint"].values[0] * units.K,
    )
    skewt.plot(
        this_ds_mean["P"].values,
        parcel_path,
        color="grey",
        linestyle="dashed",
        linewidth=2,
    )

    # Create a hodograph
    ax_hod = inset_axes(skewt.ax, "40%", "40%", loc=1)
    h = Hodograph(ax_hod, component_range=35.0)
    h.add_grid(increment=10)
    h.plot_colormapped(
        this_ds_mean["UC"].values,
        this_ds_mean["VC"].values,
        this_ds_mean["z"].values * units("m"),
    )

    skewt.ax.set_xlabel("Temperature (°C)")
    skewt.ax.set_ylabel("Pressure (hPa)")

    ax_hod.set_xlabel("U (m/s)")
    ax_hod.set_ylabel("V (m/s)")

    fig.suptitle("Initial sounding")

    return fig


def fig_multisave(
    fig: matplotlib.figure.Figure,
    name: Union[str, PathLike],
    dirs: Union[PathLike, List[PathLike]],
    no_title_version: bool = False,
    resize_to_width: Optional[float] = None,
    extensions: Optional[List[str]] = None,
) -> matplotlib.figure.Figure:
    """
    Save a figure into multiple directories (with the same filename). That's easily accomplished
    with a for loop on its own, but the no_title_version flag also allows for the creation of two copies
    of the figure in each directory: one with its suptitle (if present), and one without. These copies
    will be suffixed with 'title-yes' and 'title-no' respectively.
    """
    # Make the dirs argument a list if a single directory was passed
    if not isinstance(dirs, list):
        dirs = [dirs]

    # Handle mutable default argument
    if extensions is None:
        extensions = [".pdf", ".png"]

    # Clean up the file extensions
    if isinstance(extensions, str):
        extensions = [extensions]
    extensions = [
        "." + extension if extension[0] != "." else extension
        for extension in extensions
    ]

    # Remove any file extensions from name
    name = Path(name).stem

    if resize_to_width:
        # Resize the figure with the same aspect ratio if we should
        current_size = fig.get_size_inches()
        current_ar = current_size[1] / current_size[0]
        fig.set_size_inches(resize_to_width, current_ar * resize_to_width)

    # Ignore the no_title_version argument if the figure doesn't have a suptitle
    if not fig._suptitle:
        no_title_version = False
    for this_dir in dirs:
        suffix = "_title-yes" if no_title_version else ""
        for extension in extensions:
            fig.savefig(this_dir.joinpath(name + suffix + extension))
    # Make a no title version if we're doing that
    if no_title_version:
        fig._suptitle.remove()
        fig._suptitle = None
        for this_dir in dirs:
            for extension in extensions:
                fig.savefig(this_dir.joinpath(name + f"_title-no{extension}"))
    return fig


def sequential_cmap(colors_list, name=None, N=512):
    colors_list = [
        colors_list.to_rgb(color) if isinstance(color, str) else color
        for color in colors_list
    ]
    return colors.LinearSegmentedColormap.from_list(
        name or f"cd_{str(colors)}", colors_list, N=N
    )


def single_color_cmap(color, linear_opacity=False, name=None, N=512):
    if isinstance(color, str):
        color = colors.to_rgb(color)
    start_color = (color[0], color[1], color[2], 0) if linear_opacity else (1, 1, 1, 1)
    return sequential_cmap([start_color, color], name=name, N=N)


def transparent_under_cmap(cmap, bad=True):
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    cmap = cmap.copy()
    cmap.set_under((0, 0, 0, 0))
    if bad:
        cmap.set_bad((0, 0, 0, 0))
    return cmap


def share_axes(axs, x=True, y=True):
    # Just share them all with the first one
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    base_ax = axs[0]
    for ax in axs[1:]:
        if x:
            ax.sharex(base_ax)
        if y:
            ax.sharey(base_ax)
    return axs


def scale_axes_ticks(ax, scale=1000, x=True, y=True):
    # Scale the horizontal axes to km rather than m
    ticks = mpl.ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / scale))
    if x:
        ax.xaxis.set_major_formatter(ticks)
    if y:
        ax.yaxis.set_major_formatter(ticks)


def prepend_axes_letters(
    axs: Union[matplotlib.axes.Axes, np.ndarray],
    start_letter: str = "a",
    format_string: str = "({}) ",
) -> Union[matplotlib.axes.Axes, np.ndarray]:
    """
    Prepend letters to the title of each axes in row-major order.

    Given a set of axes (as returned by plt.subplots), this function prepends
    sequential letters to each axes title. The lettering proceeds in row-major
    order (left-to-right within each row, then top-to-bottom across rows).

    Args:
        axs: Single axes or array of axes from plt.subplots()
        start_letter: The starting letter (default: 'a')
        format_string: Format string for the letter label. Use {} as placeholder
                      for the letter (default: '({}) ' produces '(a) ', '(b) ', etc.)

    Returns:
        The modified axes (same type as input)

    Examples:
        >>> fig, axs = plt.subplots(2, 3)
        >>> prepend_axes_letters(axs)
        # Produces: (a), (b), (c) in first row
        #           (d), (e), (f) in second row

        >>> fig, ax = plt.subplots()
        >>> ax.set_title("My Plot")
        >>> prepend_axes_letters(ax)
        # Produces: (a) My Plot
    """
    # Handle single axes case
    if isinstance(axs, matplotlib.axes.Axes):
        axs_flat = [axs]
    else:
        # Flatten in row-major order (C-style)
        axs_flat = axs.flatten() if isinstance(axs, np.ndarray) else axs

    # Get the starting letter as an integer (a=0, b=1, etc.)
    start_ord = ord(start_letter.lower())

    # Iterate through axes in row-major order
    for i, ax in enumerate(axs_flat):
        # Generate the letter label
        letter = chr(start_ord + i)
        label = format_string.format(letter)

        # Get current title and prepend the letter
        current_title = ax.get_title()
        new_title = label + current_title
        ax.set_title(new_title)

    return axs


def subplots_with_legends(
    nrows: int = 3,
    ncols: int = 3,
    legend_positions: Optional[Dict[int, str]] = None,
    subplot_size: Tuple[float, float] = (3.0, 2.5),
    legend_width: float = 1.2,
    legend_height: float = 0.8,
    wspace: float = 0.3,
    hspace: float = 0.4,
    sharex: Union[bool, str] = False,
    sharey: Union[bool, str] = False,
    xlabel_space: float = 0.5,
    **subplot_kw: Any,
) -> Tuple[matplotlib.figure.Figure, np.ndarray]:
    """
    Create a grid of subplots with automatic sizing for row-specific legends.

    This function creates a flexible subplot grid where each row can optionally have
    a legend positioned either to the right (vertical) or below (horizontal). The
    figure size and spacing are automatically calculated to accommodate the legends
    without manual adjustment.

    Args:
        nrows: Number of rows of subplots
        ncols: Number of columns of subplots
        legend_positions: Dictionary mapping row index (0-based) to legend position.
                         Use 'right' for vertical legend on right side of row,
                         or 'below' for horizontal legend below row.
                         Example: {0: 'right', 2: 'below'}
                         Rows not in dict will have no legend space allocated.
        subplot_size: Base (width, height) in inches for each subplot
        legend_width: Width in inches for 'right' legends
        legend_height: Height in inches for 'below' legends
        wspace: Width spacing between subplots (fraction of subplot width)
        hspace: Height spacing between rows (fraction of subplot height)
        sharex: Share x-axis among subplots. Options:
                - False or 'none': no sharing (default)
                - True or 'all': all subplots share x-axis
                - 'row': subplots in same row share x-axis
                - 'col': subplots in same column share x-axis
        sharey: Share y-axis among subplots (same options as sharex)
        xlabel_space: Extra height (inches) to add for x-tick labels when placing
                     'below' legends. Automatically applied to rows that will show
                     x-tick labels based on sharex setting.
        **subplot_kw: Additional keyword arguments passed to add_subplot()

    Returns:
        fig: The matplotlib Figure object
        axs: 2D numpy array of subplot axes (nrows x ncols)
             For rows with legends, additional axes are created but not returned
             in the main array. Access them via fig.axes if needed.

    Examples:
        >>> # Create 3x3 grid with shared x-axis and legends
        >>> fig, axs = subplots_with_legends(
        ...     legend_positions={0: 'right', 2: 'below'},
        ...     sharex='all',  # Only bottom row will show x-tick labels
        ...     sharey='all'
        ... )
        >>>
        >>> # Plot on the main axes
        >>> for i in range(3):
        ...     axs[0, i].plot([1, 2, 3], [1, 2, 3], label=f'Line {i}')
        >>>
        >>> # Add legend (automatically accounts for x-label space on bottom row)
        >>> add_row_legend(axs[0], position='right')
    """
    import matplotlib.gridspec as gridspec

    if legend_positions is None:
        legend_positions = {}

    # Normalize sharex and sharey parameters
    sharex = 'all' if sharex is True else ('none' if sharex is False else sharex)
    sharey = 'all' if sharey is True else ('none' if sharey is False else sharey)

    # Determine which rows will display x-tick labels (need extra space for 'below' legends)
    def row_has_xlabels(row):
        """Check if a row will display x-tick labels based on sharex setting."""
        if sharex in ('none', False):
            return True  # All rows have their own labels
        elif sharex in ('all', 'col'):
            return row == nrows - 1  # Only bottom row
        elif sharex == 'row':
            return True  # Each row independent
        return False

    # Calculate figure dimensions
    subplot_width, subplot_height = subplot_size

    # Calculate total figure width
    # Base width from subplots + spacing between them
    total_plot_width = ncols * subplot_width + (ncols - 1) * wspace * subplot_width

    # Add legend width if any row has a 'right' legend
    has_right_legend = any(pos == "right" for pos in legend_positions.values())
    if has_right_legend:
        # Add space for legend plus a small gap
        total_width = total_plot_width + legend_width + wspace * subplot_width
    else:
        total_width = total_plot_width

    # Calculate total figure height
    # For each row: subplot height + potential below-legend height + spacing
    total_height = 0
    for row in range(nrows):
        total_height += subplot_height
        if legend_positions.get(row) == "below":
            # Add legend height plus extra space for x-tick labels if needed
            extra_space = xlabel_space if row_has_xlabels(row) else 0
            total_height += legend_height + extra_space + hspace * subplot_height * 0.5
        if row < nrows - 1:  # Add spacing between rows
            total_height += hspace * subplot_height

    # Create figure
    fig = plt.figure(figsize=(total_width, total_height))

    # Create GridSpec with appropriate ratios
    # We'll use a fine-grained grid and allocate cells as needed

    # For width: allocate based on whether we have right legends
    if has_right_legend:
        # Width ratios: subplot cols + legend
        width_ratios = [subplot_width] * ncols + [legend_width]
        gs = gridspec.GridSpec(
            nrows * 2,  # Double rows to handle below legends
            ncols + 1,  # Extra column for right legends
            figure=fig,
            width_ratios=width_ratios,
            wspace=wspace / (ncols + 1),
            hspace=0,  # We'll handle height spacing manually
        )
    else:
        gs = gridspec.GridSpec(
            nrows * 2,  # Double rows to handle below legends
            ncols,
            figure=fig,
            wspace=wspace / ncols,
            hspace=0,  # We'll handle height spacing manually
        )

    # Calculate height ratios for each grid row
    height_ratios = []
    for row in range(nrows):
        height_ratios.append(subplot_height)
        if legend_positions.get(row) == "below":
            height_ratios.append(legend_height)
        else:
            # Add a tiny spacer to maintain grid structure
            height_ratios.append(hspace * subplot_height if row < nrows - 1 else 0.01)

    # Update GridSpec with height ratios
    gs.set_height_ratios(height_ratios)

    # Create subplot axes array
    axs = np.empty((nrows, ncols), dtype=object)

    # Create each subplot
    for row in range(nrows):
        grid_row = row * 2  # Each logical row takes 2 grid rows

        for col in range(ncols):
            axs[row, col] = fig.add_subplot(gs[grid_row, col], **subplot_kw)

        # Create legend axes if needed
        if legend_positions.get(row) == "right":
            # Create axes in the rightmost column for this row
            legend_ax = fig.add_subplot(gs[grid_row, ncols])
            legend_ax.axis("off")  # Hide axes for legend space
            # Store reference for easy access
            setattr(axs[row, 0], f"_legend_ax_right", legend_ax)

        elif legend_positions.get(row) == "below":
            # Create axes spanning all columns in the row below
            legend_ax = fig.add_subplot(gs[grid_row + 1, :ncols])
            legend_ax.axis("off")  # Hide axes for legend space
            # Store reference for easy access
            setattr(axs[row, 0], f"_legend_ax_below", legend_ax)

    return fig, axs


def add_row_legend(
    axs_row: np.ndarray,
    position: str = "right",
    **legend_kwargs: Any,
) -> matplotlib.legend.Legend:
    """
    Add a legend to a row of subplots created with create_subplot_grid_with_legends.

    This is a convenience function that collects handles and labels from all axes
    in a row and creates a unified legend in the designated legend space.

    Args:
        axs_row: 1D array of axes from a single row (e.g., axs[0, :])
        position: Legend position - 'right' or 'below'
        **legend_kwargs: Additional keyword arguments passed to fig.legend()
                        Common options: ncol, fontsize, frameon, etc.

    Returns:
        The created Legend object

    Examples:
        >>> fig, axs = create_subplot_grid_with_legends(
        ...     legend_positions={0: 'right', 1: 'below'}
        ... )
        >>> # Plot data on row 0
        >>> for i, ax in enumerate(axs[0]):
        ...     ax.plot([1, 2, 3], [i, i+1, i+2], label=f'Line {i}')
        >>>
        >>> # Add legend to the right of row 0
        >>> add_row_legend(axs[0], position='right')
        >>>
        >>> # For row 1 with horizontal legend
        >>> for i, ax in enumerate(axs[1]):
        ...     ax.plot([1, 2, 3], [i, i+1, i+2], label=f'Series {i}')
        >>> add_row_legend(axs[1], position='below', ncol=3)
    """
    # Collect all unique handles and labels from the row
    handles_dict = {}
    labels_list = []

    for ax in axs_row:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in handles_dict:
                handles_dict[label] = handle
                labels_list.append(label)

    handles = [handles_dict[label] for label in labels_list]

    # Get the figure
    fig = axs_row[0].get_figure()

    # Determine legend location based on position
    if position == "right":
        # Check if legend axes exists
        if not hasattr(axs_row[0], "_legend_ax_right"):
            raise ValueError(
                "No right legend space allocated for this row. Did you specify"
                " legend_positions={'row': 'right'} when creating the grid?"
            )

        legend_ax = axs_row[0]._legend_ax_right

        # Default kwargs for right legends (vertical)
        default_kwargs = {
            "loc": "center left",
            "bbox_to_anchor": (0, 0.5),
            "bbox_transform": legend_ax.transAxes,
            "frameon": False,
        }
        default_kwargs.update(legend_kwargs)

        return fig.legend(handles, labels_list, **default_kwargs)

    elif position == "below":
        # Check if legend axes exists
        if not hasattr(axs_row[0], "_legend_ax_below"):
            raise ValueError(
                "No below legend space allocated for this row. Did you specify"
                " legend_positions={'row': 'below'} when creating the grid?"
            )

        legend_ax = axs_row[0]._legend_ax_below

        # Default kwargs for below legends (horizontal)
        default_kwargs = {
            "loc": "upper center",
            "bbox_to_anchor": (0.5, 1),
            "bbox_transform": legend_ax.transAxes,
            "ncol": len(labels_list),  # Horizontal layout by default
            "frameon": False,
        }
        default_kwargs.update(legend_kwargs)

        return fig.legend(handles, labels_list, **default_kwargs)

    else:
        raise ValueError(f"position must be 'right' or 'below', got '{position}'")


def gif_from_pngs(gif_path, pngs_fpaths, fps=24):
    import imageio

    with imageio.get_writer(str(gif_path), mode="I", fps=fps) as writer:
        for png_fpath in tqdm(pngs_fpaths):
            image = imageio.imread(str(png_fpath))
            writer.append_data(image)
