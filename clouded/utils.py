"""Utility functions for datetime handling, file operations, and data processing."""

import re
import datetime as dt
import numpy as np
from pathlib import Path
import sys
import importlib
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from warnings import warn
import pickle as pkl
from functools import wraps

from .types_core import PathLike, NumpyNumeric

DEFAULT_SEED = 137504983571204
NUMERICAL_DT_FORMAT = r"%Y%m%d%H%M%S"
RAMS_DT_FORMAT = r"%Y-%m-%d-%H%M%S"
HUMAN_DT_FORMAT = r"%Y-%m-%d %H:%M:%S"

ALL_CUSTOM_DT_FORMATS = [NUMERICAL_DT_FORMAT, RAMS_DT_FORMAT, HUMAN_DT_FORMAT]


class TwoWayDict:
    """
    A wrapper for nested dictionaries that allows indexing by outer keys or inner keys.

    When indexed by an outer key, returns the corresponding inner dictionary.
    When indexed by an inner key, returns a dictionary mapping outer keys to their
    corresponding inner values for that key.

    Example:
        >>> data = {
        ...     'mesh1': {'type': 'trajectory', 'opacity': 0.5},
        ...     'mesh2': {'type': 'simulation', 'opacity': 0.8},
        ...     'mesh3': {'type': 'trajectory', 'opacity': 0.3}
        ... }
        >>> accessor = NestedDictAccessor(data)
        >>> accessor['mesh1']  # Returns {'type': 'trajectory', 'opacity': 0.5}
        >>> accessor['type']   # Returns {'mesh1': 'trajectory', 'mesh2': 'simulation', 'mesh3': 'trajectory'}
        >>> accessor['opacity'] # Returns {'mesh1': 0.5, 'mesh2': 0.8, 'mesh3': 0.3}
    """

    def __init__(self, nested_dict: Dict[Any, Dict[Any, Any]]):
        """
        Initialize with a dictionary of dictionaries.

        Args:
            nested_dict: Dictionary where values are themselves dictionaries.
        """
        self._data = nested_dict

        self._inner_keys = set()
        for inner_dict in nested_dict.values():
            if isinstance(inner_dict, dict):
                self._inner_keys.update(inner_dict.keys())

    def __getitem__(self, key):
        """
        Get item by key, supporting both outer and inner key access.

        Args:
            key: Either an outer key or an inner key.

        Returns:
            If key is an outer key: the corresponding inner dictionary.
            If key is an inner key: dictionary mapping outer keys to inner values.

        Raises:
            KeyError: If key is found in neither outer nor inner keys.
        """
        # Try as outer key first
        if key in self._data:
            return self._data[key]

        # Try as inner key
        if key in self._inner_keys:
            result = {}
            for outer_key, inner_dict in self._data.items():
                if isinstance(inner_dict, dict) and key in inner_dict:
                    result[outer_key] = inner_dict[key]
            return result

        # Key not found anywhere
        raise KeyError(f"Key '{key}' not found in outer keys or inner keys")

    def __contains__(self, key):
        """Check if key exists in either outer or inner keys."""
        return key in self._data or key in self._inner_keys

    def __iter__(self):
        """Iterate over outer keys."""
        return iter(self._data)

    def keys(self):
        """Return outer keys."""
        return self._data.keys()

    def inner_keys(self):
        """Return all unique inner keys."""
        return self._inner_keys

    def items(self):
        """Return outer key-value pairs."""
        return self._data.items()

    @property
    def values(self):
        return [x for l in self._data.values() for x in l.values()]

    def __repr__(self):
        """Return HTML table representation of the TwoWayDict."""
        if not self._data or not self._inner_keys:
            return "TwoWayDict({})"

        # Get sorted keys for consistent output
        outer_keys = sorted(self._data.keys())
        inner_keys = sorted(self._inner_keys)

        # Start building HTML table
        html = [
            '<table border="1" style="border-collapse: collapse; font-family:'
            ' monospace;">'
        ]

        # Build header row
        html.append("  <thead>")
        html.append("    <tr>")
        html.append(
            '      <th style="padding: 4px 8px; background-color: #f0f0f0;"></th>'
        )  # Empty top-left cell
        for inner_key in inner_keys:
            html.append(
                '      <th style="padding: 4px 8px; background-color:'
                f' #f0f0f0;">{str(inner_key)}</th>'
            )
        html.append("    </tr>")
        html.append("  </thead>")

        # Build data rows
        html.append("  <tbody>")
        for outer_key in outer_keys:
            html.append("    <tr>")
            html.append(
                '      <th style="padding: 4px 8px; background-color:'
                f' #f8f8f8;">{str(outer_key)}</th>'
            )

            inner_dict = self._data.get(outer_key, {})
            for inner_key in inner_keys:
                if inner_key in inner_dict:
                    value = inner_dict[inner_key]
                    # Check if value is iterable (but not string) and has length
                    try:
                        if hasattr(value, "__len__") and not isinstance(value, str):
                            cell_value = str(len(value))
                        else:
                            cell_value = "✓"
                    except TypeError:
                        cell_value = "✓"
                else:
                    cell_value = ""

                html.append(
                    '      <td style="padding: 4px 8px; text-align:'
                    f' center;">{cell_value}</td>'
                )

            html.append("    </tr>")
        html.append("  </tbody>")
        html.append("</table>")

        return "\n".join(html)


def dt_to_str(dt_like: Any, date_format: str = NUMERICAL_DT_FORMAT) -> str:
    """
    Convert datetime-like objects to formatted strings.

    Args:
        dt_like: Datetime-like object (datetime, numpy.datetime64, pandas.Timestamp, string, etc.)
        date_format: strftime format string

    Returns:
        Formatted datetime string

    Raises:
        ValueError: If dt_like cannot be converted to datetime
        TypeError: If date_format is invalid
    """

    # Handle None/empty inputs
    if dt_like is None:
        raise ValueError("Cannot convert None to datetime string")

    # If it's already a string, try to parse it first
    if isinstance(dt_like, str):
        dt_like = str_to_dt(dt_like)

    # Handle native Python datetime objects (including subclasses like pandas.Timestamp)
    if hasattr(dt_like, "strftime"):
        try:
            return dt_like.strftime(date_format)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to format datetime with format '{date_format}': {e}"
            )

    # Handle numpy datetime64
    if hasattr(dt_like, "astype") and hasattr(dt_like, "dtype"):
        if np.issubdtype(dt_like.dtype, np.datetime64):
            try:
                # Convert to datetime64[s] first to avoid precision issues
                dt_as_seconds = dt_like.astype("datetime64[s]")
                # Then convert to Python datetime
                py_datetime = dt_as_seconds.astype(dt.datetime)
                return py_datetime.strftime(date_format)
            except (ValueError, TypeError, OverflowError) as e:
                raise ValueError(f"Failed to convert numpy datetime64 to string: {e}")

    # Handle time.struct_time
    if hasattr(dt_like, "tm_year"):
        try:
            py_datetime = dt.datetime(*dt_like[:6])
            return py_datetime.strftime(date_format)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to convert struct_time to string: {e}")

    # Handle timestamp (Unix epoch) - both int and float
    if isinstance(dt_like, (int, float)) and dt_like > 0:
        try:
            py_datetime = dt.datetime.fromtimestamp(dt_like)
            return py_datetime.strftime(date_format)
        except (ValueError, TypeError, OSError) as e:
            raise ValueError(f"Failed to convert timestamp {dt_like} to string: {e}")

    # Try pandas Timestamp if pandas is available
    try:
        import pandas as pd

        if isinstance(dt_like, pd.Timestamp):
            return dt_like.strftime(date_format)
    except ImportError:
        pass

    # Last resort: try to convert to datetime using str_to_dt
    try:
        parsed_dt = str_to_dt(str(dt_like))
        return parsed_dt.strftime(date_format)
    except (ValueError, TypeError):
        pass

    raise ValueError(
        f"Cannot convert object of type {type(dt_like)} to datetime string: {dt_like}"
    )


def str_to_dt(
    s: str,
    date_format: Optional[str] = None,
    try_digits_only=True,
    raise_if_failure: bool = True,
) -> Optional[dt.datetime]:
    """
    Coerce datetime-like strings to native datetime objects.

    Args:
        s: String to parse
        date_format: Specific format to try, or None to try all formats
        raise_if_failure: Whether to raise exception on parsing failure

    Returns:
        datetime object or None (if raise_if_failure=False)
    """

    def _parse_str(s):
        if not date_format:
            possible_calls = [
                lambda s: dt.datetime.fromisoformat(s),
            ] + [
                lambda s: dt.datetime.strptime(s, fmt) for fmt in ALL_CUSTOM_DT_FORMATS
            ]

            # Add pandas parsing if available
            try:
                import pandas as pd

                possible_calls.append(lambda s: pd.Timestamp(s).to_pydatetime())
            except ImportError:
                pass

            # Add dateutil parsing if available
            try:
                from dateutil.parser import parse as dateutil_parse

                possible_calls.append(lambda s: dateutil_parse(s))
            except ImportError:
                pass
        else:
            possible_calls = [lambda s: dt.datetime.strptime(s, date_format)]

        for possible_call in possible_calls:
            try:
                return possible_call(s)
            except (ValueError, TypeError, AttributeError):
                continue
        return None

    this_dt = _parse_str(s)
    # Also try just stripping everything that's not a digit
    if not this_dt and try_digits_only:
        this_dt = _parse_str("".join([c for c in s if c.isdigit()]))

    if not this_dt and raise_if_failure:
        raise ValueError(f"Could not coerce string '{s}' to datetime")
    return this_dt


class RaiseIfExistsException(Exception):
    # Type of error specifically for trying to write if something exists
    # Having this makes it easier to catch only this exception
    pass


def to_kv_pairs(
    s: Union[str, Path], parse_datetimes: bool = False, parse_floats: bool = False
) -> Dict[str, Any]:
    """
    Parse key-value pairs from a string or Path stem.

    Args:
        s: String or Path to parse (if Path, uses stem)
        parse_datetimes: Whether to attempt parsing values as datetime objects
        parse_floats: Whether to attempt parsing values as floats

    Returns:
        Dictionary of key-value pairs
    """
    # Handle some convenient cases
    if isinstance(s, Path):
        s = str(s.stem)
    else:
        s = str(s)
    d = {}
    for kv_pair in s.split("_"):
        splits = kv_pair.split("-")
        # Handle the case in which there's more than one - in the name, e.g.
        # if the value is a negative number
        k = splits[0]
        v = "-".join(splits[1:])
        if parse_datetimes:
            try:
                # Try to parse as datetime first
                v = str_to_dt(v)
            except:
                pass
        if parse_floats and isinstance(v, str):
            try:
                v = float(v)
            except:
                pass
        d[k] = v
    return d


def to_kv_str(d: Dict[str, Any]) -> str:
    """
    Convert a dictionary to a key-value filename string.

    Args:
        d: Dictionary to convert

    Returns:
        String in format "key1-value1_key2-value2_..."
    """
    sanitize = lambda x: str(x).replace("_", "")
    return "_".join([f"{sanitize(k)}-{sanitize(v)}" for k, v in d.items()])


def key_in_selector(
    key: Dict[str, Any], selector: Dict[str, Union[str, List[str]]]
) -> bool:
    """
    Check if a key matches the selector criteria.

    Args:
        key: Dictionary to check
        selector: Selection criteria

    Returns:
        True if key matches all selector criteria
    """
    key = dict(key)
    selector = dict(selector)
    for k, v in selector.items():
        if isinstance(v, str):
            v = [v]
        if key.get(k) not in v:
            return False
    return True


def filter_paths_by_selector(
    paths: List[PathLike], selector: Dict[str, List[Any]], parse_floats: bool = True
) -> List[PathLike]:
    """
    Filter a list of paths based on key-value selector criteria.

    Args:
        paths: List of file paths to filter
        selector: Dictionary of selection criteria
        parse_floats: Whether to parse numeric values as floats

    Returns:
        Filtered list of paths
    """
    filtered = []
    for this_path in paths:
        # Pull out all of the keys from the directory name
        this_path_kv_pairs = to_kv_pairs(
            Path(this_path).stem, parse_floats=parse_floats
        )
        for selector_key, selector_values in selector.items():
            if this_path_kv_pairs.get(selector_key) not in selector_values:
                continue
        filtered.append(this_path)
    return filtered


def current_dt_str(format: str = NUMERICAL_DT_FORMAT) -> str:
    """
    Get current datetime as formatted string.

    Args:
        format: strftime format string

    Returns:
        Current datetime as formatted string
    """
    return dt.datetime.now().strftime(format)


def filter_to_points(
    da: Any, as_dicts: bool = True
) -> Union[List[Dict[str, Any]], np.ndarray]:
    """
    Filter a DataArray to extract points where values are True.

    Args:
        da: xarray DataArray to filter
        as_dicts: If True, return list of dictionaries; if False, return numpy array

    Returns:
        Filtered points as either list of dicts or numpy array
    """
    # Assume da is a DataArray of the boolean we want to filter by
    # assert set(da.dims) == set(['x', 'y', 'z'])
    s = da.to_series()
    filtered_points = s.loc[s.astype(bool)]
    # Dumb that this is the only way I can figure out to convert the array of tuples to a 2D array
    if as_dicts:
        return [
            {dim_name: l[dim_ix] for dim_ix, dim_name in enumerate(da.dims)}
            for l in filtered_points.index
        ]
    else:
        return np.array([np.array(l) for l in filtered_points.index])


def raise_if_exists(fpath: PathLike) -> None:
    """
    Raise OSError if the given file path exists.

    Args:
        fpath: File path to check

    Raises:
        OSError: If the file exists
    """
    if Path(fpath).exists():
        raise OSError(f"Output path {str(fpath)} exists and exist_ok=False was passed")


def maybe_random_choice(
    arr: np.ndarray, size: int, seed: int = DEFAULT_SEED
) -> np.ndarray:
    """
    Return at most `size` elements from array, handling case where array is smaller than size.

    Args:
        arr: Array to sample from
        size: Maximum number of elements to return
        seed: Random seed for reproducibility

    Returns:
        Array with at most `size` elements
    """
    # Return at most `size` elements from arr; just handles the case where `arr`
    # is smaller than `size`
    if size >= len(arr):
        return arr
    else:
        return np.random.default_rng(seed=seed).choice(arr, size=size, replace=False)


def prepend_to_stem(to_prepend: str, fpath: PathLike) -> Path:
    """
    Prepend text to the stem of a file path.

    Args:
        to_prepend: Text to prepend
        fpath: File path to modify

    Returns:
        Path with modified stem
    """
    fpath = Path(fpath)
    return fpath.with_stem(to_prepend + fpath.stem)


def append_to_stem(fpath: PathLike, to_append: str) -> Path:
    """
    Append text to the stem of a file path.

    Args:
        fpath: File path to modify
        to_append: Text to append

    Returns:
        Path with modified stem
    """
    # Opposite argument order of prepend_to_stem, probably bad design but I like it
    # this way
    fpath = Path(fpath)
    return fpath.with_stem(fpath.stem + to_append)


def fps(simulation_minutes_per_second: float, simulation_time_per_frame: Any) -> float:
    """
    Calculate frames per second from simulation parameters.

    Args:
        simulation_minutes_per_second: Simulation time rate
        simulation_time_per_frame: Time duration per frame

    Returns:
        Frames per second
    """
    simulation_seconds_per_frame = simulation_time_per_frame.nanos / 1e9
    return (simulation_minutes_per_second * 60) / simulation_seconds_per_frame


def is_evenly_spaced(arr: np.ndarray, exact: bool = True) -> bool:
    """
    Check if array elements are evenly spaced.

    Args:
        arr: Array to check
        exact: If True, require exact spacing; if False, use approximate comparison

    Returns:
        True if elements are evenly spaced
    """
    if len(arr) > 1:
        diffs = np.diff(arr)
        are_evenly_spaced = (
            all(diffs == diffs[0]) if exact else np.allclose(diffs, diffs[0])
        )
        return are_evenly_spaced
    else:
        return True


def raise_if_not_evenly_spaced_(arr: np.ndarray, exact: bool = True) -> None:
    """
    Raise ValueError if array elements are not evenly spaced.

    Args:
        arr: Array to check
        exact: If True, require exact spacing; if False, use approximate comparison

    Raises:
        ValueError: If array is not evenly spaced
    """
    if not is_evenly_spaced(arr, exact=exact):
        raise ValueError(f"Array is not evenly spaced")


def spacing(
    arr: np.ndarray, raise_if_not_evenly_spaced: bool = True, exact: bool = True
) -> NumpyNumeric:
    """
    Calculate spacing between array elements.

    Args:
        arr: Array to calculate spacing for
        raise_if_not_evenly_spaced: Whether to raise error if not evenly spaced
        exact: If True, require exact spacing; if False, use approximate comparison

    Returns:
        Spacing between elements as numpy integer or float

    Raises:
        ValueError: If array is not evenly spaced and raise_if_not_evenly_spaced is True
    """
    if raise_if_not_evenly_spaced:
        raise_if_not_evenly_spaced_(arr, exact=exact)
    else:
        warn(
            "`spacing` called with raise_if_not_evenly_spaced=False; be sure to "
            "check this manually"
        )

    return arr[1] - arr[0]


def maybe_cast_to_float(arr: np.ndarray) -> np.ndarray:
    """
    Attempt to cast array to float, returning original array if cast fails.

    Args:
        arr: Array to cast

    Returns:
        Float array if successful, original array otherwise
    """
    try:
        return arr.astype(float)
    except ValueError:
        return arr


def recursive_reload(module: Any, silent=False) -> None:
    """Recursively reload a module and all its submodules"""
    # Get all submodules that start with the module's name
    module_name = module.__name__
    submodules_to_reload = []

    for name, mod in sys.modules.items():
        if name.startswith(module_name + ".") and mod is not None:
            submodules_to_reload.append((name, mod))

    # Sort by depth (deeper modules first) to avoid dependency issues
    submodules_to_reload.sort(key=lambda x: x[0].count("."), reverse=True)

    # Reload submodules first
    for name, mod in submodules_to_reload:
        try:
            importlib.reload(mod)
        except Exception as e:
            if not silent:
                print(f"Failed to reload {name}: {e}")

    # Finally reload the main module
    importlib.reload(module)
    if not silent:
        print(f"Reloaded {module_name}")


def empty_directory(dir_path: PathLike, delete_directory: bool = False):
    import shutil

    dir_path = Path(dir_path)
    assert dir_path.is_dir()
    if dir_path.exists():
        shutil.rmtree(dir_path)
    if not delete_directory:
        dir_path.mkdir(parents=False, exist_ok=False)


def read_file(filepath, *args, **kwargs):
    """Read a file using the appropriate library based on its extension.

    Args:
        filepath: Path to the file to read

    Returns:
        File contents loaded with the appropriate library

    Supported formats:
        - .nc: xarray.open_dataset
        - .npy: numpy.load
        - .csv: pandas.read_csv
        - .pkl: pickle
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext == ".nc":
        import xarray as xr

        return xr.open_dataset(path, *args, **kwargs)
    elif ext == ".npy":
        return np.load(path, *args, **kwargs)
    elif ext == ".csv":
        import pandas as pd

        return pd.read_csv(path, *args, **kwargs)
    elif ext == ".pkl":
        with path.open("rb") as f:
            return pkl.load(f, *args, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def read_or_cache_to(filepath):
    """Decorator that caches function results to a pickle file.

    If the filepath exists, loads and returns the cached result.
    Otherwise, calls the function, saves the result to the filepath, and returns it.

    Args:
        filepath: Path to the pickle file for caching
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            path = Path(filepath)
            if path.exists():
                result = read_file(filepath)
                print(f"Read result from {str(filepath)}")
                return result
            else:
                print(f"No file at {str(filepath)}, computing")
                result = func(*args, **kwargs)
                # These computations are generally expensive so be sure to return
                # the result even if the filepath to cache to is bad
                try:
                    with open(path, "wb") as f:
                        pkl.dump(result, f)
                    print(f"Wrote computation result to {str(filepath)}")
                except:
                    print(
                        f"Failed to write to {str(filepath)}; returning result without"
                        " caching"
                    )

                return result

        return wrapper

    return decorator


def is_arraylike(maybe_arr):
    return hasattr(maybe_arr, "__iter__") and not isinstance(maybe_arr, str)


def list_if_single(maybe_single):
    return (
        maybe_single
        if is_arraylike(maybe_single)
        else [
            maybe_single,
        ]
    )
