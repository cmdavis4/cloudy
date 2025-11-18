"""RAMS (Regional Atmospheric Modeling System) utilities.

This module provides functions for working with RAMS atmospheric model output,
including file parsing, data processing, unit handling, and visualization.
"""

import re
from pathlib import Path
import datetime as dt
from typing import Any, Dict, List, Optional, Union, Tuple, Callable

import xarray as xr
from tqdm.notebook import tqdm
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pint_xarray
from pint import UnitRegistry
import pandas as pd
import metpy.calc as mpc
from metpy.units import units
import metpy.constants as mpconstants

from .utils import current_dt_str, RAMS_DT_FORMAT, str_to_dt
from .types_core import PathLike, ConfigDict

# Alias for backwards compatibility
RAMS_DT_STRFTIME_STR = RAMS_DT_FORMAT

# Load our custom units
from pint import UnitRegistry

# First we create the registry.
ureg = UnitRegistry()
# Then we append the new definitions
ureg.load_definitions(
    str(Path(__file__).parent.joinpath("data", "rams_pint_units.txt"))
)
pint_xarray.setup_registry(ureg)

# Define a regex to pull out the datetime from RAMS filenames
RAMS_FILENAME_DATETIME_REGEX = r"[0-9]{4}\-[0-9]{2}\-[0-9]{2}\-[0-9]{6}"

# Mapping from the phony dim number assigned to each dimension to what it actually is. This appears to be programmatic,
# at least for full output files, but is not for lite files.
RAMS_ANALYSIS_FILE_DIMENSIONS_DICT = {
    "phony_dim_0": "y",
    "phony_dim_1": "x",
    "phony_dim_2": "z",
    "phony_dim_3": "p",
    "phony_dim_4": "kppz",
    "phony_dim_5": "zs",
    "phony_dim_6": "zg",
}

# Default variables to calculate base state-relative versions of
DEFAULT_BSR_VARIABLES = ["THETA", "UC", "VC", "THETA_v", "THETA_rho", "P"]

# Templates for the commands that are issued to the subprocesses that actually run rams
RAMS_SERIAL_COMMAND_TEMPLATE = "{rams_executable_path} -f {ramsin_path}"
RAMS_MPIEXEC_COMMAND_TEMPLATE = (
    "/home/C837213679/software/mpich-3.3.2_new/bin/mpiexec -machinefile"
    " {machsfile_path} -np {n_cores} {rams_executable_path} -f {ramsin_path}"
)

# Mapping from the names of dimensions as they are given in header files to the conventional names
HEADER_NAME_DIMENSION_DICT = {
    "__ztn{grid_number}": "z",
    "__ytn{grid_number}": "y",
    "__xtn{grid_number}": "x",
}

# List of variables that are considered part of the initial sounding for a RAMS run
SOUNDING_NAMELIST_VARIABLES = ["PS", "TS", "RTS", "US", "VS"]


# Read in RAMS output variable names, units, dimensions, descriptions
RAMS_VARIABLES_DF = pd.read_csv(
    Path(__file__).parent.joinpath("data", "rams_variables.csv")
)

# Need to do a little cleaning
RAMS_VARIABLES_DF["units"] = RAMS_VARIABLES_DF["units"].str.replace("#", "1")
RAMS_VARIABLES_DF = RAMS_VARIABLES_DF.drop(RAMS_VARIABLES_DF.columns[0], axis=1)

# Human-readable names for each hydrometeor species
HYDROMETEOR_SPECIES_FULL_NAMES = {
    "PP": "pristine ice",
    "SP": "snow",
    "AP": "aggregates",
    "HP": "hail",
    "GP": "graupel",
    "CP": "cloud",
    "DP": "drizzle",
    "RP": "rain",
}


def generate_ramsin(
    ramsin_name: str,
    parameters: Dict[str, str],
    rams_input_dir: Optional[PathLike],
    rams_output_dir: Optional[PathLike],
    ramsin_dir: PathLike,
    ramsin_template_path: PathLike,
) -> str:
    """Generates a RAMSIN file, given a template RAMSIN and a set of parameters to change relative to the template.

    Args:
        ramsin_name (str): Name of this RAMSIN; this will be used in the filename, which will be `RAMSIN.{ramsin_name}`
        parameters (dict): Dict of parameters to change relative to those contained in the template RAMSIN. Keys should
            be strs corresponding to the names of the parameters, and values should be strs, to avoid any ambiguity
            with how they are written. Values will be written exactly as given, meaning that any quotes must be included
            within the str.
        rams_input_dir (str or pathlib.Path): Directory from which to read input files in the RAMS run; this sets the
            prefix of the TOPFILES, SFCFILES, SSTFPFX, and NDVIFPFX parameters in the RAMSIN. This behavior can be
            overridden on a per-parameter basis by passing any of the four aforementioned parameters explicitly in
            `parameters`.
        rams_output_dir (str or pathlib.Path): Directory to which to write the output of the RAMS run; this sets the
            prefix of the AFILEPREF parameter in the RAMSIN. This behavior can be overridden by passing AFILEPREF
            explicitly in `parameters`.
        ramsin_dir (str or pathlib.Path): Directory to which the RAMSIN will be written.
        ramsin_template_path (str or pathlib.Path): Path to a template RAMSIN file. The generated RAMSIN will be exactly
            the contents of the template RAMSIN, with only the values of the parameters given in `parameters` changed.

    Returns:
        str: Text of the generated RAMSIN
    """
    # Make a copy of parameters since we're going to modify it
    parameters = {k: v for k, v in parameters.items()}

    # First make the 4 paths into pathlib.Paths if they're not already
    rams_input_dir = (
        Path(rams_input_dir) if rams_input_dir is not None else rams_input_dir
    )
    rams_output_dir = (
        Path(rams_output_dir) if rams_output_dir is not None else rams_output_dir
    )
    ramsin_dir = Path(ramsin_dir)
    ramsin_template_path = Path(ramsin_template_path)

    # First replace the IO paths
    with ramsin_template_path.open("r") as f:
        ramsin = f.read()

    # Add the input and output directories to the parameter dict so that we can replace them like
    # normal parameters
    input_dir_sub_suffixes = {
        "TOPFILES": "toph",
        "SFCFILES": "sfch",
        "SSTFPFX": "ssth",
        "NDVIFPFX": "ndh",
    }
    output_dir_sub_suffixes = {"AFILEPREF": "a"}

    # Replace these in the RAMSIN, if they're not explicitly being passed in the parameters
    for param_name, suffix in input_dir_sub_suffixes.items():
        if param_name not in parameters and rams_input_dir is not None:
            parameters[param_name] = f"'{str(rams_input_dir.joinpath(suffix))}'"
    for param_name, suffix in output_dir_sub_suffixes.items():
        if param_name not in parameters and rams_output_dir is not None:
            parameters[param_name] = f"'{str(rams_output_dir.joinpath(suffix))}'"

    # Now replace the parameters given in the RAMSIN with the rendered paths
    # This isn't super efficient but it doesn't matter for a text file of this size
    for parameter_name, parameter_value in parameters.items():
        parameter_regex = r"(^\s*{}\s*\=\s*).*?(\n[^\n\!]*[\=\$])".format(
            parameter_name
        )
        replacement_regex = r"\g<1>{},\g<2>".format(parameter_value)
        ramsin, n_subs = re.subn(
            parameter_regex,
            replacement_regex,
            ramsin,
            count=1,
            flags=re.MULTILINE | re.DOTALL,
        )
        if n_subs == 0:
            raise ValueError(
                "Field {} not found in template RAMSIN".format(parameter_name)
            )

    # Write the rendered RAMSIN to disk
    with ramsin_dir.joinpath("RAMSIN.{}".format(ramsin_name)).open("w") as f:
        f.write(ramsin)

    # Return the rendered RAMSIN in case we wanna look at it
    return ramsin


def run_rams_for_ramsin(
    ramsin_path: PathLike,
    stdout_path: PathLike,
    rams_executable_path: PathLike,
    machsfile_path: Optional[PathLike] = None,
    log_command: bool = True,
    log_ramsin: bool = True,
    dry_run: bool = False,
    asynchronous: bool = True,
    verbose: bool = True,
) -> Union[bool, subprocess.Popen]:
    """Run RAMS for a specific RAMSIN configuration file.

    Args:
        ramsin_path: Path to the RAMSIN configuration file
        stdout_path: Path where stdout will be written
        rams_executable_path: Path to the RAMS executable
        machsfile_path: Path to machine file for parallel execution
        log_command: Whether to log the command being executed
        log_ramsin: Whether to log the RAMSIN contents
        dry_run: If True, don't actually execute, just return True
        asynchronous: If True, run asynchronously and return Popen object
        verbose: Whether to print verbose output

    Returns:
        Either True (for dry runs), or subprocess.Popen object for async runs,
        or subprocess.CompletedProcess for synchronous runs

    Raises:
        ValueError: If ramsin path is longer than 256 characters
    """
    # First check if the ramsin path is more than 256 characters, which RAMS can't handle
    if len(str(Path(ramsin_path).resolve())) > 256:
        raise ValueError("RAMS cannot handle ramsin paths longer than 256 characters")
    # Convert the rams executable path to an absolute path
    rams_executable_path = str(Path(rams_executable_path).resolve())
    if not machsfile_path:
        # Running serially
        command = RAMS_SERIAL_COMMAND_TEMPLATE.format(
            rams_executable_path=rams_executable_path,
            ramsin_path=str(Path(ramsin_path).resolve()),
        )
    else:
        # Running in parallel, so need mpiexec
        # Total up the number of cores from the nodelist
        with Path(machsfile_path).open("r") as f:
            nodelist = f.readlines()
        n_cores = sum([int(s.split(":")[1]) for s in nodelist])
        # Write the nodelist to a machs file

        nodes_str = ",".join(nodelist)
        command = RAMS_MPIEXEC_COMMAND_TEMPLATE.format(
            machsfile_path=str(Path(machsfile_path).resolve()),
            n_cores=n_cores,
            rams_executable_path=rams_executable_path,
            ramsin_path=str(Path(ramsin_path).resolve()),
        )

    write_mode = "w"

    if log_command:
        import hashlib

        with Path(rams_executable_path).open("rb") as rams_exe_f:
            rams_checksum = hashlib.md5(rams_exe_f.read()).hexdigest()
        with Path(stdout_path).open(write_mode) as stdout_f:
            hashes = "#" * 47
            stdout_f.write(f"{hashes}\nRAMS CHECKSUM: {rams_checksum}\n{hashes}\n\n")
            stdout_f.write(
                "##############################\n         BEGIN"
                " COMMAND\n##############################\n"
            )
            stdout_f.write(f"{command} > {stdout_path}")
            stdout_f.write(
                "\n##############################\n         END"
                " COMMAND\n##############################\n\n"
            )
        write_mode = "a"

    if log_ramsin:
        # If we're logging the whole ramsin, we need to open the file in write mode first and write the ramsin,
        # then open it in append mode and write the stdout from rams that way
        with Path(stdout_path).open(write_mode) as stdout_f:
            with Path(ramsin_path).open("r") as ramsin_f:
                stdout_f.write(
                    "##############################\n         BEGIN"
                    " RAMSIN\n##############################\n"
                )
                stdout_f.write(ramsin_f.read())
                stdout_f.write(
                    "\n##############################\n         END"
                    " RAMSIN\n##############################\n\n"
                )
        write_mode = "a"
    # Print the command we'll run, plus a pipe indicating where the stdout will be written (in reality this is
    # handled by the `stdout` argument to `subprocess.run` below, but this is convenient)
    if verbose:
        print(f"{command} > {stdout_path}")
    if dry_run:
        completed = True
    else:
        with Path(stdout_path).open(write_mode) as stdout_f:
            if asynchronous:
                completed = subprocess.Popen(
                    command.split(" "), stdout=stdout_f, start_new_session=True
                )

            else:
                completed = subprocess.run(
                    command.split(" "),
                    stdout=stdout_f,
                )

    return completed


def run_rams(
    parameter_sets_dict: Dict[str, Dict[str, str]],
    run_dir: PathLike,
    rams_executable_path: Union[PathLike, Dict[str, PathLike]],
    ramsin_template_path: PathLike,
    nodelist: Optional[Union[List[str], Dict[str, List[str]]]] = None,
    log_command: bool = True,
    log_ramsin: bool = True,
    dry_run: bool = False,
    parallel: bool = True,
    block: bool = True,
    date_filenames: bool = False,
    verbose: bool = True,
) -> List[Union[bool, subprocess.Popen]]:
    """Run RAMS simulations for multiple parameter sets.

    Args:
        parameter_sets_dict: Dictionary mapping parameter set names to parameter dictionaries
        run_dir: Base directory for all simulation runs
        rams_executable_path: Path to RAMS executable, or dict mapping parameter sets to paths
        ramsin_template_path: Path to template RAMSIN file
        nodelist: Node configuration for parallel runs
        log_command: Whether to log commands
        log_ramsin: Whether to log RAMSIN contents
        dry_run: If True, don't actually execute simulations
        parallel: Whether to run simulations in parallel
        block: If True, wait for all processes to complete
        date_filenames: Whether to append timestamps to filenames
        verbose: Whether to print verbose output

    Returns:
        List of process objects or completion status for each parameter set

    Raises:
        ValueError: If nodelist keys don't match parameter_sets_dict keys
    """
    # Make the run_dir a Path if it's not already
    run_dir = Path(run_dir)

    # Make directories for the IO and ramsins
    # General file structure this creates will be:
    # - {run_name}/
    #   - {run_name}_machsfile.master (if one nodelist passed)
    #   - {parameter_set_name}/
    #     - input
    #     - output
    #     - RAMSIN.{parameter_set_name}
    #     - {parameter_set_name}.stdout
    run_dir.mkdir(parents=False, exist_ok=True)

    # Create the date suffix if we're using it
    fname_suffix = ("_dt-" + current_dt_str()) if date_filenames else ""
    # Create the file structure for each parameter set
    parameter_set_dirs = {}
    for parameter_set_name, parameters in parameter_sets_dict.items():
        this_parameter_set_dir = run_dir.joinpath(parameter_set_name + fname_suffix)
        this_input_dir = this_parameter_set_dir.joinpath("input")
        this_output_dir = this_parameter_set_dir.joinpath("output")
        this_parameter_set_dir.mkdir(parents=False, exist_ok=True)
        this_input_dir.mkdir(parents=False, exist_ok=True)
        this_output_dir.mkdir(parents=False, exist_ok=True)
        parameter_set_dirs[parameter_set_name] = {"top": this_parameter_set_dir}

        generate_ramsin(
            parameter_set_name + fname_suffix,
            parameters,
            rams_input_dir=str(this_input_dir),
            rams_output_dir=str(this_output_dir),
            ramsin_dir=str(this_parameter_set_dir),
            ramsin_template_path=ramsin_template_path,
        )

    # Create machsfile(s) if we were passed a nodelist/nodelists
    if nodelist:
        # nodelist can be a single list/array, or a dictionary of list/arrays with one for each parameter_set
        if isinstance(nodelist, dict):
            # Check that the keys in the nodelist dict are the same as those of the parameter sets dict
            if not nodelist.keys() == parameter_sets_dict.keys():
                raise ValueError(
                    "If nodelist is a dict, its keys must match those of"
                    " parameter_sets_dict exactly"
                )
        else:
            # If it's not actually a dict, make one with the same value for all parameter sets
            nodelist = {
                parameter_set_name: nodelist
                for parameter_set_name in parameter_set_dirs.keys()
            }
        for parameter_set_name, parameter_set_dir in parameter_set_dirs.items():
            # Figure out the machsfile path for this parameter set
            this_machsfile_path = parameter_set_dir["top"].joinpath(
                "{}_machsfile.master".format(parameter_set_name)
            )
            with this_machsfile_path.open("w") as f:
                # Write the nodelist for this parameter set to it
                f.write("\n".join(nodelist[parameter_set_name]))
            # Save the path in parameter_set_dirs
            parameter_set_dir["machsfile"] = this_machsfile_path
    else:
        for parameter_set_dir in parameter_set_dirs.values():
            parameter_set_dir.update({"machsfile": None})

    # Now run rams
    run_results = []
    for parameter_set_name in parameter_sets_dict.keys():
        this_parameter_set_dirs = parameter_set_dirs[parameter_set_name]
        run_results.append(
            run_rams_for_ramsin(
                ramsin_path=str(
                    this_parameter_set_dirs["top"].joinpath(
                        "RAMSIN.{}".format(parameter_set_name + fname_suffix)
                    )
                ),
                stdout_path=str(
                    this_parameter_set_dirs["top"].joinpath(
                        "{}.stdout".format(parameter_set_name + fname_suffix)
                    )
                ),
                rams_executable_path=(
                    rams_executable_path[parameter_set_name]
                    if isinstance(rams_executable_path, dict)
                    else rams_executable_path
                ),
                machsfile_path=this_parameter_set_dirs["machsfile"],
                log_command=log_command,
                log_ramsin=log_ramsin,
                dry_run=dry_run,
                asynchronous=parallel,
                verbose=verbose,
            )
        )
    if parallel and block and not dry_run:
        # We want to block until all of the subprocesses we spawned are finished
        # Call the wait method of each subprocess; it will return if the process is finished
        try:
            [sp.wait() for sp in run_results]
        finally:
            # If we interrupt this it won't kill the processes, so we implement that manually
            [sp.kill() for sp in run_results]
    return run_results


def get_rams_dimension_values(
    header_filepath: PathLike,
    grid_number: int = 1,
) -> Dict[str, List[float]]:
    # Read in the actual values of the coordinates in this dataset from the header file
    # Just use the first filepath given; they should all be the same for the same RAMS run
    dimension_vals = {}
    with Path(header_filepath).open("r") as f:
        # Make a copy of the header name dimension dict since we'll be popping from it
        header_name_dimension_dict = {
            k.format(**{"grid_number": str(grid_number).zfill(2)}): v
            for k, v in HEADER_NAME_DIMENSION_DICT.items()
        }
        while header_name_dimension_dict:
            for line in f:
                line = line.strip()
                if line in header_name_dimension_dict.keys():
                    this_header_name = line
                    break
            n_levels = int(
                next(f).strip()
            )  # The line after the fieldname lists the number of levels
            levels = [
                float(next(f).strip()) for _ in range(n_levels)
            ]  # Get the levels themselves
            dimension_vals[header_name_dimension_dict[this_header_name]] = levels
            header_name_dimension_dict.pop(this_header_name)
    return dimension_vals


def infer_rams_dimensions(
    single_time_rams_ds: xr.Dataset,
    grid_number: int = 1,
) -> Tuple[Dict[str, str], Dict[str, List[float]]]:
    # Convert the header filepath to a data filepath
    header_filepath = to_header_filepath(
        rams_output_filepath=single_time_rams_ds.encoding["source"]
    )
    # Get the dimension values from the header file
    dimension_vals = get_rams_dimension_values(
        header_filepath=header_filepath, grid_number=grid_number
    )

    # Now we've got the sizes and levels of the grid dimensions
    # If two of these are the same, we can't figure out the shape of the data just from
    # this, so require dimensions to be passed manually
    # Check if any dimension has the same length as another
    header_dimension_lengths = {k: len(v) for k, v in dimension_vals.items()}
    if len(header_dimension_lengths.values()) != len(
        set(header_dimension_lengths.values())
    ):
        raise ValueError(
            f"Cannot determine dimension mapping when dimensions have identical"
            f" lengths."
        )

    # Get the lengths of the dimensions in the dataset
    ds_dimension_lengths = {
        dim: len(single_time_rams_ds[dim]) for dim in single_time_rams_ds.dims
    }
    dim_names_mapping = {}
    for header_dim_name, header_dim_length in header_dimension_lengths.items():
        ds_dims_matching_length = [
            ds_dim
            for ds_dim, ds_dim_length in ds_dimension_lengths.items()
            if ds_dim_length == header_dim_length
        ]
        if len(ds_dims_matching_length) > 1:
            raise ValueError(
                "Multiple dimensions of same length in dataset; cannot infer"
                " dimension names and values"
            )
        elif len(ds_dims_matching_length) < 1:
            raise ValueError(
                f"No dimensions of length {header_dim_length} found in dataset;"
                " this shouldn't happen"
            )
        else:
            dim_names_mapping[ds_dims_matching_length[0]] = header_dim_name
    # All dimensions should now be mapped
    assert len(dim_names_mapping) == len(HEADER_NAME_DIMENSION_DICT)

    return dim_names_mapping, dimension_vals


def fill_rams_output_dimensions(
    ds: xr.Dataset,
    dimension_names_mapping: Dict[str, str],
    dimension_values: Dict[str, List[float]],
) -> xr.Dataset:
    """Fill RAMS output dimensions with proper coordinate values from header file.

    Args:
        ds: xarray Dataset with RAMS output data
        header_filepath: Path to RAMS header file containing dimension information
        dim_names: Dictionary mapping dimension names to standard names
        grid_number: Grid number for multi-grid simulations

    Returns:
        Dataset with proper dimension coordinates assigned

    Raises:
        ValueError: If dimension lengths don't match between dataset and header
    """
    try:
        ds = ds.rename_dims(dimension_names_mapping).assign_coords(dimension_values)

    except ValueError:
        print(
            "Mismatch between dimension lengths in dataset and header;\nPassed"
            f" dimension dict: {dimension_names_mapping}\nDimension sizes in dataset:"
            f" {ds.dims}\nDimension lengths from header:"
            f" { {k: len(v) for k, v in dimension_values.items()} }"
        )
        raise
    return ds


def to_header_filepath(rams_output_filepath: PathLike):
    rams_output_filepath = Path(rams_output_filepath)
    return rams_output_filepath.with_name(
        re.sub(r"g[1-9]+.h5", r"head.txt", rams_output_filepath.name)
    )


def get_grid_number(rams_output_filepath: PathLike):
    match = re.search(r"g([1-9]+).h5", Path(rams_output_filepath).name)
    if not match:
        raise ValueError(
            f"Unable to parse grid number from filepath {rams_output_filepath}"
        )
    return int(match.group(1))


def get_datetime(
    rams_output_filepath: PathLike, filename_datetime_regex=RAMS_FILENAME_DATETIME_REGEX
):
    rams_output_filepath = Path(rams_output_filepath)
    match = re.search(filename_datetime_regex, rams_output_filepath.name)
    if not match:
        raise ValueError(
            f"Unable to parse datetime from filepath {rams_output_filepath}"
        )
    return str_to_dt(match.group(0))


def read_rams_output(
    input_filenames: List[PathLike],
    fill_dim_names: bool = True,
    dim_names: Optional[Dict[str, str]] = None,
    keep_unknown_dims: bool = False,
    drop_vars: List[str] = [],
    keep_vars: List[str] = [],
    preprocess: Optional[Callable] = None,
    time_dim_name: str = "time",
    parallel: bool = True,
    chunks: Union[str, Dict[str, int]] = "auto",
    concatenate: bool = True,
    silent: bool = False,
    open_dataset_kwargs: Dict[str, Any] = {},
    filename_datetime_regex: str = RAMS_FILENAME_DATETIME_REGEX,
    units: bool = False,
) -> Union[xr.Dataset, List[xr.Dataset]]:
    """Read RAMS output files and return as xarray Dataset.

    Args:
        input_filenames: List of RAMS output file paths
        dim_names: Dictionary mapping phony dimension names to real names
        keep_vars: List of variables to keep (empty means keep all)
        preprocess: Optional preprocessing function applied to each file
        time_dim_name: Name for the time dimension
        parallel: Whether to use parallel reading with dask
        chunks: Chunking strategy for dask arrays
        concatenate: Whether to concatenate multiple files
        silent: Whether to suppress progress output
        open_dataset_kwargs: Additional arguments for xarray.open_dataset
        units: Whether to assign units using pint

    Returns:
        Dataset containing RAMS output data, or list of datasets if concatenate=False

    Note:
        Parallel reading is significantly faster and uses less memory than serial reading.
        If dask issues occur, try upgrading xarray and dask.
    """

    # Fail if both keep_vars and drop_vars are passed
    if drop_vars and keep_vars:
        raise ValueError("Cannot pass both drop_vars and keep_vars")

    # Define function for printing if we're not running silently
    def maybe_print(x):
        if not silent:
            print(x)

    # If trying to read in parallel, first check if dask is installed
    if parallel:
        try:
            import dask
        except ImportError:
            print(
                "dask must be installed to use the `parallel` option; falling back to"
                " reading serially"
            )
            parallel = False

    # Convert input filenames to paths if they're not already
    input_filenames = [Path(x) for x in input_filenames]
    # Pull the times out from these
    input_datetimes = []
    for fpath in input_filenames:
        fpath = Path(fpath)
        time = get_datetime(fpath, filename_datetime_regex=filename_datetime_regex)
        if not time:
            raise ValueError(
                f"File {str(fpath.name)} does not contain a valid timestamp in the"
                " filename"
            )
        input_datetimes.append(time)

    # Figure out the variables we want to drop, since that's what xarray needs for open_dataset
    if keep_vars:
        print("Determining drop_vars from keep_vars...")
        # Read in the first RAMS file to get the variable names
        present_vars = xr.open_dataset(input_filenames[0]).data_vars
        drop_vars = [x for x in present_vars if x not in keep_vars]

    # Have to handle RAMS dimension names and ordering as part of the actual
    # preprocess argument to open_mfdataset, since it won't let you read in files
    # if these conflict
    def _sanitized_preprocess(ds):
        if fill_dim_names:
            # Get dimension name mapping if we need it
            if dim_names:
                # Just get the values
                _dimension_names = dim_names
                dimension_values = get_rams_dimension_values(
                    header_filepath=to_header_filepath(ds.encoding["source"]),
                    grid_number=get_grid_number(ds.encoding["source"]),
                )
            else:
                # Get the dimension names and values
                _dimension_names, dimension_values = infer_rams_dimensions(
                    single_time_rams_ds=ds,
                    grid_number=get_grid_number(ds.encoding["source"]),
                )
            ds = fill_rams_output_dimensions(
                ds=ds,
                dimension_names_mapping=_dimension_names,
                dimension_values=dimension_values,
            )
        if not keep_unknown_dims:
            # Drop any remaining phony dims and variables that have those dimensions
            phony_dims = [dim for dim in ds.dims if dim.startswith("phony_")]
            if phony_dims:
                # Find variables that have phony dimensions
                vars_with_phony_dims = [
                    var
                    for var in ds.data_vars
                    if any(phony_dim in ds[var].dims for phony_dim in phony_dims)
                ]
                # Drop the variables, which will drop unused dimensions
                ds = ds.drop_vars(vars_with_phony_dims)
        if preprocess:
            ds = preprocess(ds)
        return ds

    # The main difference between the serial and parallel processing is that serial calls xr.open_dataset and
    # parallel calls xr.open_mfdataset; the former requires manual concatenation while the latter handles it
    # within open_mfdataset. Note that there is actually a parallel argument to open_mfdataset, i.e. it is possible
    # to use this function serially, but the only reason not to use open_mfdataset is dask/xarray installation/version
    # issues and if these are resolved then the parallel option should work, so we don't currently have an option
    # for using open_mfdataset with `parallel=False` since I'm not sure why that would be useful.
    if parallel:
        maybe_print(
            f"Reading and concatenating {len(input_filenames)} individual timestep"
            " outputs..."
        )
        from dask.diagnostics import ProgressBar
        from contextlib import nullcontext

        open_ds_context_manager = nullcontext if silent else ProgressBar
        # Now actually call open_mfdataset
        with open_ds_context_manager():
            ds = xr.open_mfdataset(
                input_filenames,
                concat_dim=time_dim_name,
                combine="nested",
                preprocess=_sanitized_preprocess,
                phony_dims="sort",
                engine="h5netcdf",
                drop_variables=drop_vars,
                parallel=True,
                chunks=chunks,
                **open_dataset_kwargs,
            )
    # Serial reading logic
    else:
        maybe_print(f"Reading {len(input_filenames)} individual timestep outputs...")
        # Create a list to store the datasets we'll co=ncat
        to_concat = []
        wrapped_to_read = tqdm(input_filenames) if not silent else input_filenames
        for ds_path in wrapped_to_read:
            # Read in this dataset
            ds = xr.open_dataset(
                ds_path,
                drop_variables=drop_vars,
                engine="h5netcdf",
                phony_dims="sort",
                **open_dataset_kwargs,
            )
            # Append this to our list
            to_concat.append(ds)
        # Now concatenate along the time dimension
        if len(to_concat) > 1:
            if concatenate:
                maybe_print("Concatenating along time...")
                ds = xr.concat(to_concat, dim=time_dim_name)
            else:
                ds = to_concat
        else:
            ds = to_concat[0]

    # Assign file datetimes
    ds = ds.assign_coords(**{time_dim_name: input_datetimes})

    # Sort this across time
    ds = ds.sortby(time_dim_name)

    # Align the chunks if we used dask
    if parallel:
        ds = ds.unify_chunks()

    # Give the dataset units if we should
    if units:
        ds = ds.pint.quantify(
            RAMS_VARIABLES_DF.set_index("name")["units"].to_dict(), unit_registry=ureg
        )
    # Give the variables attributes either way
    rams_attrs_dicts = RAMS_VARIABLES_DF.set_index("name").to_dict(orient="index")
    for var in ds.data_vars:
        ds[var] = ds[var].assign_attrs(rams_attrs_dicts.get(var, {}))
    return ds


def write_rams_formatted_sounding(
    df: pd.DataFrame, output_path: PathLike, second_copy: Optional[PathLike] = None
) -> None:
    """Write sounding data to RAMS-formatted CSV file.

    Args:
        df: DataFrame containing sounding data with required columns
        output_path: Primary output file path
        second_copy: Optional secondary output path for backup

    Raises:
        ValueError: If required columns are missing or data validation fails
    """
    # First do a bunch of checks on the input data
    # Make sure we have all of the required columns
    if not all([x in df.columns for x in SOUNDING_NAMELIST_VARIABLES]):
        raise ValueError(
            f"Sounding dataframes must contain columns {SOUNDING_NAMELIST_VARIABLES}"
        )
    # Make sure pressure is monotonically decreasing and has no duplicate values
    if not (df["PS"].is_monotonic_decreasing and df["PS"].nunique() == len(df)):
        raise ValueError(
            "'PS' field must be monotonically decreasing with no duplicate values"
        )
    # Should be all good, write it out
    # Always write to output_path, and to second_copy if it's passed
    output_paths = [output_path]
    if second_copy:
        output_paths.append(second_copy)
    for this_output_path in output_paths:
        df[SOUNDING_NAMELIST_VARIABLES].to_csv(
            str(this_output_path),
            sep=",",
            header=False,
            index=False,
            float_format="%.4f",
            lineterminator=",\n",
        )


def get_z_levels(
    deltaz: float,
    dzrat: float,
    dzmax: float,
    nnzp: Optional[int] = None,
    max_height: Optional[float] = None,
) -> np.ndarray:
    """Generate z-coordinate levels for RAMS vertical grid.

    Args:
        deltaz: Initial vertical grid spacing
        dzrat: Grid stretch ratio
        dzmax: Maximum grid spacing
        nnzp: Number of vertical levels (alternative to max_height)
        max_height: Maximum height to reach (alternative to nnzp)

    Returns:
        Array of z-coordinate levels

    Raises:
        ValueError: If neither nnzp nor max_height is provided
    """
    if not nnzp and not max_height:
        raise ValueError("Must pass one of nnzp or max_height")
    # Could maybe do this analytically but we'll just brute force it
    # First there's a sub-ground layer and immediately above ground layer, each of height deltaz / 2
    heights = [-1 * deltaz / 2, deltaz / 2]

    def need_more_heights():
        if nnzp:
            return len(heights) <= nnzp
        elif max_height:
            return heights[-1] < max_height

    while need_more_heights():
        deltaz = min(deltaz * dzrat, dzmax)
        heights.append(heights[-1] + deltaz)
    return np.array(heights)


def format_sounding_field_ramsin_str(values: Union[List[float], np.ndarray]) -> str:
    """Format sounding field values as RAMSIN-compatible string.

    Args:
        values: Array of values for a sounding field

    Returns:
        Formatted string suitable for RAMSIN files
    """
    values = np.array(values)

    return ",\n          ".join([
        np.array2string(
            values[ix : ix + 5],
            formatter={"float_kind": lambda x: "%.4f" % x},
            separator=",    ",
        )[1:-1]
        for ix in range(0, len(values), 5)
    ])


def split_snowfall_nodelists(parameter_sets: Dict[str, Any]) -> Dict[str, List[str]]:
    """Split parameter sets across snowfall cluster nodes.

    Args:
        parameter_sets: Dictionary of parameter set configurations

    Returns:
        Dictionary mapping parameter set names to node specifications
    """
    node_assignments = {
        k: int(ix // (len(parameter_sets) / 3)) + 1
        for ix, k in enumerate(parameter_sets.keys())
    }
    n_jobs_per_node = {1: 0, 2: 0, 3: 0}
    for v in node_assignments.values():
        n_jobs_per_node[v] += 1
    n_cores_per_job = {k: 64 // v for k, v in n_jobs_per_node.items()}
    return {
        k: [f"snowfall{node_assignments[k]}:{n_cores_per_job[node_assignments[k]]}"]
        for ix, (k, v) in enumerate(parameter_sets.items())
    }


def parse_rams_stdout_walltimes(
    rams_stdout_path: PathLike, plot: bool = True
) -> Tuple[List[float], List[float]]:
    """Parse walltime information from RAMS stdout file.

    Args:
        rams_stdout_path: Path to RAMS stdout file
        plot: Whether to create a plot of walltime vs simulation time

    Returns:
        Tuple of (simulation_times, walltimes) lists
    """
    sim_times = []
    walltimes = []
    with Path(rams_stdout_path).open("r") as f:
        for line in f.readlines():
            maybe_match = re.search(
                r"Timestep.*Sim time\(sec\)=\s*([0-9\.]+).*Wall"
                r" time\(sec\)=\s*([0-9\.]+)",
                line,
            )
            if maybe_match:
                sim_times.append(float(maybe_match.group(1)))
                walltimes.append(float(maybe_match.group(2)))
    # Drop the first entry from each, since it's always way longer
    sim_times = sim_times[1:]
    walltimes = walltimes[1:]

    if plot:
        fig, ax = plt.subplots()
        ax.plot(sim_times, walltimes)
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel("Walltime per timestep (s)")
    return (sim_times, walltimes)


def calculate_thermodynamic_variables(
    ds: xr.Dataset, fail_if_missing_vars: bool = False
) -> xr.Dataset:
    """Calculate basic derived thermodynamic and physical variables from RAMS output.

    This function computes a comprehensive set of thermodynamic variables from basic
    RAMS model output (or any dataset with identical variable names, such as
    trajectories) including temperature, pressure, humidity variables, buoyancy,
    and hydrometeor classifications.

    Args:
        ds: xarray Dataset containing RAMS model output with basic variables
        fail_if_missing_vars: If True, raise ValueError when required variables are missing.
            If False, silently skip calculations that cannot be performed due to missing variables.

    Returns:
        Dataset with additional derived thermodynamic variables added:
            - T: Temperature (K)
            - R_condensate: Condensate mixing ratio (kg/kg)
            - P: Pressure (hPa)
            - dewpoint: Dewpoint temperature (K)
            - vapor_pressure: Water vapor pressure (hPa)
            - theta_e: Equivalent potential temperature (K)
            - saturation_vapor_pressure: Saturation vapor pressure (hPa)
            - air_mass: Air mass in each grid cell (kg)
            - RH: Relative humidity (fraction)
            - supersaturated: Boolean flag for supersaturated conditions
            - theta_v: Virtual potential temperature (K)
            - theta_rho: Density potential temperature (K)
            - buoyancy: Buoyancy acceleration (m/s²)
            - R_liquid: Total liquid water mixing ratio (kg/kg)
            - R_ice: Total ice mixing ratio (kg/kg)
            - PCPRR_mm_hr: Precipitation rate (mm/hr)

    Raises:
        ValueError: If fail_if_missing_vars is True and required variables are missing

    Note:
        Required input variables for full calculation: PI, THETA, RTP, RV, DN0,
        RCP, RRP, RPP, RSP, RAP, RGP, RHP, PCPRR. Dimensions x and y are also required
        for buoyancy calculation. The function will calculate
        whatever subset of derived variables is possible based on available inputs.
    """
    needed_vars = [
        "PI",
        "THETA",
        "RTP",
        "RV",
        "DN0",
        "RCP",
        "RRP",
        "RPP",
        "RSP",
        "RAP",
        "RGP",
        "RHP",
    ]

    # Write a little helper function
    # In hindsight I wish I had done this by writing a function that takes a function
    # that creates the new variable and then silently continuing if that function
    # throws an error, rather than all of these manual checks; a project for another day
    def vars_are_present(vars):
        return all([x in ds.data_vars for x in vars])

    if not vars_are_present(needed_vars) and fail_if_missing_vars:
        raise ValueError(
            "Not all variables needed for thermodynamic variable calculationsare"
            " present in dataset and fail_if_vars_missing was passed as True;"
            f" requiredvariables are {needed_vars}"
        )
    if vars_are_present(["PI", "THETA"]):
        ds["T"] = ds["PI"] * ds["THETA"] / 1004.0

    if vars_are_present(["RTP", "RV"]):
        ds["R_condensate"] = ds["RTP"] - ds["RV"]

    if vars_are_present(["PI"]):
        ds["P"] = 1000.0 * ((ds["PI"] / 1004.0) ** (1004.0 / 287.0))

    if vars_are_present(["P", "RV"]):
        vp = mpc.vapor_pressure(ds["P"] * units("hPa"), ds["RV"] * units("kg/kg"))
        ds["dewpoint"] = mpc.dewpoint(vp).pint.to("K").pint.dequantify()
        ds["vapor_pressure"] = vp.pint.to("hPa").pint.dequantify()

    if vars_are_present(["P", "RV"]):
        ds["theta_e"] = (
            mpc.equivalent_potential_temperature(
                pressure=ds["P"] * units("hPa"),
                temperature=ds["T"] * units("K"),
                dewpoint=ds["dewpoint"] * units("K"),
            )
            .pint.to("K")
            .pint.dequantify()
        )

    if vars_are_present(["T"]):
        ds["saturation_vapor_pressure"] = (
            mpc.saturation_vapor_pressure(ds["T"] * units("K"))
            .pint.to("hPa")
            .pint.dequantify()
        )

    if vars_are_present(["DN0"]):
        ds["air_mass"] = ds["DN0"] * 500**2 * ds["z"].diff(dim="z")

    if vars_are_present(["P", "T", "RV"]):
        ds["RH"] = mpc.relative_humidity_from_mixing_ratio(
            ds["P"] * units("hPa"), ds["T"] * units("K"), ds["RV"]
        ).pint.dequantify()
        ds["supersaturated"] = ds["RH"] >= 1
        # Virtual potential temperature
        ds["theta_v"] = mpc.virtual_potential_temperature(
            pressure=ds["P"] * units("hPa"),
            temperature=ds["T"] * units("K"),
            mixing_ratio=ds["RV"],
        )

    if vars_are_present(["THETA", "RV", "R_condensate"]):
        ds["theta_rho"] = ds["THETA"] * (1 + 0.608 * ds["RV"] - ds["R_condensate"])

    if vars_are_present(["theta_rho"]) and "x" in ds.dims and "y" in ds.dims:
        tr_layer_mean = ds["theta_rho"].mean(["x", "y"])
        ds["buoyancy"] = (
            mpconstants.g * (ds["theta_rho"] - tr_layer_mean) / tr_layer_mean
        ).pint.dequantify()

    if vars_are_present(["RCP", "RRP"]):
        ds["R_liquid"] = ds["RCP"] + ds["RRP"]

    if vars_are_present(["RPP", "RSP", "RAP", "RGP", "RHP"]):
        ds["R_ice"] = ds["RPP"] + ds["RSP"] + ds["RAP"] + ds["RGP"] + ds["RHP"]

    if vars_are_present(["PCPRR"]):
        # Create a version of pcprr in mm/hr
        # 1 kg/m^2/s = 1 mm/s (since density of water = 1000 kg/m^3)
        ds["PCPRR_mm_hr"] = (
            (ds["PCPRR"] * units("kg/m^2/s") / mpconstants.density_water)
            .pint.to("mm/hr")
            .pint.dequantify()
        )

    if vars_are_present(["vapor_pressure", "P"]):
        ds["mixing_ratio"] = mpc.mixing_ratio(
            partial_press=ds["vapor_pressure"] * units("hPa"),
            total_press=ds["P"] * units("hPa"),
        ).pint.dequantify()

    if vars_are_present(["P", "T", "mixing_ratio"]):
        ds["air_density"] = (
            mpc.density(
                pressure=ds["P"] * units("hPa"),
                temperature=ds["T"] * units("K"),
                mixing_ratio=ds["mixing_ratio"],
            )
            .pint.to("kg/m^3")
            .pint.dequantify()
        )

    return ds


def calculate_derived_variables(
    storm_ds: xr.Dataset,
) -> xr.Dataset:
    """Calculate derived variables and perform basic preprocessing on RAMS output.

    Args:
        storm_ds: xarray Dataset containing RAMS storm output

    Returns:
        Dataset with derived variables and preprocessing applied
    """
    storm_ds = calculate_thermodynamic_variables(storm_ds)

    # Shift the x and y coords so that they start from 0
    storm_ds["x"] = storm_ds["x"] - min(storm_ds["x"])
    storm_ds["y"] = storm_ds["y"] - min(storm_ds["y"])

    # Add a time in minutes coordinate
    storm_ds = storm_ds.assign_coords(
        t_minutes=(storm_ds["time"] - storm_ds["time"].values[0]).dt.total_seconds()
        // 60
    )

    # Add horizontal vorticity and divergence
    # Vertical vorticity
    storm_ds["vertical_vorticity"] = storm_ds["VC"].differentiate("x") - storm_ds[
        "UC"
    ].differentiate("y")
    # Horizontal divergence
    storm_ds["divergence"] = storm_ds["UC"].differentiate("x") + storm_ds[
        "VC"
    ].differentiate("y")

    # Add the middle values of x and y as attributes
    for var in ["x", "y"]:
        storm_ds[f"{var}_middle"] = storm_ds[var].max().values / 2
        # The // 2 makes this inexact, but this should only be used for rough things
        # anyway
        storm_ds[f"{var}_middle_ix"] = len(storm_ds[var]) // 2

    return storm_ds


def calculate_bsr_variables(
    ds: xr.Dataset, base_state: xr.Dataset, bsr_variables=[]
) -> xr.Dataset:
    """
    Calculate versions of variables in ds relative to a base_state.

    The base state
    is horizontally averaged to get a vertical profile, which is then subtracted
    from the actual values of the variable in ds to get the base-state relative
    version of the variable.

    Args:
        ds (xr.Dataset): The input dataset containing variables to calculate base-state
            relative versions for.
        base_state (xr.Dataset): The base state dataset used for reference. Must not
            contain a time dimension.
        bsr_variables (list, optional): List of variable names to calculate base-state
            relative versions for. If empty, uses DEFAULT_BSR_VARIABLES. Defaults to [].

    Returns:
        xr.Dataset: A copy of the input dataset with additional variables named
            '{var}_bsr' containing the base-state relative versions of the specified
            variables.

    Raises:
        ValueError: If base_state dataset contains a time dimension.
    """
    # Check that base_state doesn't have a time dimension
    if "time" in base_state.dims:
        raise ValueError(
            "base_state dataset must not have a time dimension, to avoid confusion"
        )
    # Copy the original dataset
    ds = ds.copy()
    # Calculate the layer-averaged base state dataset
    base_state = base_state.mean(["x", "y"])
    # Do the calculation
    for var in bsr_variables or DEFAULT_BSR_VARIABLES:
        # Calculate these for whatever variables are present
        if var in ds.data_vars:
            ds[f"{var}_bsr"] = ds[var] - base_state[var]
    return ds


def to_t_minutes(time_values, start_time):
    # Massage start time if needed
    try:
        start_time = start_time.to_numpy()
    except:
        pass
    if isinstance(time_values, (np.ndarray, xr.DataArray)):
        return (time_values - start_time).dt.total_seconds() // 60
    else:
        return [int((x - start_time) / np.timedelta64(1, "m")) for x in time_values]


def with_t_minutes_coord(ds, start_time=None):
    start_time = start_time or ds["time"].min().values
    return ds.assign_coords(t_minutes=to_t_minutes(ds["time"], start_time=start_time))


def dask_diagnostics(ds):
    for var_name in ds.data_vars:
        var = ds[var_name]
        if hasattr(var.data, "chunksize"):
            chunks = var.data.chunksize
            n_chunks = var.data.npartitions
            bytes_per = np.prod(chunks) * var.dtype.itemsize
            mb_per = bytes_per / 1e6
            total_mb = mb_per * n_chunks
            print(
                f"{var_name:20s}: {chunks} | {n_chunks:6d} chunks |"
                f" {mb_per:6.2f} MB/chunk | {total_mb:8.1f} MB total"
            )


def reload_intermediate(
    ds,
    cache_dir,
    cache_name=None,
    force=False,
    auto_analyze=True,
    min_graph_depth=5,
    min_tasks=1000,
    verbose=True,
):
    """
    Cache an xarray dataset/dataarray to disk and read it back, potentially
    improving performance by breaking up large task graphs.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        The dataset or dataarray to cache
    cache_path : Path or str, optional
        Path to cache the data. If None, creates a temporary file in DERIVED_DIR
    force : bool, default False
        If True, always cache even if auto-analysis suggests it won't help
    auto_analyze : bool, default True
        If True, analyze the task graph to determine if caching is likely to help.
        If False, always cache (unless file already exists)
    min_graph_depth : int, default 5
        Minimum graph depth to consider caching beneficial
    min_tasks : int, default 1000
        Minimum number of tasks to consider caching beneficial
    verbose : bool, default True
        If True, print information about the caching decision

    Returns
    -------
    xr.Dataset or xr.DataArray
        The cached and reloaded dataset/dataarray

    Notes
    -----
    This function helps work around Dask scheduler overhead by materializing
    intermediate results. It's most useful when:
    - Task graphs are very deep (many chained operations)
    - You'll reuse the result multiple times
    - Graph optimization is taking a long time
    """
    import hashlib

    cache_dir = Path(cache_dir)

    # Generate cache path if not provided
    if not cache_name:
        # Create a hash of the dataset structure for a unique filename
        hash_str = hashlib.md5(
            str(ds.dims).encode() + str(sorted(ds.data_vars)).encode()
        ).hexdigest()[:8]
        cache_path = cache_dir / f"cache_{hash_str}.zarr"
    else:
        cache_path = (cache_dir / cache_name).with_suffix(".zarr")

    # Check if data is lazy (dask-backed)
    has_dask = False
    if isinstance(ds, xr.Dataset):
        has_dask = any(hasattr(var.data, "dask") for var in ds.data_vars.values())
    elif isinstance(ds, xr.DataArray):
        has_dask = hasattr(ds.data, "dask")

    if not has_dask:
        if verbose:
            print("Data already computed, skipping cache")
        return ds

    # Analyze task graph if requested
    should_cache = force
    if auto_analyze and not force:
        try:
            # Get the dask graph
            if isinstance(ds, xr.Dataset):
                graph = ds.__dask_graph__()
            else:
                graph = ds.data.__dask_graph__()

            n_tasks = len(graph)

            # Estimate graph depth (rough heuristic)
            # Count the maximum chain length by looking at dependencies
            if hasattr(graph, "dependencies"):
                # For newer dask versions with HighLevelGraph
                dependencies = graph.dependencies
                max_depth = 0

                def get_depth(key, visited=None):
                    if visited is None:
                        visited = set()
                    if key in visited:
                        return 0
                    visited.add(key)

                    if key not in dependencies:
                        return 1

                    deps = dependencies[key]
                    if not deps:
                        return 1

                    return 1 + max(
                        (get_depth(d, visited.copy()) for d in deps), default=0
                    )

                # Sample a few keys to estimate depth
                sample_keys = list(dependencies.keys())[:10]
                depths = [get_depth(k) for k in sample_keys]
                max_depth = max(depths) if depths else 0
            else:
                # Fallback: just count tasks
                max_depth = min(n_tasks // 100, 10)  # Rough estimate

            # Decide whether to cache
            should_cache = (max_depth >= min_graph_depth) or (n_tasks >= min_tasks)

            if verbose:
                print(
                    f"Task graph analysis: {n_tasks} tasks, estimated depth"
                    f" ~{max_depth}"
                )
                print(f"Caching {'recommended' if should_cache else 'not recommended'}")

        except Exception as e:
            if verbose:
                print(f"Could not analyze graph ({e}), proceeding with cache")
            should_cache = True

    # If we decide not to cache, return original
    if not should_cache:
        if verbose:
            print("Skipping cache (use force=True to override)")
        return ds

    # Cache the data
    if cache_path.exists():
        if verbose:
            print(f"Loading from existing cache: {cache_path}")
        return xr.open_zarr(cache_path)

    if verbose:
        print(f"Computing and caching to: {cache_path}")

    # Compute and save
    ds_computed = ds.compute()
    ds_computed.to_zarr(cache_path, mode="w")

    # Read back in
    ds_cached = xr.open_zarr(cache_path)

    if verbose:
        print("Cached and reloaded successfully")

    return ds_cached
