"""Trajectory analysis and calculation utilities.

This module provides functions for initializing, calculating, and managing
atmospheric trajectories using Julia-based computation engines.
"""

import pandas as pd
import xarray as xr
from pathlib import Path, PosixPath
from jinja2 import Template
import subprocess
import datetime as dt
from typing import Any, Dict, List, Optional, Union

from .types_core import PathLike

from .utils import raise_if_exists
from .rams import RAMS_DT_STRFTIME_STR


def validate_parcel_initializations(pi_dicts: List[Dict[str, Any]]) -> None:
    """
    Validate parcel initialization dictionaries.

    Args:
        pi_dicts: List of parcel initialization dictionaries

    Raises:
        ValueError: If validation fails
    """
    required_keys = {"x", "y", "z", "initialization_time", "parcel_ix"}
    # Check if they all have the same keys
    first_keys = set(pi_dicts[0].keys())
    if not all(set(d.keys()) == first_keys for d in pi_dicts):
        raise ValueError(
            "All parcel initialization dictionaries must have the same keys"
        )
    if not all([x in first_keys for x in required_keys]):
        raise ValueError(f"Parcel initializations require keys {required_keys}")
    # Check parcel ixs aren't repeated
    parcel_ixs = [d["parcel_ix"] for d in pi_dicts]
    if not len(parcel_ixs) == len(set(parcel_ixs)):
        raise ValueError("parcel_ix values are not unique")
    extra_keys = [x for x in first_keys if x not in required_keys]
    print(f"Received extra keys {extra_keys}, propagating through")


def write_parcel_initializations(
    pi_dicts: List[Dict[str, Any]],
    output_fpath: PathLike,
    exist_ok: bool = False,
    mkdir: bool = False,
    assign_parcel_ixs: bool = False,
) -> pd.DataFrame:
    """
    Write parcel initializations to CSV file.

    Args:
        pi_dicts: List of parcel initialization dictionaries
        output_fpath: Output file path
        exist_ok: Whether to overwrite existing files
        mkdir: Whether to create parent directories
        assign_parcel_ixs: Whether to assign parcel indices automatically

    Returns:
        DataFrame containing the parcel initializations
    """
    validate_parcel_initializations(pi_dicts)
    if not exist_ok:
        raise_if_exists(output_fpath)
    if mkdir:
        output_fpath.parent.mkdir(parents=False, exist_ok=True)
    # Need to assign these parcel_ixs if they don't have them
    if assign_parcel_ixs:
        for parcel_ix, pi_dict in enumerate(pi_dicts):
            pi_dict["parcel_ix"] = parcel_ix
    # Now write to disk
    # Convert to a pandas dataframe for better datetime handling
    pi_df = pd.DataFrame(pi_dicts)
    pi_df.to_csv(output_fpath, index=False)
    return pi_df


def calculate_trajectories_julia(
    parcel_initialization_csv_path: PathLike,
    simulation_output_dir: PathLike,
    output_path: PathLike,
    trajectory_timestep: Union[int, float],
    trajectory_end_time: dt.datetime,
    back_trajectories: bool,
    restart_ds_path: Optional[PathLike] = None,
    tracked_scalars: List[str] = [],
    parallel: bool = True,
    grid_stagger: float = 0.5,
    write_interval: int = 1,
    flush_interval: Optional[int] = None,
    max_iterations: int = 5,
    raise_on_nonconvergence: bool = False,
    stop_on_early_convergence: bool = True,
    use_subdomain_limiting: bool = True,
    raise_on_oob: bool = True,
    exist_ok: bool = False,
    test_write: bool = True,
    progress_bar: bool = True,
    verbose: bool = True,
    pipe_stdout: bool = True,
    block: bool = True,
    dry_run: bool = False,
    save_on_error: bool = True,
) -> Optional[Union[xr.Dataset, subprocess.Popen]]:
    """Calculate atmospheric trajectories using Julia-based computation engine.

    Args:
        parcel_initialization_csv_path: Path to CSV file containing parcel initial conditions
        simulation_output_dir: Directory containing RAMS simulation output files
        output_path: Path where trajectory results will be saved
        trajectory_timestep: Time step for trajectory calculation (seconds)
        trajectory_end_time: End time for trajectory calculation
        back_trajectories: Whether to calculate backward trajectories
        restart_ds_path: Path to restart dataset for continuing calculations
        tracked_scalars: List of scalar variables to track along trajectories
        parallel: Whether to use parallel computation
        grid_stagger: Grid staggering parameter for interpolation
        write_interval: Interval for writing output (in timesteps)
        flush_interval: Interval for flushing output buffers
        max_iterations: Maximum iterations for convergence
        raise_on_nonconvergence: Whether to raise error on convergence failure
        stop_on_early_convergence: Whether to stop when converged early
        use_subdomain_limiting: Whether to use subdomain limiting for efficiency
        raise_on_oob: Whether to raise error when parcels go out of bounds
        exist_ok: Whether to overwrite existing output files
        test_write: Whether to test write permissions before computation
        progress_bar: Whether to show progress bar
        verbose: Whether to print verbose output
        pipe_stdout: Whether to pipe stdout to file
        block: Whether to wait for computation to complete
        dry_run: If True, generate script but don't execute
        save_on_error: Whether to save partial results on error

    Returns:
        Dataset containing trajectory results (if block=True and not dry_run),
        subprocess.Popen object (if block=False), or None (if dry_run or file doesn't exist)
    """
    # Read the template file
    template_path = Path(__file__).parent.joinpath("../run_trajectories_template.jl")
    with open(template_path, "r") as template_file:
        template_content = template_file.read()

    # Create a Jinja2 template object
    template = Template(template_content)

    # Prepare the arguments for templating
    template_args = {
        "parcel_initialization_csv_path": f'"{str(parcel_initialization_csv_path)}"',
        "simulation_output_dir": f'"{str(simulation_output_dir)}"',
        "output_path": f'"{str(output_path)}"',
        "trajectory_timestep": trajectory_timestep,
        "trajectory_end_time": (
            f'"{str(trajectory_end_time.strftime(RAMS_DT_STRFTIME_STR))}"'
        ),
        "back_trajectories": str(back_trajectories).lower(),
        "restart_ds_path": (
            f'"{str(restart_ds_path)}"' if restart_ds_path else "nothing"
        ),
        "tracked_scalars": str(tracked_scalars).replace("'", '"'),
        "parallel": str(parallel).lower(),
        "grid_stagger": grid_stagger,
        "write_interval": write_interval,
        "flush_interval": flush_interval or 0,
        "max_iterations": max_iterations,
        "stop_on_early_convergence": str(stop_on_early_convergence).lower(),
        "raise_on_nonconvergence": str(raise_on_nonconvergence).lower(),
        "use_subdomain_limiting": str(use_subdomain_limiting).lower(),
        "raise_on_oob": str(raise_on_oob).lower(),
        "exist_ok": str(exist_ok).lower(),
        "test_write": str(test_write).lower(),
        "progress_bar": str(progress_bar).lower(),
        "verbose": str(verbose).lower(),
        "save_on_error": str(save_on_error).lower(),
    }

    # Render the template with the arguments
    rendered_script = template.render(template_args)

    # Write the rendered script to a temporary file
    print(rendered_script)
    temp_script_path = Path(output_path).with_suffix(".jl")
    print(f"\nWriting julia script to {temp_script_path}")
    with open(temp_script_path, "w") as temp_script_file:
        temp_script_file.write(rendered_script)

    # Execute the Julia script using subprocess

    if not dry_run:
        stdout_path = Path(output_path).with_suffix(".stdout")
        print(f"Writing stdout to {stdout_path}")
        with stdout_path.open("w") as stdout_f:
            sp = subprocess.Popen(
                ["julia", "--threads", "64", temp_script_path],
                start_new_session=True,
                stdout=stdout_f if pipe_stdout else None,
                stderr=stdout_f if pipe_stdout else None,
            )
        if block:
            try:
                sp.wait()
            finally:
                # If we interrupt this it won't kill the processes, so we implement that manually
                sp.kill()
                return sp
        else:
            return sp

    if Path(output_path).exists():
        return xr.open_dataset(output_path)
    else:
        print("output_path does not exist, returning None")
        return None
