#   Imports and Setup
#   ================

using NCDatasets
using Interpolations
using LabelledArrays
using Dates
using FilePathsBase;
using FilePathsBase: /;
using OrderedCollections
using Plots
using DataFrames
using CSV
using JLD2
using ProgressBars
using Base

import Base.@kwdef

ENV["JULIA_PROGRESS_BAR_ENABLED"] = "true"

#   Utility Functions
#   ================

# Overwrite print to force flushing to stdout
function println(x)
	Base.println(x)
	flush(stdout)
	flush(stderr)
end

function print(x)
	Base.print(x)
	flush(stdout)
	flush(stderr)
end

# File path utilities
function stem(fpath)
	return splitdir(splitext(fpath)[1])[2]
end

function ext(fpath)
	return splitext(fpath)[2]
end

function with_stem(fpath, new_stem)
	fpath = string(fpath)
	(fpath, ext) = splitext(fpath)
	(parent, old_stem) = splitdir(fpath)
	return Path(parent) / (new_stem * ext)
end

function with_ext(fpath, ext)
	return Path(splitext(fpath)[1] * ext)
end

function first_value(d::AbstractDict)
	return d[collect(keys(d))[1]]
end

#   Constants and Type Definitions
#   =============================

# Format of RAMS filepath datetimes
const RAMS_DATEFORMAT = dateformat"yyyy-mm-dd-HHMMSS"
const wind_vars = ["UC", "VC", "WC"]
const POSITION_VARS = ["x", "y", "z"]
const POSITION_IX_VARS = ["x_ix", "y_ix", "z_ix"]

ParcelPositionIxs = @SLVector Float64 (:x_ix, :y_ix, :z_ix)
ParcelPosition = @SLVector Float64 (:x, :y, :z)

@kwdef struct CorrectionIterationResult
	correction::ParcelPositionIxs
	corrected_position::ParcelPositionIxs
	converged::Bool
end

@kwdef struct CorrectionResult
	correction_iteration_results::Vector{CorrectionIterationResult}
	n_iterations::Integer
	converged::Bool
	oob::Bool
end

@kwdef struct ParcelState
	position_ixs::ParcelPositionIxs
	scalars::Dict{String, Float64}
	correction_result::CorrectionResult
end

@kwdef struct ParcelInitialization
	parcel_ix::Integer
	initial_x::Float64
	initial_y::Float64
	initial_z::Float64
	initialization_time::DateTime
	propagated_vars::Dict{String, Any}
end

# Circular buffer for trajectory data
mutable struct TrajectoryBuffer
	time_t_plus_1_states::Union{Dict{Integer, ParcelState}, Nothing}
	time_t_plus_1::Union{DateTime, Nothing}
	time_t_states::Union{Dict{Integer, ParcelState}, Nothing}
	time_t::Union{DateTime, Nothing}
end

ParcelStateDict = Dict{Integer, ParcelState}

function shift_buffer!(buffer::TrajectoryBuffer)
	# Shift: previous <- current, current <- new
	buffer.time_t_states, buffer.time_t_plus_1_states =
		buffer.time_t_plus_1_states, nothing
	buffer.time_t, buffer.time_t_plus_1 = buffer.time_t_plus_1, nothing
end

# NetCDF streaming writer
mutable struct TrajectoryWriter
	dataset::NCDataset
	parcel_ixs::Vector{Int}
	tracked_scalars::Vector{String}
	max_timesteps::Int
end

#   Grid and Data Reading Functions
#   ==============================

function read_rams_grid(rams_output_fpath, grid_number=1)
	# Get the path to the header file
	# The last two characters of the RAMS output filepath are the grid, but all
	# the info is in one header file, so remove these and replace with "head" to
	# get to the right path
	rams_output_fpath = string(rams_output_fpath)
	# Fix the grid string
	grid_number = lpad(string(grid_number), 2, "0")
	header_fpath =
		with_ext(
			with_stem(rams_output_fpath, stem(rams_output_fpath)[1:(end-2)] * "head"),
			".txt",
		) |> string
	# Now parse the header file
	header_name_dimension_dict =
		Dict(var => "__$(var)tn$(grid_number)" for var in ["x", "y", "z"])
	# Read the whole file
	header_lines = [strip(x) for x in readlines(header_fpath)]
	dimension_values = Dict()
	for (var_name, var_start_string) in header_name_dimension_dict
		# Find the line where this string occurrs
		var_start_line_number = findfirst(x -> x == var_start_string, header_lines)
		if isnothing(var_start_line_number)
			error("Phrase $(var_start_string) was not found in RAMS header file")
		end
		# The next line gives how many values it has
		n_levels = parse(Int, header_lines[var_start_line_number+1])
		# Read this many lines
		levels = [
			parse(Float64, x) for x in
			header_lines[(var_start_line_number+2):(var_start_line_number+1+n_levels)]
		]
		# Offset this so that it starts at 0 in x and y
		if var_name in ["x", "y"]
			levels = levels .- levels[1]
		end
		dimension_values[var_name] = levels
	end
	return dimension_values
end

function read_parcel_initial_states(fpath)::Dict{Int, ParcelInitialization}
	parcel_initial_positions_csv = CSV.File(fpath, stringtype=String)
	parcel_initializations = Dict{Int, ParcelInitialization}()

	required_keys = ["x", "y", "z", "initialization_time", "parcel_ix"]

	for row in parcel_initial_positions_csv
		this_initialization_time =
			DateTime(row[:initialization_time], "yyyy-mm-dd HH:MM:SS")
		# Create a ParcelPosition from this row
		# Get all properties that aren't in required_keys
		propagated_vars = Dict{String, Any}()
		for prop in propertynames(row)
			prop_str = string(prop)
			if !(prop_str in required_keys)
				propagated_vars[prop_str] = getproperty(row, prop)
			end
		end
		this_initialization = ParcelInitialization(
			parcel_ix=row[:parcel_ix],
			initial_x=row[:x],
			initial_y=row[:y],
			initial_z=row[:z],
			initialization_time=this_initialization_time,
			propagated_vars=propagated_vars,
		)
		parcel_initializations[row[:parcel_ix]] = this_initialization
	end
	return parcel_initializations
end

function get_simulation_output_path(
	simulation_output_dir::Union{PosixPath, AbstractString},
	dt::DateTime,
)::PosixPath
	return Path(simulation_output_dir) /
		   "a-L-$(Dates.format(dt, RAMS_DATEFORMAT))-g1.h5"
end

#   Interpolation and Grid Setup Functions
#   =====================================

function setup_grid_interpolators(grid_coords, grid_stagger)
	# Create indexes of each of the coords in the same way
	grid_ixs = Dict(
		coord => 1:size(grid_coords[coord])[1] |> collect for
		coord in keys(grid_coords)
	)

	# Create grids of coordinate indices for interpolation
	grid_ix_nodes = Dict(
		"UC" => float.((
			grid_ixs["x"] .+ grid_stagger["x"],
			grid_ixs["y"],
			grid_ixs["z"],
		)),
		"VC" => float.((
			grid_ixs["x"],
			grid_ixs["y"] .+ grid_stagger["y"],
			grid_ixs["z"],
		)),
		"WC" => float.((
			grid_ixs["x"],
			grid_ixs["y"],
			grid_ixs["z"] .+ grid_stagger["z"],
		)),
		"scalars" => float.((grid_ixs["x"], grid_ixs["y"], grid_ixs["z"])),
	)

	# Define interpolators
	coord_to_index_interpolators = Dict(
		coord =>
			interpolate((grid_coords[coord],), grid_ixs[coord], Gridded(Linear()))
		for coord in keys(grid_ixs)
	)
	index_to_coord_interpolators = Dict(
		coord =>
			interpolate((grid_ixs[coord],), grid_coords[coord], Gridded(Linear()))
		for coord in keys(grid_ixs)
	)

	return grid_ixs,
	grid_ix_nodes,
	coord_to_index_interpolators,
	index_to_coord_interpolators
end

function get_interpolators(
	simulation_output_path,
	interped_vars,
	grid_ix_nodes,
	grid_nx,
	grid_ny;
	subdomain_bounds=Vector{ParcelPositionIxs}(),
)::Dict{String, Function}
	# Get all of the interpolators we need for a given time
	interpolators = Dict()
	this_ds = NCDataset(simulation_output_path |> string, "r")
	# Create indexers for limiting the reading in of the data to the subdomain
	if (length(subdomain_bounds) == 2)
		subdomain_indexers = [
			range(Integer(subdomain_bounds[1].x_ix), Integer(subdomain_bounds[2].x_ix)),
			range(Integer(subdomain_bounds[1].y_ix), Integer(subdomain_bounds[2].y_ix)),
			range(Integer(subdomain_bounds[1].z_ix), Integer(subdomain_bounds[2].z_ix)),
		]
		# Reverse these if it's mirror world
		original_subdomain_indexers = copy(subdomain_indexers)

	elseif (length(subdomain_bounds) == 0)
		# Simplest way to handle this is to just still select from the arrays
		# using indexes, but just use colon objects to keep all the data
		subdomain_indexers=[Colon(), Colon(), Colon()]
		original_subdomain_indexers = copy(subdomain_indexers)
	else
		# Something's wrong
		error(
			"subdomain_bounds must be of length 0 (no bounds) or 2 (lower bound and upper bound)",
		)
	end

	for var in interped_vars
		# Create a var object for the data, without actually reading the values in
		this_var_data = this_ds[var]
		# Get the number of dimensions on this variable
		this_var_n_dims = length(size(this_var_data))
		# Subset the indexers to the correct number of dimensions
		this_var_subdomain_indexers = subdomain_indexers[1:this_var_n_dims]

		# Subset the data to the subdomain
		this_var_data = this_var_data[this_var_subdomain_indexers...]
		# Pull the grid nodes with appropriate stagger and dimensions,
		# and subset them
		# This is a tuple of arrays, because it has to be for the interpolator
		this_var_grid_ix_nodes = Tuple([
			grid_ix_nodes[var in wind_vars ? var : "scalars"][i][this_var_subdomain_indexers[i]]
			for i in range(1, this_var_n_dims)
		])
		# Create an interpolator object that will only use the correct number
		# of dimensions for this variable
		# First make an interpolator
		this_interpolator =
			interpolate(this_var_grid_ix_nodes, this_var_data, Gridded(Linear()))
		function wrapped_interpolator(position_ixs...)
			return this_interpolator(position_ixs[1:this_var_n_dims]...)
		end
		interpolators[var] = wrapped_interpolator
	end
	close(this_ds)
	return interpolators
end

function interpolate_scalars(
	this_position_ix,
	interpolators;
	scalars,
)::Dict{String, Float64}
	return Dict(
		scalar => interpolators[scalar](
			this_position_ix[:x_ix],
			this_position_ix[:y_ix],
			this_position_ix[:z_ix],
		) for scalar in scalars
	)
end

#   Trajectory Calculation Functions
#   ===============================

function calculate_corrections_singleiter(
	current_position,
	this_time_interpolators,
	trajectory_timestep,
	grid_deltax,
	coord_to_index_interpolators,
	index_to_coord_interpolators,
	time_direction_factor,
)::Tuple{Dict{String, Float64}, ParcelPositionIxs}
	u = this_time_interpolators["UC"](
		current_position[:x_ix],
		current_position[:y_ix],
		current_position[:z_ix],
	)
	v = this_time_interpolators["VC"](
		current_position[:x_ix],
		current_position[:y_ix],
		current_position[:z_ix],
	)
	w = this_time_interpolators["WC"](
		current_position[:x_ix],
		current_position[:y_ix],
		current_position[:z_ix],
	)
	dxix::Float64 = u * Second(trajectory_timestep).value / grid_deltax
	dyix::Float64 = v * Second(trajectory_timestep).value / grid_deltax
	dz::Float64 = w * Second(trajectory_timestep).value
	# Calculate the new z position and convert it back to an index
	dzix::Float64 =
		coord_to_index_interpolators["z"](
			index_to_coord_interpolators["z"](current_position[:z_ix]) + dz,
		) - current_position[:z_ix]

	return Dict("u" => u, "v" => v, "w" => w),
	ParcelPositionIxs(dxix, dyix, dzix)*time_direction_factor
end

function has_converged(
	current_position::ParcelPositionIxs,
	previous_position::ParcelPositionIxs,
	index_to_coord_interpolators;
	max_horizontal_gridpoint_difference=0.1,
	max_vertical_difference=1,  # m
)
	# Define convergence as changing less than 1/10th of a horizontal gridpoint
	# and less than 1 m vertically, following Miltenberger et al. 2014
	delta_x_gridpoints = abs(current_position.x_ix - previous_position.x_ix)
	delta_y_gridpoints = abs(current_position.y_ix - previous_position.y_ix)
	# Since calculating delta z will require slightly more calculation, only
	# do so if we've converged horizontally
	if (delta_x_gridpoints > max_horizontal_gridpoint_difference) |
	   (delta_y_gridpoints > max_horizontal_gridpoint_difference)
		return false
	end
	# Now calculate whether we've converged in z
	delta_z = abs(
		index_to_coord_interpolators["z"](current_position.z_ix) -
		index_to_coord_interpolators["z"](current_position.z_ix),
	)
	# Since we already returned false if we didn't converge horizontally,
	# convergence is now just whether it converged in z, so return that
	return delta_z <= max_vertical_difference
end

function calculate_corrected_states(
	initial_position::ParcelPositionIxs,
	current_time_interpolators,
	next_time_interpolators,
	trajectory_timestep,
	grid_deltax,
	coord_to_index_interpolators,
	index_to_coord_interpolators,
	time_direction_factor,
	z_ix_z0,
	tracked_scalars;
	max_iterations=5,
	stop_on_early_convergence=true,
	raise_on_oob=true,
	raise_on_nonconvergence=false,
)::ParcelState
	# Initialize the oob flag to false
	oob = false
	correction_iteration_results =
		Vector{CorrectionIterationResult}(undef, max_iterations)
	correction_iteration_winds = Vector{Dict{String, Float64}}(undef, max_iterations)
	# Get the corrections for the current time (i.e. u(x, t_i))
	# This will always be corrections[1]; this is the only one that uses the
	# wind field at the current time, all of the subsequent corrections use
	# the interpolators from the next time
	initial_winds, initial_correction = calculate_corrections_singleiter(
		initial_position,
		current_time_interpolators,
		trajectory_timestep,
		grid_deltax,
		coord_to_index_interpolators,
		index_to_coord_interpolators,
		time_direction_factor,
	)
	# Get the new position this gives; in Miltenberger et al. 2014 this is
	# denoted as x_1
	initial_corrected_position =
		ParcelPositionIxs(initial_position + initial_correction)
	correction_iteration_results[1] = CorrectionIterationResult(
		correction=initial_correction,
		corrected_position=initial_corrected_position,
		converged=false,
	)
	correction_iteration_winds[1] = initial_winds
	try
		# Now calculate subsequent iterations
		# We start i_iteration at 2 because we count this^ as an iteration
		for i_iteration in range(2, max_iterations)
			# Calculate the wind/correction at x_(n-1) and t+1 (so use
			# next_time_interpolators)
			this_correction_winds, this_correction = calculate_corrections_singleiter(
				correction_iteration_results[i_iteration-1].corrected_position,
				next_time_interpolators,
				trajectory_timestep,
				grid_deltax,
				coord_to_index_interpolators,
				index_to_coord_interpolators,
				time_direction_factor,
			)
			# Now to get the new position, we take the mean of this correction and
			# the first correction (i.e. the correction we applied to get x_1) and
			# apply them to the initial position
			this_corrected_position = ParcelPositionIxs(
				initial_position +
				(correction_iteration_results[1].correction + this_correction)/2,
			)
			# Check if we converged whether or not we're stopping for it
			this_converged = has_converged(
				this_corrected_position,
				correction_iteration_results[i_iteration-1].corrected_position,
				index_to_coord_interpolators,
			)
			# Store the result from this iteration
			correction_iteration_results[i_iteration] = CorrectionIterationResult(
				correction=this_correction,
				corrected_position=this_corrected_position,
				converged=this_converged,
			)
			correction_iteration_winds[i_iteration] = this_correction_winds
			# This is complete if we're doing a fixed number of iterations; if we're
			# stopping on convergence, need to do that now
			if stop_on_early_convergence && this_converged
				correction_iteration_results =
					correction_iteration_results[1:i_iteration]
				correction_iteration_winds = correction_iteration_winds[1:i_iteration]
				break
			end
		end
	catch e
		# Handle the case where we're out of bounds
		if e isa BoundsError && !raise_on_oob
			# We can essentially just leave the corrections as they are, and 
			# use the last one, since we're out of bounds so this data is bad
			# anyway
			# So we don't actually need to do anything besides set the oob flag
			oob = true
			# Make a dummy correction iteration result
			correction_iteration_results =
				CorrectionIterationResult[CorrectionIterationResult(
					correction=ParcelPositionIxs(0, 0, 0),
					corrected_position=initial_position,
					converged=false,
				),]
			correction_iteration_winds = [initial_winds]
		else
			rethrow(e)
		end
	end

	# Handle the case where we require convergence but it didn't converge
	if raise_on_nonconvergence && !(correction_iteration_results[end].converged) && !oob
		error("Convergence was required but a parcel did not converge")
	end

	# We now have the final position; make sure we don't intersect terrain
	final_position = ParcelPositionIxs(
		correction_iteration_results[end].corrected_position[:x_ix],
		correction_iteration_results[end].corrected_position[:y_ix],
		max(correction_iteration_results[end].corrected_position[:z_ix], z_ix_z0),
	)

	# Get the winds so we can save time interpolating scalars
	final_winds = correction_iteration_winds[end]

	# Now interpolate the scalars (minus the winds, since we already have
	# them) onto this position
	interpolated_scalars_nowinds = interpolate_scalars(
		final_position,
		next_time_interpolators;
		scalars=[x for x in tracked_scalars if !(x in keys(final_winds))],
	)
	interpolated_scalars = merge(final_winds, interpolated_scalars_nowinds)
	return ParcelState(
		position_ixs=final_position,
		scalars=interpolated_scalars,
		correction_result=CorrectionResult(
			correction_iteration_results=correction_iteration_results,
			n_iterations=length(correction_iteration_results),
			converged=correction_iteration_results[end].converged,
			oob=oob,
		),
	)
end

#   Restart and Initialization Functions
#   ===================================

function load_restart_data(restart_ds_path, tracked_scalars, back_trajectories)
	throw("Presently deprecated")
	# Open the dataset
	restart_data = Dict()
	NCDataset(restart_ds_path) do restart_ds
		# Read in the data for all the variables, since we'll need it all
		for var in tqdm(keys(restart_ds))
			restart_data[var] = restart_ds[var] |> Array
		end
	end
	println(
		"Restarting from $((back_trajectories ? minimum : maximum)(restart_data["time"]))...",
	)

	# Iterate over these
	for (this_time_ix, this_time) in tqdm(enumerate(restart_data["time"]))
		# Make a dict for the parcels at this time
		this_time_parcels = Dict{Integer, ParcelState}()
		# Loop over the parcels
		for (parcel_ix_ix, parcel_ix) in enumerate(restart_data["parcel_ix"])
			# First check if this one has been initialized by this time
			if (
				!back_trajectories &&
				(restart_data["initialization_time"][parcel_ix_ix] > this_time)
			) || (
				back_trajectories &&
				(restart_data["initialization_time"][parcel_ix_ix] < this_time)
			)
				continue
			end
			# Get the position
			this_position_ix = ParcelPositionIxs(
				restart_data["x_ix"][this_time_ix, parcel_ix_ix],
				restart_data["y_ix"][this_time_ix, parcel_ix_ix],
				restart_data["z_ix"][this_time_ix, parcel_ix_ix],
			)
			# Get the scalars
			this_scalars = Dict(
				scalar => restart_data[scalar][this_time_ix, parcel_ix_ix] for
				scalar in tracked_scalars
			)
			# Get the correction result
			this_correction_result = CorrectionResult(
				# Don't have the iteration results so don't include them
				correction_iteration_results=Vector{CorrectionIterationResult}(),
				n_iterations=restart_data["n_iterations"][this_time_ix, parcel_ix_ix],
				converged=restart_data["converged"][this_time_ix, parcel_ix_ix],
				oob=restart_data["oob"][this_time_ix, parcel_ix_ix],
			)
			# Now save this as a ParcelState
			this_time_parcels[parcel_ix] = ParcelState(
				position_ixs=this_position_ix,
				scalars=this_scalars,
				correction_result=this_correction_result,
			)
		end
	end

	println("Done reading restart data")
	return restart_data
end

function initialize_new_parcels!(
	parcels_to_initialize,
	time_t_states,
	coord_to_index_interpolators,
	interpolators;
)

	# Get names of scalars from the interpolators
	tracked_scalars = [x for x in keys(interpolators[2]) if !(x in wind_vars)]

	if length(parcels_to_initialize) > 0
		println("Initializing $(length(parcels_to_initialize)) new parcels...")
	else
		return time_t_states
	end

	# We have the positions for these but need to get the scalars for them
	for parcel_initialization in parcels_to_initialize
		parcel_ix = parcel_initialization.parcel_ix
		# Check if this parcel is already present; this should only be the case if
		# we are doing a history restart, and then only if the last timestep
		# in the history had parcels initialized at that time
		# Don't re-initialize parcels that already exist
		if parcel_ix in keys(time_t_states)
			continue
		end
		# Interpolate initial positions from absolute coordinates to grid indices
		parcel_position_ix = ParcelPositionIxs(
			coord_to_index_interpolators["x"](parcel_initialization.initial_x),
			coord_to_index_interpolators["y"](parcel_initialization.initial_y),
			coord_to_index_interpolators["z"](parcel_initialization.initial_z),
		)

		# Interpolate the scalars onto these positions so we can make ParcelStates
		time_t_states[parcel_ix] = ParcelState(
			position_ixs=parcel_position_ix,
			# We haven't shuffled the interpolators yet, so the interpolator
			# at position 2 in the list of interpolators is now
			# the t interpolator; the subdomain limiting at the previous timestep
			# should have accounted for this
			scalars=interpolate_scalars(
				parcel_position_ix,
				interpolators[2];
				scalars=tracked_scalars,
			),
			# Just pass an empty vector for the correction iteration results
			correction_result=CorrectionResult(
				correction_iteration_results=Vector{CorrectionIterationResult}(),
				n_iterations=0,
				converged=true,
				oob=false,
			),
		)
	end

	return time_t_states
end

function get_subdomain_bounds(
	time_t_states,
	grid_ixs,
	use_subdomain_limiting,
	initialization_dict,
	time_t_plus_1,
)
	if use_subdomain_limiting & !(time_t_plus_1 in collect(keys(initialization_dict)))
		# Get the extent of the parcels at time t
		function get_bounds(bound_fn, varname)
			return bound_fn([
				getproperty(parcel_state.position_ixs, Symbol(varname)) for
				parcel_state in values(time_t_states)
			])
		end
		# Get the bounds, ensuring we don't go out of bounds
		# Besides near the bottom of the domain, it's probably a problem if
		# we're near going out of bounds, but the trajectory calculations
		# themselves should handle that, not this step which is just to cut
		# down on computation time
		lower_bound = ParcelPositionIxs(
			max((get_bounds(minimum, "x_ix") |> floor |> Integer) - 1, 1),
			max((get_bounds(minimum, "y_ix") |> floor |> Integer) - 1, 1),
			max((get_bounds(minimum, "z_ix") |> floor |> Integer) - 1, 1),
		)
		upper_bound = ParcelPositionIxs(
			min(
				(get_bounds(maximum, "x_ix") |> ceil |> Integer) + 1,
				maximum(grid_ixs["x"]),
			),
			min(
				(get_bounds(maximum, "y_ix") |> ceil |> Integer) + 1,
				maximum(grid_ixs["y"]),
			),
			min(
				(get_bounds(maximum, "z_ix") |> ceil |> Integer) + 1,
				maximum(grid_ixs["z"]),
			),
		)
		subdomain_bounds = [lower_bound, upper_bound]
	else
		subdomain_bounds::Vector{ParcelPositionIxs} = []
	end
	return subdomain_bounds
end

#   Output and File Writing Functions
#   ================================
function setup_trajectory_netcdf(
	output_path::String,
	parcel_ixs::Vector{Int},
	tracked_scalars::Vector{String},
	propagated_vars_types::Dict{String, Type},
	max_timesteps::Int;
	deflatelevel::Int=0,
	parcel_chunk_size=5000,
)::TrajectoryWriter
	# Get total number of parcels
	n_parcels = length(parcel_ixs)

	ds = NCDataset(output_path, "c")

	# Define dimensions
	defDim(ds, "parcel_ix", n_parcels)
	defDim(ds, "time", max_timesteps)

	# Create dimension variables
	parcel_var = defVar(ds, "parcel_ix", Int, ("parcel_ix",), deflatelevel=deflatelevel)
	parcel_var[:] = parcel_ixs

	# Define time variable
	defVar(
		ds,
		"time",
		Int,
		("time",),
		attrib=Dict(
			"units" => "seconds since 1970-01-01 00:00:00",
			"calendar" => "standard",
		),
		deflatelevel=deflatelevel,
	)

	# Define trajectory variables with chunking for efficient writes
	parcel_chunk_size = (min(n_parcels, parcel_chunk_size), 1)

	# Position variables (both index and coordinate)
	for coord in ["x", "y", "z"]
		defVar(
			ds,
			"$(coord)_ix",
			Float64,
			("parcel_ix", "time"),
			# chunksizes=parcel_chunk_size,
			fillvalue=-999.0,
			deflatelevel=deflatelevel,
		)
		defVar(
			ds,
			coord,
			Float64,
			("parcel_ix", "time"),
			# chunksizes=parcel_chunk_size,
			fillvalue=-999.0,
			deflatelevel=deflatelevel,
		)
	end

	# Trajectory calculation metadata variables
	defVar(
		ds,
		"converged",
		Int,
		("parcel_ix", "time"),
		# chunksizes=parcel_chunk_size,
		fillvalue=-1,
		deflatelevel=deflatelevel,
	)
	defVar(
		ds,
		"n_iterations",
		Int,
		("parcel_ix", "time"),
		# chunksizes=parcel_chunk_size,
		fillvalue=-1,
		deflatelevel=deflatelevel,
	)
	defVar(
		ds,
		"oob",
		Int,
		("parcel_ix", "time"),
		# chunksizes=parcel_chunk_size,
		fillvalue=-1,
		deflatelevel=deflatelevel,
	)

	# Scalar variables
	for scalar in tracked_scalars
		defVar(
			ds,
			scalar,
			Float64,
			("parcel_ix", "time"),
			# chunksizes=parcel_chunk_size,
			fillvalue=-999.0,
			deflatelevel=deflatelevel,
		)
	end

	attribute_var_types = Dict(
		string(fieldnames(ParcelInitialization)[i]) =>
			ParcelInitialization.types[i] for
		i in 1:length(fieldnames(ParcelInitialization)) if !(
			string(fieldnames(ParcelInitialization)[i]) in
			["parcel_ix", "propagated_vars"]
		)
	)
	println(attribute_var_types)
	println(propagated_vars_types)
	# Add in the propagated vars types
	attribute_var_types = merge(attribute_var_types, propagated_vars_types)

	for (attr_name, attr_type) in attribute_var_types
		if attr_type == DateTime
			attr_var = defVar(
				ds,
				string(attr_name),
				Int,
				("parcel_ix",),
				attrib=Dict(
					"units" => "seconds since 1970-01-01 00:00:00",
					"calendar" => "standard",
				),
				deflatelevel=deflatelevel,
			)
		else
			concrete_attr_type = attr_type == Integer ? Int64 : attr_type
			attr_var = defVar(
				ds,
				string(attr_name),
				concrete_attr_type,
				("parcel_ix",),
				# deflatelevel=deflatelevel,
			)
		end
	end

	return TrajectoryWriter(ds, parcel_ixs, tracked_scalars, max_timesteps)
end

function write_timestep!(
	writer::TrajectoryWriter,
	buffer::TrajectoryBuffer,
	time_t_index,
	index_to_coord_interpolators,
)
	# Write time
	writer.dataset["time"][time_t_index] = buffer.time_t

	# Precalculate arrays for each variable to minimize write operations
	data_to_write = Dict(
		var => fill(-999.0, length(writer.parcel_ixs)) for var in [
			POSITION_VARS;
			POSITION_IX_VARS;
			writer.tracked_scalars;
			["converged", "n_iterations", "oob"]
		]
	)
	# Populate these with data from each parcel
	for (parcel_ix_ix, parcel_ix) in enumerate(writer.parcel_ixs)
		if haskey(buffer.time_t_states, parcel_ix)
			this_parcel_state = buffer.time_t_states[parcel_ix]

			# Write positions (indices)
			data_to_write["x_ix"][parcel_ix_ix] = this_parcel_state.position_ixs.x_ix
			data_to_write["y_ix"][parcel_ix_ix] = this_parcel_state.position_ixs.y_ix
			data_to_write["z_ix"][parcel_ix_ix] = this_parcel_state.position_ixs.z_ix

			# Convert and write coordinates
			data_to_write["x"][parcel_ix_ix] =
				index_to_coord_interpolators["x"](this_parcel_state.position_ixs.x_ix)
			data_to_write["y"][parcel_ix_ix] =
				index_to_coord_interpolators["y"](this_parcel_state.position_ixs.y_ix)
			data_to_write["z"][parcel_ix_ix] =
				index_to_coord_interpolators["z"](this_parcel_state.position_ixs.z_ix)

			# Write trajectory metadata
			data_to_write["converged"][parcel_ix_ix] =
				Int(this_parcel_state.correction_result.converged)
			data_to_write["n_iterations"][parcel_ix_ix] =
				this_parcel_state.correction_result.n_iterations
			data_to_write["oob"][parcel_ix_ix] =
				Int(this_parcel_state.correction_result.oob)

			# Write scalars
			for scalar in writer.tracked_scalars
				if haskey(this_parcel_state.scalars, scalar)
					data_to_write[scalar][parcel_ix_ix] =
						this_parcel_state.scalars[scalar]
				end
			end
		end
	end

	# Now write these to the file
	for (var, this_var_data) in data_to_write
		writer.dataset[var][:, time_t_index] = this_var_data
	end
end

function write_parcel_attributes_from_initializations!(
	writer::TrajectoryWriter,
	parcel_initializations_dict::Dict{Int, ParcelInitialization},
)
	attribute_varnames = ["initialization_time", "initial_x", "initial_y", "initial_z"]
	# Get vars we're propagating through
	propagated_varnames =
		keys(first_value(parcel_initializations_dict).propagated_vars) |> collect
	# Initialize arrays for all of the variables
	data_to_write = Dict(
		k => Vector(undef, length(writer.parcel_ixs)) for
		k in [attribute_varnames; propagated_varnames]
	)

	for (parcel_ix_ix, parcel_ix) in enumerate(writer.parcel_ixs)
		for (varname, var_arr) in data_to_write
			if varname in attribute_varnames
				var_arr[parcel_ix_ix] =
					getfield(parcel_initializations_dict[parcel_ix], Symbol(varname))
			else
				# A propagated var
				var_arr[parcel_ix_ix] =
					parcel_initializations_dict[parcel_ix].propagated_vars[varname]
			end
		end
	end

	# Now actually write to disk
	for (attribute_var, attribute_arr) in data_to_write
		writer.dataset[attribute_var][:] = attribute_arr
	end
	NCDatasets.sync(writer.dataset)
end

function close_trajectory_writer!(writer::TrajectoryWriter)
	close(writer.dataset)
end

#   Main Trajectory Calculation Function
#   ===================================

function calculate_trajectories(;
	parcel_initializations::Dict{Int, ParcelInitialization},
	simulation_output_dir::Union{PosixPath, AbstractString},
	output_path::Union{PosixPath, AbstractString},
	trajectory_timestep::Period,
	trajectory_end_time::DateTime,
	back_trajectories::Bool,
	restart_ds_path::Union{PosixPath, AbstractString, Nothing}=nothing,
	tracked_scalars::Vector{String}=Vector{String}(),
	parallel::Bool=true,
	grid_stagger=0.5,
	write_interval=1,
	flush_interval=0,
	max_iterations=5,
	stop_on_early_convergence=true,
	raise_on_nonconvergence=false,
	use_subdomain_limiting=true,
	raise_on_oob=true,
	save_on_error::Bool=true,
	exist_ok::Bool=false,
	test_write=true,
	verbose=true,
	progress_bar::Bool=true,
)
	# Input validation
	if (isfile(output_path) & !exist_ok)
		error("output_path exists, but exist_ok was passed as false")
	end

	if (max_iterations < 2)
		error("max_iterations must be at least 2")
	end

	# Handle the single number vs dict case for stagger
	if grid_stagger isa Number
		grid_stagger =
			Dict("x" => grid_stagger, "y" => grid_stagger, "z" => grid_stagger)
	end

	# Rearrange the initializations so that they're in a dict by time
	initializations_by_time::Dict{DateTime, Vector{ParcelInitialization}} =
		Dict{DateTime, Vector{ParcelInitialization}}()
	for (this_parcel_ix, this_parcel_init) in parcel_initializations
		this_init_time = this_parcel_init.initialization_time
		this_time_existing_inits =
			get(initializations_by_time, this_init_time, Vector{ParcelInitialization}())
		push!(this_time_existing_inits, this_parcel_init)
		initializations_by_time[this_init_time] = this_time_existing_inits
	end

	# Get the propagated attribute types from any of the initializations
	# Get the propagated attribute types from any of the initializations
	propagated_vars_types = Dict{String, Type}()
	if !isempty(parcel_initializations)
		sample_initialization = first(values(parcel_initializations))
		for (var_name, var_value) in sample_initialization.propagated_vars
			propagated_vars_types[var_name] = typeof(var_value)
		end
	end

	# Time setup and validation
	trajectory_start_time =
		(back_trajectories ? maximum : minimum)(keys(initializations_by_time))
	last_initialization_time =
		(back_trajectories ? minimum : maximum)(keys(initializations_by_time))

	if ((back_trajectories) & (last_initialization_time < trajectory_end_time)) |
	   ((!back_trajectories) & (last_initialization_time > trajectory_end_time))
		error(
			"Argument `trajectory_end_time` must be before the earliest parcel " *
			"initialization time when running backtrajectories, or after the latest " *
			"initialization time when running forward trajectories; " *
			"last initialization time was $(last_initialization_time) and " *
			"trajectory_end_time was $(trajectory_end_time)",
		)
	end

	println("Using " * string(Threads.nthreads()) * " threads")
	println("Output path is $(output_path)")

	# Grid setup
	grid_coords = read_rams_grid(
		get_simulation_output_path(simulation_output_dir, trajectory_start_time),
	)

	# Validate grid spacing
	delta_xs = diff(grid_coords["x"])
	delta_ys = diff(grid_coords["y"])
	if !(
		(length(unique(delta_xs)) == 1) & (length(unique(delta_ys)) == 1) &
		(unique(delta_xs) == unique(delta_ys))
	)
		error("Grid spacing is not regular and identical in x and/or y")
	end
	grid_deltax = delta_xs[1]  # meters

	# Grid dimensions
	grid_nx = length(grid_coords["x"])
	grid_ny = length(grid_coords["y"])

	# Setup interpolators
	grid_ixs,
	grid_ix_nodes,
	coord_to_index_interpolators,
	index_to_coord_interpolators = setup_grid_interpolators(grid_coords, grid_stagger)

	# Time direction and trajectory times
	time_direction_factor::Float64 = back_trajectories ? -1 : 1
	trajectory_times =
		trajectory_start_time:(trajectory_timestep*time_direction_factor):trajectory_end_time |>
		collect
	# Make a separate timeseries of times at which we'll write the data
	write_times = trajectory_times[1:write_interval:end]
	# Make sure this has the last time as well
	write_times =
		trajectory_times[end] in write_times ? write_times :
		[write_times; [trajectory_times[end]]]

	# Calculate the z index of z=0
	z_ix_z0 = coord_to_index_interpolators["z"](0)

	# Initialize data containers
	# We immediately shift this when the time loop starts, so just put the relevant
	# values in the t+1 positions
	positions_buffer = TrajectoryBuffer(Dict(), trajectory_times[1], Dict(), nothing)

	# Get all parcel ixs from initialization dict
	all_parcel_ixs = unique(
		vcat(
			[
				x.parcel_ix for
				x in vcat([v for v in values(initializations_by_time)]...)
			]...,
		),
	)
	println("Using $(length(all_parcel_ixs)) parcels")

	# Initialize netcdf writer
	nc_writer = setup_trajectory_netcdf(
		string(output_path),
		all_parcel_ixs,
		tracked_scalars,
		propagated_vars_types,
		length(write_times),
	)

	# Handle restart if needed
	if !isnothing(restart_ds_path)
		restart_data =
			load_restart_data(restart_ds_path, tracked_scalars, back_trajectories)

		# Adjust trajectory times
		_ =
			back_trajectories ?
			filter!(x -> x <= minimum(restart_data["time"]), trajectory_times) :
			filter!(x -> x >= maximum(restart_data["time"]), trajectory_times)
	else
		# Write parcel attributes to disk if not doing a restart
		println("Writing parcel attributes...")
		write_parcel_attributes_from_initializations!(nc_writer, parcel_initializations)
		println("Done.")
	end

	# Setup interpolators for first timestep
	interpolators = [
		nothing,
		get_interpolators(
			get_simulation_output_path(simulation_output_dir, trajectory_times[1]),
			[wind_vars; tracked_scalars],
			grid_ix_nodes,
			grid_nx,
			grid_ny,
		),
	]

	println("Starting $(back_trajectories ? "back" : "forward") trajectories...")

	trajectory_times_generator = enumerate(trajectory_times[1:(end-1)])
	if progress_bar
		trajectory_times_generator =
			ProgressBar(trajectory_times_generator; output_stream=stdout)
	end

	try
		for (time_t_ix, time_t) in trajectory_times_generator
			flush(stdout)
			flush(stderr)
			# Shift buffer so that time_t reflects the current time_t
			shift_buffer!(positions_buffer)

			# Initialize new parcels if needed
			parcels_to_initialize =
				get(initializations_by_time, time_t, Vector{ParcelInitialization}())

			initialize_new_parcels!(
				parcels_to_initialize,
				positions_buffer.time_t_states,
				coord_to_index_interpolators,
				interpolators;
			)

			# Get next time and setup subdomain bounds
			time_t_plus_1 = trajectory_times[time_t_ix+1]
			subdomain_bounds = get_subdomain_bounds(
				positions_buffer.time_t_states,
				grid_ixs,
				use_subdomain_limiting,
				initializations_by_time,
				time_t_plus_1,
			)

			# Update interpolators
			interpolators = [
				interpolators[2],
				get_interpolators(
					get_simulation_output_path(simulation_output_dir, time_t_plus_1),
					[wind_vars; tracked_scalars],
					grid_ix_nodes,
					grid_nx,
					grid_ny;
					subdomain_bounds=subdomain_bounds,
				),
			]

			# Calculate corrected positions
			# Get the existing parcel ixs by filtering to all of them that aren't oob
			existing_parcel_ixs = filter(
				parcel_ix -> !(
					positions_buffer.time_t_states[parcel_ix].correction_result.oob
				),
				keys(positions_buffer.time_t_states),
			)

			time_t_plus_1_states = Dict{Integer, ParcelState}()

			if parallel
				parcel_ix_chunks = Iterators.partition(
					existing_parcel_ixs,
					floor(Int, length(existing_parcel_ixs) / Threads.nthreads()),
				)

				calculate_corrected_states_chunk(parcel_ix_chunk) = ParcelStateDict(
					parcel_ix => calculate_corrected_states(
						getfield(
							positions_buffer.time_t_states[parcel_ix],
							:position_ixs,
						),
						interpolators[1],
						interpolators[2],
						trajectory_timestep,
						grid_deltax,
						coord_to_index_interpolators,
						index_to_coord_interpolators,
						time_direction_factor,
						z_ix_z0,
						tracked_scalars;
						max_iterations=max_iterations,
						stop_on_early_convergence=stop_on_early_convergence,
						raise_on_oob=raise_on_oob,
						raise_on_nonconvergence=raise_on_nonconvergence,
					) for parcel_ix in parcel_ix_chunk
				)

				tasks = map(parcel_ix_chunks) do parcel_ix_chunk
					Threads.@spawn calculate_corrected_states_chunk(parcel_ix_chunk)
				end

				chunk_corrected_states = fetch.(tasks)
				time_t_plus_1_states =
					merge(time_t_plus_1_states, chunk_corrected_states...)
			else
				corrected_states = Dict()
				for parcel_ix in existing_parcel_ixs
					corrected_states[parcel_ix] = calculate_corrected_states(
						positions_buffer.time_t_states[parcel_ix].position_ixs,
						interpolators[1],
						interpolators[2],
						trajectory_timestep,
						grid_deltax,
						coord_to_index_interpolators,
						index_to_coord_interpolators,
						time_direction_factor,
						z_ix_z0,
						tracked_scalars;
						max_iterations=max_iterations,
						stop_on_early_convergence=stop_on_early_convergence,
						raise_on_oob=raise_on_oob,
						raise_on_nonconvergence=raise_on_nonconvergence,
					)
				end
				corrected_states = ParcelStateDict(corrected_states)
				time_t_plus_1_states = merge(time_t_plus_1_states, corrected_states)
			end

			# Store time t+1 positions

			# Write time t positions to disk
			# It seems counterintuitive to calculate time t+1 positions and then
			# not write them until the next timestep, but this is because we do
			# parcel initializations for time t in each step of the loop, not time
			# t+1; so we can actually modify the parcel states at time t within
			# the loop logic, hence needing to wait until calculating t+1 to write them
			if trajectory_times[time_t_ix] in write_times
				write_timestep!(
					nc_writer,
					positions_buffer,
					# Get the index of the write times that this time corresponds to
					findall(x -> x == trajectory_times[time_t_ix], write_times)[1],
					index_to_coord_interpolators,
				)
			end

			# Flush to disk if needed
			if ((flush_interval > 0) && (time_t_ix % flush_interval == 0)) ||
			   (time_t_ix == length(trajectory_times)-1) ||
			   ((time_t_ix == 1) & test_write)
				if ((time_t_ix == 1) & test_write)
					println("Testing writing to disk...")
				end

				println("Flushing to disk...")
				NCDatasets.sync(nc_writer.dataset)
				if ((time_t_ix == 1) & test_write)
					println("Test write successful.")
				end
			end

			# Prepare for next iteration
			positions_buffer.time_t_plus_1_states = time_t_plus_1_states
			positions_buffer.time_t_plus_1 = time_t_plus_1
		end

		# Write the t+1 states, now that the loop's over
		shift_buffer!(positions_buffer)
		write_timestep!(
			nc_writer,
			positions_buffer,
			length(write_times),
			index_to_coord_interpolators,
		)
		NCDatasets.sync(nc_writer.dataset)
	catch e
		# Error handling with optional save
		bt = catch_backtrace()
		showerror(stdout, e, bt)
		println("")

		if save_on_error
			println("Caught exception; saving trajectories to disk and rethrowing")
			NCDatasets.sync(nc_writer.dataset)
		end
		throw(e)
	finally
		close_trajectory_writer!(nc_writer)
	end

	return true
end