include("/home/cmdavis4/projects/common/trajectories.jl");
using Dates;
calculate_trajectories(
	parcel_initializations=read_parcel_initial_states({{parcel_initialization_csv_path}},),
	simulation_output_dir={{simulation_output_dir}},
	output_path={{output_path}},
	trajectory_timestep=Second({{trajectory_timestep}}),
	trajectory_end_time=DateTime({{trajectory_end_time}}, RAMS_DATEFORMAT),
	back_trajectories={{back_trajectories}},
	restart_ds_path={{restart_ds_path}},
	tracked_scalars={{tracked_scalars}},
	parallel={{parallel}},
	grid_stagger={{grid_stagger}},
	write_interval={{write_interval}},
	max_iterations={{max_iterations}},
	stop_on_early_convergence={{stop_on_early_convergence}},
	raise_on_nonconvergence={{raise_on_nonconvergence}},
	use_subdomain_limiting={{use_subdomain_limiting}},
	raise_on_oob={{raise_on_oob}},
	save_on_error={{save_on_error}},
	exist_ok={{exist_ok}},
	test_write={{test_write}},
	verbose={{verbose}},
	progress_bar={{progress_bar}},
)

