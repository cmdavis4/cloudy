"""
Benchmark to test the performance impact of individual_meshes argument.

This benchmark tests whether creating separate meshes for each isosurface
(individual_meshes=True) has a performance impact compared to creating a
single mesh with multiple isosurfaces (individual_meshes=False), and whether
this impact varies with the number of isosurfaces.
"""

import numpy as np
import xarray as xr
import time
import pytest
from pathlib import Path
import pyvista as pv
import datetime as dt

from common_experimental.pvplotting.types_pvplotting import (
    PVConfig,
    PVContourSpec,
    PVRamsData,
)
from common_experimental.pvplotting.core_pvplotting import (
    plot_rams_and_trajectories,
)


def create_sample_rams_ds(nx=50, ny=50, nz=30):
    """Create a sample RAMS dataset for benchmarking."""
    x = np.linspace(0, 10000, nx)
    y = np.linspace(0, 10000, ny)
    z = np.linspace(0, 5000, nz)
    time = np.array([dt.datetime(2020, 1, 1, 0, 0, 0)], dtype='datetime64[ns]')  # Single time point

    # Create a realistic 3D temperature field with some variation
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Create temperature field with some interesting features
    temp = 280 + 20 * np.sin(X / 2000) * np.cos(Y / 2000) + 10 * np.exp(-Z / 2000)
    # Add some noise
    temp += np.random.randn(*temp.shape) * 0.5

    # Add time dimension
    temp = temp[..., np.newaxis]

    ds = xr.Dataset(
        {"temperature": (["x", "y", "z", "time"], temp)},
        coords={"x": x, "y": y, "z": z, "time": time},
    )
    return ds


def benchmark_isosurface_creation(
    n_isosurfaces,
    individual_meshes,
    grid_size=(50, 50, 30),
    n_runs=5
):
    """
    Benchmark isosurface creation with given parameters.

    Args:
        n_isosurfaces: Number of isosurface values to create
        individual_meshes: Whether to create individual meshes
        grid_size: Tuple of (nx, ny, nz) for grid dimensions
        n_runs: Number of benchmark runs to average

    Returns:
        dict: Timing results
    """
    # Create dataset once
    ds = create_sample_rams_ds(*grid_size)

    # Generate isosurface values based on data range
    temp_min = ds.temperature.min().values
    temp_max = ds.temperature.max().values
    isosurfaces = np.linspace(temp_min + 5, temp_max - 5, n_isosurfaces).tolist()

    times = []
    mesh_counts = []

    for _ in range(n_runs):
        # Set PyVista to off-screen rendering
        pv.OFF_SCREEN = True

        # Create contour spec
        contour_spec = PVContourSpec(
            varname="temperature",
            isosurfaces=isosurfaces,
            individual_meshes=individual_meshes,
            scalar_bar=False,
        )

        # Create RAMS data
        rams_data = PVRamsData(
            simulation_ds=ds,
            varspecs=(contour_spec,)
        )

        # Create plotter
        plotter = pv.Plotter(off_screen=True)
        pv_config = PVConfig(
            plotter=plotter,
            show=False,
        )

        # Time the mesh creation and rendering
        start_time = time.perf_counter()

        meshes = plot_rams_and_trajectories(
            pv_config=pv_config,
            pv_datas=[rams_data]
        )

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        times.append(elapsed)

        # Count meshes created
        # meshes is a dict with time keys, each containing a dict of meshes
        mesh_count = sum(len(mesh_dict) for mesh_dict in meshes.values())
        mesh_counts.append(mesh_count)

        # Clean up
        plotter.close()

    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'all_times': times,
        'n_meshes': mesh_counts[0],  # Should be same for all runs
        'n_isosurfaces': n_isosurfaces,
        'individual_meshes': individual_meshes,
        'grid_size': grid_size,
    }


@pytest.mark.benchmark
class TestIndividualMeshesPerformance:
    """Test performance of individual_meshes parameter."""

    @pytest.mark.parametrize("n_isosurfaces", [1, 3, 5, 10])
    @pytest.mark.parametrize("individual_meshes", [False, True])
    def test_benchmark_varying_isosurfaces(self, n_isosurfaces, individual_meshes):
        """Benchmark with varying number of isosurfaces."""
        result = benchmark_isosurface_creation(
            n_isosurfaces=n_isosurfaces,
            individual_meshes=individual_meshes,
            grid_size=(30, 30, 20),  # Smaller grid for faster tests
            n_runs=3
        )

        # Print results for comparison
        mode = "individual" if individual_meshes else "combined"
        print(f"\n{mode} meshes, {n_isosurfaces} isosurfaces:")
        print(f"  Mean time: {result['mean_time']:.4f}s ± {result['std_time']:.4f}s")
        print(f"  Meshes created: {result['n_meshes']}")

        # Assert that the operation completed
        assert result['mean_time'] > 0
        assert result['n_meshes'] > 0

    def test_benchmark_comparison_detailed(self):
        """
        Detailed benchmark comparing individual_meshes=True vs False
        across different numbers of isosurfaces.
        """
        isosurface_counts = [1, 2, 5, 10]
        results = []

        print("\n" + "="*80)
        print("DETAILED PERFORMANCE COMPARISON")
        print("="*80)

        for n_iso in isosurface_counts:
            # Test with individual_meshes=False
            result_combined = benchmark_isosurface_creation(
                n_isosurfaces=n_iso,
                individual_meshes=False,
                grid_size=(40, 40, 25),
                n_runs=5
            )

            # Test with individual_meshes=True
            result_individual = benchmark_isosurface_creation(
                n_isosurfaces=n_iso,
                individual_meshes=True,
                grid_size=(40, 40, 25),
                n_runs=5
            )

            # Calculate speedup/slowdown
            speedup = result_individual['mean_time'] / result_combined['mean_time']

            results.append({
                'n_isosurfaces': n_iso,
                'combined': result_combined,
                'individual': result_individual,
                'speedup': speedup
            })

            print(f"\nIsosurfaces: {n_iso}")
            print(f"  Combined meshes:    {result_combined['mean_time']:.4f}s ± {result_combined['std_time']:.4f}s")
            print(f"  Individual meshes:  {result_individual['mean_time']:.4f}s ± {result_individual['std_time']:.4f}s")
            print(f"  Ratio (individual/combined): {speedup:.2f}x")
            print(f"  Meshes created (combined): {result_combined['n_meshes']}")
            print(f"  Meshes created (individual): {result_individual['n_meshes']}")

        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"{'N_Isosurfaces':<15} {'Combined (s)':<15} {'Individual (s)':<15} {'Ratio':<10}")
        print("-"*80)
        for r in results:
            print(
                f"{r['n_isosurfaces']:<15} "
                f"{r['combined']['mean_time']:<15.4f} "
                f"{r['individual']['mean_time']:<15.4f} "
                f"{r['speedup']:<10.2f}"
            )

        # Assert all benchmarks completed
        assert len(results) == len(isosurface_counts)

        return results


def run_standalone_benchmark():
    """
    Run benchmark standalone (not through pytest) for easy execution.
    """
    print("Running standalone benchmark for individual_meshes performance...")
    print("This will test the impact of individual_meshes on performance")
    print("with varying numbers of isosurfaces.\n")

    test = TestIndividualMeshesPerformance()
    results = test.test_benchmark_comparison_detailed()

    # Additional analysis
    print("\nANALYSIS:")
    print("-" * 80)

    # Check if there's a trend with number of isosurfaces
    speedups = [r['speedup'] for r in results]
    n_isos = [r['n_isosurfaces'] for r in results]

    # Simple linear fit to see if overhead increases with isosurfaces
    if len(speedups) > 1:
        correlation = np.corrcoef(n_isos, speedups)[0, 1]
        print(f"Correlation between n_isosurfaces and slowdown: {correlation:.3f}")

        if abs(correlation) > 0.7:
            if correlation > 0:
                print("→ Performance impact INCREASES with more isosurfaces")
            else:
                print("→ Performance impact DECREASES with more isosurfaces")
        else:
            print("→ Performance impact does NOT strongly vary with number of isosurfaces")

    # Overall recommendation
    avg_speedup = np.mean(speedups)
    print(f"\nAverage ratio (individual/combined): {avg_speedup:.2f}x")

    if avg_speedup > 1.2:
        print("→ RECOMMENDATION: individual_meshes=True has significant overhead")
        print("  Use only when you need separate control of each isosurface")
    elif avg_speedup < 0.8:
        print("→ FINDING: individual_meshes=True is actually FASTER")
        print("  This is unexpected and may warrant further investigation")
    else:
        print("→ RECOMMENDATION: Performance difference is minimal (<20%)")
        print("  Choose based on your use case requirements")

    return results


def plot_benchmark_results(results, save_path=None):
    """
    Create a visualization of benchmark results.

    Args:
        results: List of result dicts from benchmark
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt

    n_isos = [r['n_isosurfaces'] for r in results]
    combined_times = [r['combined']['mean_time'] for r in results]
    individual_times = [r['individual']['mean_time'] for r in results]
    combined_std = [r['combined']['std_time'] for r in results]
    individual_std = [r['individual']['std_time'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Absolute times
    ax1.errorbar(n_isos, combined_times, yerr=combined_std,
                 marker='o', label='Combined (individual_meshes=False)',
                 linewidth=2, capsize=5)
    ax1.errorbar(n_isos, individual_times, yerr=individual_std,
                 marker='s', label='Individual (individual_meshes=True)',
                 linewidth=2, capsize=5)
    ax1.set_xlabel('Number of Isosurfaces', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Isosurface Creation Performance', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(n_isos)

    # Plot 2: Ratio (slowdown)
    speedups = [r['speedup'] for r in results]
    colors = ['green' if s < 1 else 'orange' if s < 1.5 else 'red' for s in speedups]
    ax2.bar(n_isos, speedups, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='No difference')
    ax2.set_xlabel('Number of Isosurfaces', fontsize=12)
    ax2.set_ylabel('Ratio (Individual / Combined)', fontsize=12)
    ax2.set_title('Performance Overhead of individual_meshes=True', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(n_isos)

    # Add value labels on bars
    for i, (x, y) in enumerate(zip(n_isos, speedups)):
        label = f'{y:.2f}x'
        ax2.text(x, y + 0.05, label, ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    return fig


if __name__ == "__main__":
    results = run_standalone_benchmark()

    # Create visualization
    try:
        fig = plot_benchmark_results(results, save_path='tests/benchmark_individual_meshes_results.png')
        print("\nVisualization created successfully!")
    except Exception as e:
        print(f"\nCould not create visualization: {e}")
