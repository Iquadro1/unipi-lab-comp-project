import numpy as np
import src.utils1 as utils
from typing import Dict, List
import gudhi as gd
import copy
from sklearn import manifold
import matplotlib.pyplot as plt
from numpy.typing import NDArray

def print_matrix_as_table(matrix: NDArray[np.float64], names: list[str]) -> None:

    # Create header
    header = f"\n{'Complex':<20}"
    for name in names:
        header += f"{name:<18}"
    print(header)
    print("-" * len(header))
    
    # Print matrix rows
    for i, name1 in enumerate(names):
        row = f"{name1:<20}"
        for j in range(len(names)):
            if i == j:
                row += f"{'0.000':<18}"
            elif i > j:
                row += f"{matrix[i, j]:<18.8f}"
            else:
                row += f"{matrix[j, i]:<18.8f}"
        print(row)

# def rescale_bottleneck_distance_matrix(distance_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
#     """Rescale the bottleneck distance matrix to [0, 1]. Excluding the diagonal."""
#     # Get the maximum value excluding the diagonal
#     max_value = np.max(distance_matrix[np.triu_indices_from(distance_matrix, k=1)])
#     min_value = np.min(distance_matrix[np.triu_indices_from(distance_matrix, k=1)])

#     if max_value == min_value:
#         raise ValueError("All distances are equal, cannot rescale.")

#     # Rescale the matrix
#     rescaled_matrix = (distance_matrix - min_value) / (max_value - min_value)

#     return rescaled_matrix

# def apply_infinity_correction_for_dimension(complex_result: utils.ComplexResult, dimension: int, replacement_value: float) ->  NDArray[np.float64]:
#     """Apply infinity correction and compute corrected distances for a specific dimension"""
    
#     # Create corrected intervals for this dimension
#     intervals_copy = copy.deepcopy(complex_result.intervals[dimension])
#     for interval in intervals_copy:
#         if interval[1] == float('inf'):
#             interval[1] = replacement_value
    
#     return intervals_copy

def correct_infinity_intervals(complex_results: Dict[str, utils.ComplexResult]) -> Dict[str, Dict[int, NDArray[np.float64]]]:
    """
    Corrects persistence intervals by replacing 'inf' death times with the
    complex-specific max_filtration_value. This ensures scales are not mixed.
    """
    corrected_results = {}
    max_dimension = max(res.get_max_dimension() for res in complex_results.values()) + 1

    for name, result in complex_results.items():
        corrected_intervals_for_complex = {}
        replacement_value = result.max_filtration_value
        
        # Apply a 1.5 factor for better visualization and to handle cases where
        # the max_filtration_value itself is a death time.
        replacement_value *= 1.5

        for dim in range(max_dimension):
            # Ensure we handle dimensions that might not exist for a complex
            if dim not in result.intervals:
                corrected_intervals_for_complex[dim] = np.array([])
                continue

            intervals_copy = copy.deepcopy(result.intervals[dim])
            for interval in intervals_copy:
                if interval[1] == float('inf'):
                    interval[1] = replacement_value
            corrected_intervals_for_complex[dim] = intervals_copy
        corrected_results[name] = corrected_intervals_for_complex
        
    return corrected_results

def compute_distance_matrices(intervals: Dict[str, Dict[int, NDArray[np.float64]]]) -> NDArray[np.float64]:
    """
    Computes all pairwise bottleneck distances from pre-corrected intervals.
    Returns a tensor of distance matrices, one for each dimension.
    """
    complex_names = list(intervals.keys())
    n_complexes = len(complex_names)
    
    # Determine max dimension from the corrected intervals dictionary
    max_dimension = 0
    if n_complexes > 0:
        max_dimension = max(len(d) for d in intervals.values())

    distance_matrices = np.zeros((max_dimension, n_complexes, n_complexes))

    for dim in range(max_dimension):
        distance_matrix_dim = np.zeros((n_complexes, n_complexes))
        for i, name1 in enumerate(complex_names):
            for j, name2 in enumerate(complex_names[:i]):
                intervals1 = intervals[name1][dim]
                intervals2 = intervals[name2][dim]
                distance = gd.bottleneck_distance(intervals1, intervals2)
                distance_matrix_dim[i, j] = distance
                distance_matrix_dim[j, i] = distance
        distance_matrices[dim] = distance_matrix_dim
        
    return distance_matrices

# def compute_bottleneck_distances_from_intervals(intervals: Dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
#     distance_matrix = np.zeros((len(intervals), len(intervals)))
#     # Compute corrected distances
#     names = list(intervals.keys())
#     for i, name1 in enumerate(names):
#         for j, name2 in enumerate(names[:i]):
#             intervals1 = intervals[name1]
#             intervals2 = intervals[name2]
#             distance = gd.bottleneck_distance(intervals1, intervals2)
#             distance_matrix[j, i] = distance
#             distance_matrix[i, j] = distance
    
#     return distance_matrix


def compute_mds(distance_matrices: NDArray[np.float64]) -> Dict[int, NDArray[np.float64]]:
    """Compute MDS embedding for a specific dimension's distance matrix"""
    mds = manifold.MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        dissimilarity="precomputed",
        n_jobs=1,
        n_init=1
    )
    
    mds_results = {}
    for dim in range(distance_matrices.shape[0]):
        # Check if the distance matrix is exactly all zeros (no tolerance)
        # Get upper triangular part excluding diagonal for comparison
        upper_tri = distance_matrices[dim][np.triu_indices_from(distance_matrices[dim], k=1)]
        
        if np.all(upper_tri == 0):
            # If all distances are exactly zero, create superimposed points at origin
            n_points = distance_matrices[dim].shape[0]
            mds_results[dim] = np.zeros((n_points, 2))
            print(f"Warning: All distances are exactly zero for dimension {dim}. Using superimposed points at origin.")
        else:
            mds_results[dim] = mds.fit_transform(distance_matrices[dim])

    return mds_results

# def compute_bottleneck_distances(complex_results: Dict[str, utils.ComplexResult]) -> NDArray[np.float64]:
#     """Compute all pairwise bottleneck distances and return distance matrices"""
    
#     max_dimension = max(complex_result.get_max_dimension() for complex_result in complex_results.values()) + 1
    
#     # Get complex names for labeling
#     complex_names = list(complex_results.keys())
#     n_complexes = len(complex_names)

#     # Initialize distance tensor: [max_dimension, n_complexes, n_complexes]
#     distance_matrices = np.zeros((max_dimension, n_complexes, n_complexes))

#     # print("\n--- BOTTLENECK DISTANCE COMPUTATION ---")

#     # Compute distances for each dimension
#     for dim in range(max_dimension):
#         print(f"\nComputing dimension {dim}...")
#         distance_matrices[dim] = compute_bottleneck_distances_from_intervals(
#             {name: complex_results[name].intervals[dim] for name in complex_names}
#         )
#         print_matrix_as_table(distance_matrices[dim], complex_names)
#         # if distance_matrices[dim] has an infinity entry, apply correction to all complex results
#         if np.any(np.isinf(distance_matrices[dim])):
#             all_intervals = [
#                 interval[1] for complex_result in complex_results.values()
#                 for interval in complex_result.intervals[dim]
#                 if interval[1] != float('inf')
#             ]
#             replacement_value = 1.5 * max(all_intervals) if all_intervals else 1.0
#             print(f"\nUsing replacement value for infinity intervals in dimension {dim}: {replacement_value}")
#             corrected_intervals = {}
#             for name in complex_names:
#                 corrected_intervals[name] = apply_infinity_correction_for_dimension(
#                     complex_results[name], dim, replacement_value
#                 )
#             distance_matrices[dim] = compute_bottleneck_distances_from_intervals(corrected_intervals)

#     return distance_matrices

def visualize_bottleneck_distances(mds_results: Dict[int, NDArray[np.float64]], names: list[str]) -> None:
    max_dimension = max(mds_results.keys()) + 1
    fig, axes = plt.subplots(1, max_dimension, figsize=(5 * max_dimension, 5))
    if max_dimension == 1:
        axes = [axes]  # Ensure axes is iterable for single subplot
    
    dimension_names = ['Connected Components', 'Loops/Cycles', 'Voids/Cavities']
    n_complexes = mds_results[0].shape[0]

    # Create a color array for consistent coloring across subplots
    colors = plt.cm.viridis_r(np.linspace(0, 1, n_complexes))

    for dim in range(max_dimension):
        ax = axes[dim]
        
        # Plot points with consistent colors
        ax.scatter(mds_results[dim][:, 0], mds_results[dim][:, 1], 
                           c=colors, s=60)
        
        ax.set_title(f'Dimension {dim} ({dimension_names[dim]})')
    
    # Create a common legend for all subplots
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=colors[i], markersize=8, label=name)
                      for i, name in enumerate(names)]
    
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
               ncol=min(len(names), 4), fontsize=10)
    
    plt.suptitle('Complex Similarity Based on Bottleneck Distances\n(MDS Embedding)', 
                fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    plt.show()

def analyze_bottleneck_distances(complex_results: Dict[str, utils.ComplexResult]):
    """
    A full pipeline that computes and shows bottleneck distances before and after
    infinity correction, then visualizes the corrected results with MDS.
    """
    complex_names = list(complex_results.keys())
    max_dimension = max(res.get_max_dimension() for res in complex_results.values()) + 1

    # --- Step 1: Analyze WITHOUT infinity correction ---
    print("\n" + "="*50)
    print("ANALYSIS BEFORE INFINITY CORRECTION")
    print("="*50)
    
    # Create a dictionary of the original interval sets
    original_intervals = {
        name: {dim: res.intervals.get(dim, np.array([])) for dim in range(max_dimension)}
        for name, res in complex_results.items()
    }
    
    print("\n--- COMPUTING ORIGINAL BOTTLENECK DISTANCE MATRICES ---")
    print("(Note: Distances may be 'inf' if diagrams have unmatched infinite points)")
    original_distance_matrices = compute_distance_matrices(original_intervals)
    
    for dim in range(original_distance_matrices.shape[0]):
        print(f"\nDimension {dim} Original Distance Matrix:")
        print_matrix_as_table(original_distance_matrices[dim], complex_names)

    # --- Step 2: Analyze WITH infinity correction ---
    print("\n" + "="*50)
    print("ANALYSIS AFTER INFINITY CORRECTION")
    print("="*50)
    
    print("\n--- CORRECTING INFINITY INTERVALS ---")
    corrected_intervals = correct_infinity_intervals(complex_results)

    print("\n--- COMPUTING CORRECTED BOTTLENECK DISTANCE MATRICES ---")
    corrected_distance_matrices = compute_distance_matrices(corrected_intervals)
    
    for dim in range(corrected_distance_matrices.shape[0]):
        print(f"\nDimension {dim} Corrected Distance Matrix:")
        print_matrix_as_table(corrected_distance_matrices[dim], complex_names)

    # --- Step 3: MDS and Visualization (on corrected data) ---
    print("\n--- COMPUTING MDS EMBEDDING (ON CORRECTED DATA) ---")
    # We only compute MDS on the finite, corrected distances as it's a metric embedding.
    mds_results = compute_mds(corrected_distance_matrices)

    print("\n--- VISUALIZING MDS ---")
    visualize_bottleneck_distances(mds_results, complex_names)

# def compute_convergence_sequences1(complex_res: Dict[str, Dict[str, utils.ComplexResult]]) -> Dict[str, Dict[int, List[float]]]:
#     """
#     Computes the sequence of bottleneck distances between consecutive samples for each complex type.
#     Infinity values in persistence diagrams are handled by replacing them with a scaled
#     version of the complex's maximum filtration value.

#     Args:
#         complex_res: A dictionary where keys are ordered sample names and values are
#                      dictionaries of complex results for that sample.

#     Returns:
#         A dictionary mapping complex type to another dictionary, which maps dimension
#         to a list of bottleneck distances between consecutive samples.
#     """
#     sample_names = list(complex_res.keys())
#     if len(sample_names) < 2:
#         print("Need at least two samples to compute convergence.")
#         return {}

#     complex_types = list(complex_res[sample_names[0]].keys())
#     # Determine the maximum dimension across all complexes
#     max_dimension = 0
#     for res_dict in complex_res.values():
#         for res in res_dict.values():
#             max_dimension = max(max_dimension, res.get_max_dimension())
#     max_dimension += 1 # To make it a range upper bound

#     convergence_data = {
#         complex_type: {dim: [] for dim in range(max_dimension)}
#         for complex_type in complex_types
#     }

#     print("Computing bottleneck distance convergence sequences...")

#     for i in range(len(sample_names) - 1):
#         sample1_name = sample_names[i]
#         sample2_name = sample_names[i+1]
#         print(f"\nProcessing pair: {sample1_name} -> {sample2_name}")

#         for complex_type in complex_types:
#             res1 = complex_res[sample1_name][complex_type]
#             res2 = complex_res[sample2_name][complex_type]

#             for dim in range(max_dimension):
#                 intervals1_orig = res1.intervals.get(dim, np.array([]))
#                 intervals2_orig = res2.intervals.get(dim, np.array([]))
                
#                 # Handle infinity correction for each diagram independently
#                 intervals1_corr = copy.deepcopy(intervals1_orig)
#                 replacement1 = res1.max_filtration_value * 1.5
#                 for interval in intervals1_corr:
#                     if interval[1] == float('inf'):
#                         interval[1] = replacement1
                
#                 intervals2_corr = copy.deepcopy(intervals2_orig)
#                 replacement2 = res2.max_filtration_value * 1.5
#                 for interval in intervals2_corr:
#                     if interval[1] == float('inf'):
#                         interval[1] = replacement2

#                 # Compute bottleneck distance and store it
#                 distance = gd.bottleneck_distance(intervals1_corr, intervals2_corr)
#                 convergence_data[complex_type][dim].append(distance)
    
#     print("\nConvergence sequence computation complete.")
#     return convergence_data

# def compute_convergence_sequences1(complex_res: Dict[str, Dict[str, utils.ComplexResult]]) -> Dict[str, Dict[int, List[float]]]:
#     """
#     Computes bottleneck distances between consecutive samples using compute_distance_matrices.
#     Much more efficient as it leverages existing optimized functions.
#     """
#     sample_names = list(complex_res.keys())
#     if len(sample_names) < 2:
#         print("Need at least two samples to compute convergence.")
#         return {}

#     complex_types = list(complex_res[sample_names[0]].keys())
#     max_dimension = max(
#         res.get_max_dimension() 
#         for res_dict in complex_res.values() 
#         for res in res_dict.values()
#     ) + 1

#     convergence_data = {
#         complex_type: {dim: [] for dim in range(max_dimension)}
#         for complex_type in complex_types
#     }

#     print("Computing bottleneck distance convergence sequences...")

#     for complex_type in complex_types:
#         print(f"\nProcessing complex type: {complex_type}")
        
#         # Extract all samples for this complex type
#         complex_results_for_type = {
#             sample_name: complex_res[sample_name][complex_type]
#             for sample_name in sample_names
#         }
        
#         # Use existing pipeline: correct infinities then compute all pairwise distances
#         corrected_intervals = correct_infinity_intervals(complex_results_for_type)
#         distance_matrices = compute_distance_matrices(corrected_intervals)
        
#         # Extract consecutive distances from the full distance matrix
#         for dim in range(max_dimension):
#             consecutive_distances = []
#             for i in range(len(sample_names) - 1):
#                 # Extract distance between consecutive samples i and i+1
#                 distance = distance_matrices[dim][i, i+1]
#                 consecutive_distances.append(distance)
            
#             convergence_data[complex_type][dim] = consecutive_distances

#     print("\nConvergence sequence computation complete.")
#     return convergence_data

def compute_convergence_sequences(complex_res: Dict[str, Dict[str, utils.ComplexResult]]) -> Dict[str, Dict[int, List[float]]]:
    """
    Computes bottleneck distances between consecutive samples efficiently.
    Uses existing correction pipeline but only computes needed distances.
    """
    sample_names = list(complex_res.keys())
    if len(sample_names) < 2:
        print("Need at least two samples to compute convergence.")
        return {}

    complex_types = list(complex_res[sample_names[0]].keys())
    max_dimension = max(
        res.get_max_dimension() 
        for res_dict in complex_res.values() 
        for res in res_dict.values()
    ) + 1

    convergence_data = {
        complex_type: {dim: [] for dim in range(max_dimension)}
        for complex_type in complex_types
    }

    print("Computing bottleneck distance convergence sequences...")

    for complex_type in complex_types:
        print(f"\nProcessing complex type: {complex_type}")
        
        # Extract all samples for this complex type
        complex_results_for_type = {
            sample_name: complex_res[sample_name][complex_type]
            for sample_name in sample_names
        }
        
        # Use existing correction pipeline (this is the reusable part)
        corrected_intervals = correct_infinity_intervals(complex_results_for_type)
        
        # But only compute consecutive distances (not the full matrix)
        for dim in range(max_dimension):
            consecutive_distances = []
            for i in range(len(sample_names) - 1):
                sample1_name = sample_names[i]
                sample2_name = sample_names[i+1]
                
                intervals1 = corrected_intervals[sample1_name][dim]
                intervals2 = corrected_intervals[sample2_name][dim]
                
                distance = gd.bottleneck_distance(intervals1, intervals2)
                consecutive_distances.append(distance)
            
            convergence_data[complex_type][dim] = consecutive_distances

    print("\nConvergence sequence computation complete.")
    return convergence_data

def plot_convergence_sequences(
    convergence_data: Dict[str, Dict[int, List[float]]],
    sample_names: List[str],
    log_scale: bool = False
):
    """
    Plots the convergence sequences of bottleneck distances, creating one subplot per dimension.

    Args:
        convergence_data: The data computed by compute_convergence_sequences.
        sample_names: The ordered list of sample names for labeling the x-axis.
        log_scale: Whether to use a logarithmic scale for the y-axis.
    """
    if not convergence_data:
        print("No convergence data to plot.")
        return

    complex_types = list(convergence_data.keys())
    max_dimension = max(len(d) for d in convergence_data.values())

    x_labels = [
        f"{s1.split()[-1]}->{s2.split()[-1]}" 
        for s1, s2 in zip(sample_names[:-1], sample_names[1:])
    ]

    # Create a color array for consistent coloring with visualize_bottleneck_distances
    colors = plt.cm.viridis_r(np.linspace(0, 1, len(complex_types)))

    fig, axes = plt.subplots(1, max_dimension, figsize=(7 * max_dimension, 6), sharey=log_scale)
    if max_dimension == 1:
        axes = [axes]

    for dim in range(max_dimension):
        ax = axes[dim]
        for i, complex_type in enumerate(complex_types):
            distances = convergence_data[complex_type].get(dim, [])
            if len(distances) == len(x_labels):
                # Handle zero values for log scale
                if log_scale:
                    # Add small epsilon to avoid log(0)
                    distances = [max(d, 1e-10) for d in distances]
                
                ax.plot(x_labels, distances, marker='o', linestyle='-', 
                       color=colors[i], label=complex_type.replace('_', ' ').title())

        ax.set_title(f"Dimension {dim} Convergence ")
        ax.set_xlabel("Consecutive Sample Pairs")
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        
        if log_scale:
            ax.set_yscale('log')
            ax.set_ylabel("Bottleneck Distance (Log Scale)")
        else:
            ax.set_ylabel("Bottleneck Distance")

    # Create a shared legend above the plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=min(len(complex_types), 4))
    
    fig.suptitle("Bottleneck Distance Convergence Between Consecutive Samples", fontsize=16, y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.9]) # Adjust layout to make room for legend
    plt.show()