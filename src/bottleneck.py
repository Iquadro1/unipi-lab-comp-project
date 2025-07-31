import numpy as np
import src.utils as utils
from typing import Dict
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

def rescale_bottleneck_distance_matrix(distance_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rescale the bottleneck distance matrix to [0, 1]. Excluding the diagonal."""
    # Get the maximum value excluding the diagonal
    max_value = np.max(distance_matrix[np.triu_indices_from(distance_matrix, k=1)])
    min_value = np.min(distance_matrix[np.triu_indices_from(distance_matrix, k=1)])

    if max_value == min_value:
        raise ValueError("All distances are equal, cannot rescale.")

    # Rescale the matrix
    rescaled_matrix = (distance_matrix - min_value) / (max_value - min_value)

    return rescaled_matrix

def apply_infinity_correction_for_dimension(complex_result: utils.ComplexResult, dimension: int, replacement_value: float) ->  NDArray[np.float64]:
    """Apply infinity correction and compute corrected distances for a specific dimension"""
    
    # Create corrected intervals for this dimension
    intervals_copy = copy.deepcopy(complex_result.intervals[dimension])
    for interval in intervals_copy:
        if interval[1] == float('inf'):
            interval[1] = replacement_value
    
    return intervals_copy

def compute_bottleneck_distances_from_intervals(intervals: Dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
    distance_matrix = np.zeros((len(intervals), len(intervals)))
    # Compute corrected distances
    names = list(intervals.keys())
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names[:i]):
            intervals1 = intervals[name1]
            intervals2 = intervals[name2]
            distance = gd.bottleneck_distance(intervals1, intervals2)
            distance_matrix[j, i] = distance
            distance_matrix[i, j] = distance
    
    return distance_matrix


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

def compute_bottleneck_distances(complex_results: Dict[str, utils.ComplexResult]) -> NDArray[np.float64]:
    """Compute all pairwise bottleneck distances and return distance matrices"""
    
    max_dimension = max(complex_result.get_max_dimension() for complex_result in complex_results.values()) + 1
    
    # Get complex names for labeling
    complex_names = list(complex_results.keys())
    n_complexes = len(complex_names)

    # Initialize distance tensor: [max_dimension, n_complexes, n_complexes]
    distance_matrices = np.zeros((max_dimension, n_complexes, n_complexes))

    # print("\n--- BOTTLENECK DISTANCE COMPUTATION ---")

    # Compute distances for each dimension
    for dim in range(max_dimension):
        print(f"\nComputing dimension {dim}...")
        distance_matrices[dim] = compute_bottleneck_distances_from_intervals(
            {name: complex_results[name].intervals[dim] for name in complex_names}
        )
        print_matrix_as_table(distance_matrices[dim], complex_names)
        # if distance_matrices[dim] has an infinity entry, apply correction to all complex results
        if np.any(np.isinf(distance_matrices[dim])):
            all_intervals = [
                interval[1] for complex_result in complex_results.values()
                for interval in complex_result.intervals[dim]
                if interval[1] != float('inf')
            ]
            replacement_value = 1.5 * max(all_intervals) if all_intervals else 1.0
            print(f"\nUsing replacement value for infinity intervals in dimension {dim}: {replacement_value}")
            corrected_intervals = {}
            for name in complex_names:
                corrected_intervals[name] = apply_infinity_correction_for_dimension(
                    complex_results[name], dim, replacement_value
                )
            distance_matrices[dim] = compute_bottleneck_distances_from_intervals(corrected_intervals)

    return distance_matrices

def visualize_bottleneck_distances(mds_results: Dict[int, NDArray[np.float64]], names: list[str]) -> None:
    max_dimension = max(mds_results.keys()) + 1
    _, axes = plt.subplots(1, max_dimension, figsize=(5 * max_dimension, 5))
    if max_dimension == 1:
        axes = [axes]  # Ensure axes is iterable for single subplot
    
    dimension_names = ['Connected Components', 'Loops/Cycles', 'Voids/Cavities']
    n_complexes = mds_results[0].shape[0]

    for dim in range(max_dimension):
        ax = axes[dim]
        
        # Plot points
        ax.scatter(mds_results[dim][:, 0], mds_results[dim][:, 1], 
                           c=range(n_complexes), cmap='cool')
        
        # Add labels
        for i, name in enumerate(names):
            ax.annotate(name.replace('_', '\n'), 
                       (mds_results[dim][i, 0], mds_results[dim][i, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, ha='left')
        
        ax.set_title(f'Dimension {dim} ({dimension_names[dim]})')
        # ax.grid(True, alpha=0.3)
    
    plt.suptitle('Complex Similarity Based on Bottleneck Distances\n(MDS Embedding)', 
                fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()