import numpy as np
from dataclasses import dataclass
from typing import Dict
from numpy.typing import NDArray
import gudhi as gd
import time
import psutil
import os

@dataclass
class ComplexResult:
    """Store results for a simplicial complex"""
    stree: gd.SimplexTree
    name: str
    num_simplices: int
    persistence: list[tuple[int, tuple[np.float64, np.float64]]]
    intervals: Dict[int, NDArray[np.float64]]
    points: NDArray[np.float64]
    # Performance metrics
    processing_time_s: float
    memory_usage_mb: float
    
    def get_max_dimension(self) -> int:
        """Get the maximum dimension of the complex"""
        return max(self.intervals.keys()) if self.intervals else 0

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def format_memory(mb):
    """Format memory size for display"""
    if mb < 1024:
        return f"{mb:.3f} MB"
    else:
        return f"{mb/1024:.3f} GB"

from scipy.spatial.distance import cdist

def calculate_distance(points: NDArray[np.float64]) -> tuple[NDArray[np.float64], float, float, float]:
    
    # Calculate pairwise distances between all points
    print("Calculating pairwise distances...")
    distances = cdist(points, points, metric='euclidean')
    
    # Set diagonal to infinity to exclude self-distances (distance from point to itself)
    np.fill_diagonal(distances, np.inf)
    
    # Find minimum distance for each point to all other points
    min_distances = np.min(distances, axis=1)
    
    # Calculate mean of minimum distances
    mean_min_distance = np.mean(min_distances)

    max_min_distance = np.max(min_distances)

    median_min_distance = np.median(min_distances)
    
    # print(f"Mean of minimum distances: {mean_min_distance:.6f}")
    # print(f"Standard deviation: {np.std(min_distances):.6f}")
    # print(f"Min value: {np.min(min_distances):.6f}")
    # print(f"Max value: {max_min_distance:.6f}")
    # print(f"Median value: {median_min_distance:.6f}")

    return min_distances, max_min_distance, mean_min_distance, median_min_distance

from sklearn.neighbors import NearestNeighbors
def estimate_max_edge_length_knn(point_cloud: NDArray[np.float64], k: int = 6, scale_factor: float = 1.0) -> float:
    """Estimate max edge length based on k-nearest neighbor distances"""
    
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(point_cloud)  # +1 because first neighbor is the point itself
    distances, _ = nbrs.kneighbors(point_cloud)
    
    # Use mean of k-th nearest neighbor distances
    kth_distances = distances[:, k]  # k-th neighbor (0-indexed, so k is actually k+1-th)
    mean_kth_distance = np.mean(kth_distances)
    
    return max(mean_kth_distance * scale_factor, 1e-6)

def estimate_max_edge_length_knn_robust(point_cloud, k=6, percentile=75):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(point_cloud)
    distances, _ = nbrs.kneighbors(point_cloud)
    kth_distances = distances[:, k]
    return np.percentile(kth_distances, percentile)

def estimate_max_edge_length(point_cloud: NDArray[np.float64], scale_factor: float = 0.05) -> float:
    """Estimate max edge length from point cloud"""
    data_range = np.linalg.norm(np.ptp(point_cloud, axis=0))
    return max(data_range * scale_factor, 1e-6)

def create_complexes(point_cloud: NDArray[np.float64], max_dimension: int, landmarks_factor: float=0.15) -> Dict[str, ComplexResult]:
    """Create all complexes and compute persistence with optimized parameters for each complex type"""
    complexes = {}
    
    # Prepare landmarks for witness complexes
    num_landmarks = max(1, int(len(point_cloud) * landmarks_factor))
    landmarks = gd.subsampling.choose_n_farthest_points(points=point_cloud, nb_points=num_landmarks)
    witness_points = np.array(landmarks)
    
    # Calculate adaptive parameters based on point cloud characteristics
    _, _, _, median_min_dist = calculate_distance(point_cloud)
    
    max_edge_length = median_min_dist * 2

    # Define complex-specific parameters
    rips_edge_length = max_edge_length * 1.3  # Use provided edge length
    alpha_value = (max_edge_length * 0.6) ** 2  # Slightly smaller, squared for alpha
    cech_value = (max_edge_length * 0.6) ** 2   # Between rips and alpha
    witness_value = (max_edge_length * 1.2) ** 2  # More liberal for witness complexes
    strong_witness_value = (max_edge_length * 1.4) ** 2  # More conservative for strong witness complexes   

    # Define complex configurations with optimized parameters
    complex_configs = [
        ("rips", lambda: gd.RipsComplex(points=point_cloud, max_edge_length=rips_edge_length), 
         {"max_dimension": max_dimension}, point_cloud),
        
        ("cech", lambda: gd.DelaunayCechComplex(points=point_cloud),
         {"max_alpha_square": cech_value}, point_cloud),
        
        ("delaunay_cech", lambda: gd.DelaunayCechComplex(points=point_cloud),
         {}, point_cloud),  # No filtration limit
        
        ("alpha", lambda: gd.AlphaComplex(points=point_cloud), 
         {"max_alpha_square": alpha_value}, point_cloud),
        
        ("delaunay_alpha", lambda: gd.AlphaComplex(points=point_cloud), 
         {}, point_cloud),  # No filtration limit
        
        ("witness", lambda: gd.EuclideanWitnessComplex(witnesses=point_cloud, landmarks=landmarks), 
         {"max_alpha_square": 0}, witness_points),  # Strict witness
        
        ("relaxed_witness", lambda: gd.EuclideanWitnessComplex(witnesses=point_cloud, landmarks=landmarks), 
         {"max_alpha_square": witness_value}, witness_points),
        
        ("strong_witness", lambda: gd.EuclideanStrongWitnessComplex(witnesses=point_cloud, landmarks=landmarks), 
         {"max_alpha_square": strong_witness_value}, witness_points),  # More conservative
    ]
    
    print(f"\nUsing adaptive parameters:")
    print(f"  Rips edge length: {rips_edge_length:.6f}")
    print(f"  Alpha max value: {np.sqrt(alpha_value):.6f} (alpha²={alpha_value:.6f})")
    print(f"  Cech max value: {np.sqrt(cech_value):.6f} (alpha²={cech_value:.6f})")
    print(f"  Witness max value: {np.sqrt(witness_value):.6f} (alpha²={witness_value:.6f})")
    print(f"  Strong witness max value: {np.sqrt(strong_witness_value):.6f} (alpha²={strong_witness_value:.6f})")

    # Create complexes with timing and memory tracking
    for name, complex_factory, stree_params, points in complex_configs:
        try:
            # Memory before complex creation
            mem_before = get_memory_usage()
            creation_start = time.time()
            
            # Create simplex tree
            stree = complex_factory().create_simplex_tree(**stree_params)
            
            persistence = stree.persistence()
            
            processing_time = time.time() - creation_start
            memory_usage = get_memory_usage() - mem_before

            # Store results
            intervals = {}
            # Filter to only include desired dimensions
            filtered_persistence = [(dim, interval) for dim, interval in persistence if dim < max_dimension]

            for dim in range(max_dimension):
                intervals[dim] = stree.persistence_intervals_in_dimension(dim)
                # Transform alpha intervals (square root of filtration values)
                if "alpha" in name:
                    intervals[dim] = np.sqrt(intervals[dim])

            complexes[name] = ComplexResult(
                stree=stree,
                name=name,
                num_simplices=stree.num_simplices(),
                persistence=filtered_persistence,
                intervals=intervals,
                points=points,
                processing_time_s=processing_time,
                memory_usage_mb=memory_usage
            )
            
            # print(f"{name.capitalize()} complex: {complexes[name].num_simplices} simplices "
            #       f"(processed in {processing_time:.3f}s, memory: {format_memory(memory_usage)})")
                  
        except Exception as e:
            print(f"Failed to create {name} complex: {e}")
            continue

    return complexes

def print_performance_table(complex_results: Dict[str, Dict[str, ComplexResult]]):
    """Print a table showing processing time and memory usage for each (sample, complex) combination"""
    
    # Extract all sample names and complex types
    sample_names = list(complex_results.keys())
    if not sample_names:
        print("No results to display")
        return
    
    # Get complex types from the first sample (assuming all samples have same complex types)
    complex_types = list(complex_results[sample_names[0]].keys())
    
    print("\n" + "="*156)
    print("PERFORMANCE SUMMARY TABLE")
    print("="*156)

    # Create header
    header = f"{'Sample':<15}"
    for complex_type in complex_types:
        header += f"{complex_type.replace('_', ' ').title():<18}"
    print(header)
    print("-" * 156)

    # Print time table
    print(f"{'PROCESSING TIME (s)':<15}")
    print("-" * 156)
    for sample in sample_names:
        row = f"{sample:<15}"
        for complex_type in complex_types:
            time_val = complex_results[sample][complex_type].processing_time_s
            row += f"{time_val:<18.3f}"
        print(row)
    
    print()
    
    # Print memory table
    print(f"{'MEMORY USAGE (MB)':<15}")
    print("-" * 156)
    for sample in sample_names:
        row = f"{sample:<15}"
        for complex_type in complex_types:
            memory_val = complex_results[sample][complex_type].memory_usage_mb
            row += f"{memory_val:<18.3f}"
        print(row)
    
    print("-" * 156)
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 60)
    
    # Calculate totals and averages
    for metric in [("processing_time_s"), ("memory_usage_mb")]:
        print(f"\n{metric.replace('_', ' ').title().replace('S', '(s)').replace('Mb', '(MB)')}:")
        
        # Per complex type statistics
        for complex_type in complex_types:
            values = [getattr(complex_results[sample][complex_type], metric) for sample in sample_names]
            avg_val = np.mean(values)
            total_val = np.sum(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            print(f"  {complex_type.replace('_', ' ').title():<20}: "
                  f"Total={total_val:.3f}, Avg={avg_val:.3f}, "
                  f"Min={min_val:.3f}, Max={max_val:.3f}")
    
    print("="*156)

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import src.plotly_persistence as pp

def plot_persistence_diagrams(complexes: Dict[str, ComplexResult], mode: str = "gudhi"):
    """Plot persistence diagrams for all complexes"""
    for complex_result in complexes.values():
        # print(f"\n{complex_result.name.capitalize()} Complex Persistence Diagram:")
        if mode == "plotly":
            pp.plot_persistence_diagram(
                complex_result.persistence, 
                title=f"{complex_result.name.upper()} Persistence Diagram"
            )
        elif mode == "gudhi":
            # Use GUDHI's built-in persistence diagram plotting
            gd.plot_persistence_diagram(complex_result.persistence, legend=True)
            plt.title(f"{complex_result.name.upper()} Persistence Diagram")
            plt.show()
        else:
            raise ValueError("Unsupported plotting mode. Use 'plotly' or 'gudhi'.")

from mpl_toolkits.mplot3d.art3d import Line3DCollection

def visualize_complexes(complexes: Dict[str, ComplexResult], title: str, mode: str = "matplotlib"):
    """Visualize simplicial complexes using either matplotlib or plotly"""
    # print(f"\n{complex_result.name.capitalize()} Complex:")
    if mode == "matplotlib":
        for complex_result in complexes.values():
            triangles = np.array([s[0] for s in complex_result.stree.get_skeleton(2) if len(s[0]) == 3])
            edge_indices = np.array([s[0] for s in complex_result.stree.get_skeleton(1) if len(s[0]) == 2])
            #print(f"{edge_indices=}")
            edges = complex_result.points[edge_indices]
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_trisurf(complex_result.points[:, 0], complex_result.points[:, 1], complex_result.points[:, 2], triangles = triangles)
            ax.scatter3D(complex_result.points[:,0], complex_result.points[:,1], complex_result.points[:,2])
            ax.add_collection3d(Line3DCollection(edges, alpha=0.3))
            plt.title(f"\n{complex_result.name.upper()} Complex Representation of {title}")
            plt.show()
    elif mode == "plotly":
        """Visualize simplicial complexes using Plotly"""
        for complex_result in complexes.values():
            triangles = np.array([s[0] for s in complex_result.stree.get_skeleton(2) if len(s[0]) == 3])
            edges = np.array([s[0] for s in complex_result.stree.get_skeleton(1) if len(s[0]) == 2])
            
            fig = go.Figure()
            
            # Add points
            fig.add_trace(go.Scatter3d(
                x=complex_result.points[:, 0],
                y=complex_result.points[:, 1], 
                z=complex_result.points[:, 2],
                mode='markers',
                marker=dict(size=2.5, color="#4a7fb5"),
                name='Points'
            ))
            #8f90d3
            
            # Add triangular mesh if triangles exist
            if len(triangles) > 0:
                fig.add_trace(go.Mesh3d(
                x=complex_result.points[:, 0],
                y=complex_result.points[:, 1],
                z=complex_result.points[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                # intensity=complex_result.points[:, 2],  # Use Z coordinate for coloring
                # colorscale='Haline',
                color='#1d6fa8',
                # opacity=0.6,
                name='Triangular Mesh',
                showlegend=True
                # showscale=False
                ))
            ##bbbeeb
            
            # Add edges
            if len(edges) > 0:
                edge_x = []
                edge_y = []
                edge_z = []
                for edge in edges:
                    edge_x.extend([complex_result.points[edge[0], 0], complex_result.points[edge[1], 0], None])
                    edge_y.extend([complex_result.points[edge[0], 1], complex_result.points[edge[1], 1], None])
                    edge_z.extend([complex_result.points[edge[0], 2], complex_result.points[edge[1], 2], None])
                
                fig.add_trace(go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode='lines',
                    line=dict(color='#0e456a', width=1.5),
                    name='Edges'
                ))
            ##6b6ca3
            # '#1d6fa8'
            fig.update_layout(
                title=f"{complex_result.name.upper()} Complex Representation of {title}",
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                ),
                showlegend=True
            )
            
            fig.show()
    else:
        raise ValueError("Unsupported visualization mode. Use 'matplotlib' or 'plotly'.")

import copy
from sklearn import manifold

def compute_bottleneck_distances(complexes: Dict[str, ComplexResult], max_dimension: int):
    """Compute all pairwise bottleneck distances and visualize with MDS plots"""
    
    # Get complex names for labeling
    complex_names = list(complexes.keys())
    n_complexes = len(complex_names)

        # Initialize distance tensor: [max_dimension, n_complexes, n_complexes]
    distance_matrices = np.zeros((max_dimension, n_complexes, n_complexes))

    print("\n--- BOTTLENECK DISTANCE COMPUTATION ---")

    # Compute all pairwise distances for distance matrices
    for i, name1 in enumerate(complex_names):
        for j, name2 in enumerate(complex_names[:i]):
            label = (f"{name1}-{name2}")
            print(f"\n{label.upper()}:")
            for dim in range(max_dimension):
                intervals1 = complexes[name1].intervals[dim]
                intervals2 = complexes[name2].intervals[dim]
                distance = gd.bottleneck_distance(intervals1, intervals2)
                print(f"  Dimension {dim}: {distance}")
                distance_matrices[dim, i, j] = distance
                distance_matrices[dim, j, i] = distance

    # Compute max distances across dimensions for each pair of complexes
    max_distances = np.max(distance_matrices, axis=0)
    # for i, name1 in enumerate(complex_names):
    #     for j, name2 in enumerate(complex_names[:i]):
    #         label = f"{name1}-{name2}"
    #         max_distance = max_distances[i, j]
    #         results[label] = max_distance
    
    # Print distance matrix as a table
    print("\n--- MAX BOTTLENECK DISTANCE MATRIX ---")

    # Create header
    header = f"{'Complex':<20}"
    for name in complex_names:
        header += f"{name:<18}"
    print(header)
    print("-" * len(header))
    
    # Print matrix rows
    for i, name1 in enumerate(complex_names):
        row = f"{name1:<20}"
        for j, name2 in enumerate(complex_names):
            if i == j:
                row += f"{'0.000':<18}"
            elif i > j:
                row += f"{max_distances[i, j]:<18.8f}"
            else:
                row += f"{max_distances[j, i]:<18.8f}"
        print(row)

    print("\n--- CORRECTED BOTTLENECK DISTANCE COMPUTATION ---")

    # Handle infinity intervals by finding global maximum and replacing infinities
    all_intervals = []
    for complex_result in complexes.values():
        for dim in range(max_dimension):
            all_intervals.extend(complex_result.intervals[dim])

    # Find global maximum finite death value
    finite_death_values = [
        interval[1] for interval in all_intervals 
        if interval[1] != float('inf')
    ]
    
    if not finite_death_values:
        # If all intervals have infinite death values, use a default replacement
        replacement_value = 1.0
        print(f"\nNo finite death values found. Using default replacement value: {replacement_value}")
    else:
        global_max_death = max(finite_death_values)
        replacement_value = 1.5 * global_max_death
        print(f"\nGlobal maximum death value: {global_max_death}")
        print(f"Using replacement value for infinity intervals: {replacement_value}")
    
    # Create corrected intervals with infinity replacement
    corrected_complexes = {}
    for name, complex_result in complexes.items():
        corrected_intervals = {}
        for dim in range(max_dimension):
            intervals_copy = copy.deepcopy(complex_result.intervals[dim])
            for interval in intervals_copy:
                if interval[1] == float('inf'):
                    interval[1] = replacement_value
            corrected_intervals[dim] = intervals_copy
        corrected_complexes[name] = corrected_intervals

    # Initialize distance tensor: [max_dimension, n_complexes, n_complexes]
    corrected_distance_matrices = np.zeros((max_dimension, n_complexes, n_complexes))
    
    #results = {}

    # Compute all pairwise distances for distance matrices
    for i, name1 in enumerate(complex_names):
        for j, name2 in enumerate(complex_names[:i]):
            label = (f"{name1}-{name2}")
            print(f"\n{label.upper()}:")
            for dim in range(max_dimension):
                intervals1 = corrected_complexes[name1][dim]
                intervals2 = corrected_complexes[name2][dim]
                distance = gd.bottleneck_distance(intervals1, intervals2)
                print(f"  Dimension {dim}: {distance}")
                corrected_distance_matrices[dim, i, j] = distance
                corrected_distance_matrices[dim, j, i] = distance

    # Compute max distances across dimensions for each pair of complexes
    corrected_max_distances = np.max(corrected_distance_matrices, axis=0)
    
    # Print distance matrix as a table
    print("\n--- CORRECTED MAX BOTTLENECK DISTANCE MATRIX ---")

    # Create header
    header = f"{'Complex':<20}"
    for name in complex_names:
        header += f"{name:<18}"
    print(header)
    print("-" * len(header))
    
    # Print matrix rows
    for i, name1 in enumerate(complex_names):
        row = f"{name1:<20}"
        for j, name2 in enumerate(complex_names):
            if i == j:
                row += f"{'0.000':<18}"
            elif i > j:
                row += f"{corrected_max_distances[i, j]:<18.8f}"
            else:
                row += f"{corrected_max_distances[j, i]:<18.8f}"
        print(row)
    
    # Create MDS visualization
    print("\n--- MDS VISUALIZATION ---")

    # Setup MDS
    mds = manifold.MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        dissimilarity="precomputed",
        n_jobs=1,
        n_init=1
    )
    
    # Apply MDS to each dimension's distance matrix
    mds_results = {}
    for dim in range(max_dimension):
        mds_results[dim] = mds.fit_transform(corrected_distance_matrices[dim])
    
    # Create visualization
    _, axes = plt.subplots(1, max_dimension, figsize=(5 * max_dimension, 5))
    if max_dimension == 1:
        axes = [axes]  # Ensure axes is iterable for single subplot
    
    dimension_names = ['Connected Components', 'Loops/Cycles', 'Voids/Cavities']
    
    for dim in range(max_dimension):
        ax = axes[dim]
        
        # Plot points
        ax.scatter(mds_results[dim][:, 0], mds_results[dim][:, 1], 
                           c=range(n_complexes), cmap='cool')
        
        # Add labels
        for i, name in enumerate(complex_names):
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
    
    # Print distance matrix summary
    print("\nDistance Matrix Summary:")
    for dim in range(max_dimension):
        print(f"\nDimension {dim} ({dimension_names[dim]}):")
        print("Complex names:", complex_names)
        print("Distance matrix shape:", corrected_distance_matrices[dim].shape)
        print("Max distance:", np.max(corrected_distance_matrices[dim]))
        print("Min non-zero distance:", np.min(corrected_distance_matrices[dim][corrected_distance_matrices[dim] > 0]))