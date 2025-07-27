import numpy as np
from dataclasses import dataclass
from typing import Dict
from numpy.typing import NDArray
import gudhi as gd
from mpl_toolkits.mplot3d.art3d import Line3DCollection

@dataclass
class ComplexResult:
    """Store results for a simplicial complex"""
    stree: gd.SimplexTree
    name: str
    num_simplices: int
    persistence: list[tuple[int, tuple[np.float64, np.float64]]]
    intervals: Dict[int, NDArray[np.float64]]
    points: NDArray[np.float64]

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly_persistence as pp

def plot_persistence_diagrams(complexes: Dict[str, ComplexResult], mode: str = "gudhi"):
    """Plot persistence diagrams for all complexes"""
    for complex_result in complexes.values():
        print(f"\n{complex_result.name.capitalize()} Complex Persistence Diagram:")
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

def visualize_complexes(complexes: Dict[str, ComplexResult], title: str, mode: str = "matplotlib"):
    """Visualize simplicial complexes using either matplotlib or plotly"""
    if mode == "matplotlib":
        for complex_result in complexes.values():
            print(f"\n{complex_result.name.capitalize()} Complex:")
        for complex_result in complexes.values():
            print(f"\n{complex_result.name.capitalize()} Complex:")
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
            print(f"\n{complex_result.name.capitalize()} Complex:")
            triangles = np.array([s[0] for s in complex_result.stree.get_skeleton(2) if len(s[0]) == 3])
            edges = np.array([s[0] for s in complex_result.stree.get_skeleton(1) if len(s[0]) == 2])
            
            fig = go.Figure()
            
            # Add points
            fig.add_trace(go.Scatter3d(
                x=complex_result.points[:, 0],
                y=complex_result.points[:, 1], 
                z=complex_result.points[:, 2],
                mode='markers',
                marker=dict(size=3, color="#4a7fb5"),
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

    print("\n" + "="*60)
    print("BOTTLENECK DISTANCE COMPUTATION")
    print("="*60)

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
    print("\n" + "="*80)
    print("MAX BOTTLENECK DISTANCE MATRIX")
    print("="*80)
    
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
    
    print("\n" + "="*60)
    print("CORRECTED BOTTLENECK DISTANCE COMPUTATION")
    print("="*60)
    
    # Handle infinity intervals by finding global maximum and replacing infinities
    all_intervals = []
    for complex_result in complexes.values():
        for dim in range(max_dimension):
            all_intervals.extend(complex_result.intervals[dim])
    
    # Find global maximum finite death value
    global_max_death = max(
        interval[1] for interval in all_intervals 
        if interval[1] != float('inf')
    )
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
    print("\n" + "="*80)
    print("CORRECTED MAX BOTTLENECK DISTANCE MATRIX")
    print("="*80)
    
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
    print("\n" + "="*60)
    print("MDS VISUALIZATION")
    print("="*60)
    
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