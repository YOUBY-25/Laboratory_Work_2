"""
Laboratory Work No. 2
Task 2: Comparison of KD-Tree and Complete Search

This program:
1. Generates 10,000 uniformly distributed points.
2. Generates 10,000 clustered points.
3. Visualizes both datasets.
4. Builds KD-Trees for both datasets.
5. Measures KD-Tree construction time.
6. Searches for nearest neighbors using KDTree.query().
7. Searches for nearest neighbors using a naive complete search.
8. Repeats measurements 10 times and calculates average times.
9. Saves graphs and results into the "results" folder.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree


# ============================================================
# Configuration
# ============================================================

NUM_POINTS = 10_000
NUM_CLUSTERS = 20
POINTS_PER_CLUSTER = 500
NUM_TEST_POINTS = 100
NUM_MEASUREMENTS = 10

AREA_MIN = 0
AREA_MAX = 1000

CLUSTER_STD = 30

RESULTS_DIR = "results"


# ============================================================
# Utility functions
# ============================================================

def create_results_directory():
    """
    Create the results directory if it does not already exist.

    This prevents FileNotFoundError when saving graphs and files.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_uniform_points(number_of_points):
    """
    Generate uniformly distributed 2D points.

    Parameters:
        number_of_points (int): Number of points.

    Returns:
        numpy.ndarray: Array with shape (number_of_points, 2).
    """
    return np.random.uniform(
        AREA_MIN,
        AREA_MAX,
        (number_of_points, 2)
    )


def generate_clustered_points():
    """
    Generate clustered 2D points.

    There are 20 randomly generated cluster centers.
    Each center produces 500 points using a normal distribution.

    Returns:
        tuple:
            points - generated points;
            labels - cluster number for each point;
            centers - cluster centers.
    """
    centers = np.random.uniform(
        AREA_MIN,
        AREA_MAX,
        (NUM_CLUSTERS, 2)
    )

    points = []
    labels = []

    for cluster_id, center in enumerate(centers):
        cluster_points = np.random.normal(
            loc=center,
            scale=CLUSTER_STD,
            size=(POINTS_PER_CLUSTER, 2)
        )

        # Keep points inside the required [0, 1000) area.
        cluster_points = np.clip(
            cluster_points,
            AREA_MIN,
            AREA_MAX - 0.001
        )

        points.append(cluster_points)
        labels.extend([cluster_id] * POINTS_PER_CLUSTER)

    points = np.vstack(points)
    labels = np.array(labels)

    return points, labels, centers


# ============================================================
# Visualization
# ============================================================

def visualize_datasets(
    uniform_points,
    clustered_points,
    cluster_labels,
    centers
):
    """
    Create scatter plots for the uniform and clustered datasets.

    The resulting image is saved into the results directory.
    """
    create_results_directory()

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --------------------------------------------------------
    # Uniform dataset
    # --------------------------------------------------------

    axes[0].scatter(
        uniform_points[:, 0],
        uniform_points[:, 1],
        s=5,
        alpha=0.6,
        color="blue"
    )

    axes[0].set_title(
        "Uniform Dataset (10,000 points)"
    )
    axes[0].set_xlabel("X coordinate")
    axes[0].set_ylabel("Y coordinate")
    axes[0].set_xlim(0, 1000)
    axes[0].set_ylim(0, 1000)
    axes[0].grid(alpha=0.2)

    # --------------------------------------------------------
    # Clustered dataset
    # --------------------------------------------------------

    scatter = axes[1].scatter(
        clustered_points[:, 0],
        clustered_points[:, 1],
        c=cluster_labels,
        cmap="tab20",
        s=5,
        alpha=0.6
    )

    axes[1].scatter(
        centers[:, 0],
        centers[:, 1],
        marker="x",
        color="black",
        s=80,
        linewidths=2,
        label="Cluster centers"
    )

    axes[1].set_title(
        "Clustered Dataset (20 clusters)"
    )
    axes[1].set_xlabel("X coordinate")
    axes[1].set_ylabel("Y coordinate")
    axes[1].set_xlim(0, 1000)
    axes[1].set_ylim(0, 1000)
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    figure.colorbar(
        scatter,
        ax=axes,
        label="Cluster ID"
    )

    figure.suptitle(
        "Comparison of Uniform and Clustered Point Distributions"
    )

    plt.tight_layout()

    filename = os.path.join(
        RESULTS_DIR,
        "task2_datasets.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(f"Dataset visualization saved to: {filename}")


# ============================================================
# Naive nearest-neighbor search
# ============================================================

def naive_nearest_neighbor(points, query_point):
    """
    Find the nearest point using complete enumeration.

    Parameters:
        points (numpy.ndarray): Dataset of points.
        query_point (numpy.ndarray): Query point.

    Returns:
        tuple:
            distance - distance to the nearest point;
            index - index of the nearest point.
    """
    squared_distances = np.sum(
        (points - query_point) ** 2,
        axis=1
    )

    nearest_index = np.argmin(squared_distances)

    distance = np.sqrt(
        squared_distances[nearest_index]
    )

    return distance, nearest_index


# ============================================================
# KD-Tree search
# ============================================================

def kd_tree_nearest_neighbor(tree, query_point):
    """
    Find the nearest point using KDTree.

    Parameters:
        tree (scipy.spatial.KDTree): KD-Tree.
        query_point (numpy.ndarray): Query point.

    Returns:
        tuple:
            distance - distance to the nearest point;
            index - index of the nearest point.
    """
    distance, index = tree.query(
        query_point,
        k=1
    )

    return distance, index


# ============================================================
# Generate test points
# ============================================================

def generate_test_points():
    """
    Generate 100 random query points.
    """
    return np.random.uniform(
        AREA_MIN,
        AREA_MAX,
        (NUM_TEST_POINTS, 2)
    )


# ============================================================
# Measuring query time
# ============================================================

def measure_kd_tree_queries(
    tree,
    test_points,
    measurements=NUM_MEASUREMENTS
):
    """
    Measure average KD-Tree query time.

    The same 100 test points are queried repeatedly.

    Returns:
        float: Average total time in seconds.
    """
    times = []

    for _ in range(measurements):
        start_time = time.perf_counter()

        for point in test_points:
            kd_tree_nearest_neighbor(tree, point)

        end_time = time.perf_counter()

        times.append(end_time - start_time)

    return np.mean(times)


def measure_naive_queries(
    points,
    test_points,
    measurements=NUM_MEASUREMENTS
):
    """
    Measure average complete-search query time.

    Returns:
        float: Average total time in seconds.
    """
    times = []

    for _ in range(measurements):
        start_time = time.perf_counter()

        for point in test_points:
            naive_nearest_neighbor(points, point)

        end_time = time.perf_counter()

        times.append(end_time - start_time)

    return np.mean(times)


# ============================================================
# Measuring KD-Tree construction
# ============================================================

def measure_kd_tree_build(points):
    """
    Measure the time required to construct a KD-Tree.

    Returns:
        tuple:
            tree - constructed KD-Tree;
            build_time - construction time in seconds.
    """
    start_time = time.perf_counter()

    tree = KDTree(points)

    end_time = time.perf_counter()

    build_time = end_time - start_time

    return tree, build_time


# ============================================================
# Correctness test
# ============================================================

def test_correctness(points, tree):
    """
    Compare KD-Tree and naive search results.

    Returns:
        bool: True if all results match.
    """
    test_points = generate_test_points()

    for point in test_points:
        kd_distance, kd_index = kd_tree_nearest_neighbor(
            tree,
            point
        )

        naive_distance, naive_index = naive_nearest_neighbor(
            points,
            point
        )

        if not np.isclose(kd_distance, naive_distance):
            return False

        # Different indices can sometimes correspond to points
        # at exactly the same distance, so distance is the
        # important correctness criterion.

    return True


# ============================================================
# Print experiment results
# ============================================================

def print_results(
    uniform_build_time,
    uniform_kd_time,
    uniform_naive_time,
    clustered_build_time,
    clustered_kd_time,
    clustered_naive_time
):
    """
    Print the experiment results as a formatted table.
    """
    print("\n")
    print("=" * 80)
    print("TASK 2 - KD-TREE EXPERIMENT RESULTS")
    print("=" * 80)

    print(
        f"{'Dataset':<15}"
        f"{'Build Time (s)':<20}"
        f"{'KDTree Query (s)':<20}"
        f"{'Naive Query (s)':<20}"
    )

    print("-" * 80)

    print(
        f"{'Uniform':<15}"
        f"{uniform_build_time:<20.8f}"
        f"{uniform_kd_time:<20.8f}"
        f"{uniform_naive_time:<20.8f}"
    )

    print(
        f"{'Clustered':<15}"
        f"{clustered_build_time:<20.8f}"
        f"{clustered_kd_time:<20.8f}"
        f"{clustered_naive_time:<20.8f}"
    )

    print("=" * 80)

    uniform_speedup = (
        uniform_naive_time / uniform_kd_time
    )

    clustered_speedup = (
        clustered_naive_time / clustered_kd_time
    )

    print(
        f"\nKD-Tree speedup for uniform data: "
        f"{uniform_speedup:.2f}x"
    )

    print(
        f"KD-Tree speedup for clustered data: "
        f"{clustered_speedup:.2f}x"
    )


# ============================================================
# Save results to a text file
# ============================================================

def save_results(
    uniform_build_time,
    uniform_kd_time,
    uniform_naive_time,
    clustered_build_time,
    clustered_kd_time,
    clustered_naive_time
):
    """
    Save experiment results to a text file.
    """
    create_results_directory()

    filename = os.path.join(
        RESULTS_DIR,
        "task2_results.txt"
    )

    uniform_speedup = (
        uniform_naive_time / uniform_kd_time
    )

    clustered_speedup = (
        clustered_naive_time / clustered_kd_time
    )

    with open(filename, "w", encoding="utf-8") as file:
        file.write(
            "TASK 2 - KD-TREE EXPERIMENT RESULTS\n"
        )
        file.write("=" * 70 + "\n\n")

        file.write(
            f"Number of points: {NUM_POINTS}\n"
        )
        file.write(
            f"Number of test points: {NUM_TEST_POINTS}\n"
        )
        file.write(
            f"Number of measurements: {NUM_MEASUREMENTS}\n"
        )
        file.write(
            f"Number of clusters: {NUM_CLUSTERS}\n"
        )
        file.write(
            f"Points per cluster: {POINTS_PER_CLUSTER}\n"
        )
        file.write(
            f"Cluster standard deviation: {CLUSTER_STD}\n\n"
        )

        file.write(
            f"{'Dataset':<15}"
            f"{'Build Time (s)':<20}"
            f"{'KDTree Query (s)':<20}"
            f"{'Naive Query (s)':<20}\n"
        )

        file.write("-" * 75 + "\n")

        file.write(
            f"{'Uniform':<15}"
            f"{uniform_build_time:<20.8f}"
            f"{uniform_kd_time:<20.8f}"
            f"{uniform_naive_time:<20.8f}\n"
        )

        file.write(
            f"{'Clustered':<15}"
            f"{clustered_build_time:<20.8f}"
            f"{clustered_kd_time:<20.8f}"
            f"{clustered_naive_time:<20.8f}\n"
        )

        file.write("\n")
        file.write(
            f"Uniform KD-Tree speedup: "
            f"{uniform_speedup:.2f}x\n"
        )

        file.write(
            f"Clustered KD-Tree speedup: "
            f"{clustered_speedup:.2f}x\n"
        )

    print(
        f"\nNumerical results saved to: {filename}"
    )


# ============================================================
# Create performance graph
# ============================================================

def create_performance_graph(
    uniform_kd_time,
    uniform_naive_time,
    clustered_kd_time,
    clustered_naive_time
):
    """
    Create a bar chart comparing KD-Tree and naive search.
    """
    create_results_directory()

    labels = [
        "Uniform",
        "Clustered"
    ]

    kd_times = [
        uniform_kd_time,
        clustered_kd_time
    ]

    naive_times = [
        uniform_naive_time,
        clustered_naive_time
    ]

    x_positions = np.arange(len(labels))
    width = 0.35

    figure, axis = plt.subplots(figsize=(9, 6))

    axis.bar(
        x_positions - width / 2,
        kd_times,
        width,
        label="KD-Tree",
        color="steelblue"
    )

    axis.bar(
        x_positions + width / 2,
        naive_times,
        width,
        label="Complete Search",
        color="orange"
    )

    axis.set_xlabel("Dataset")
    axis.set_ylabel("Average query time (seconds)")
    axis.set_title(
        "KD-Tree vs Complete Search"
    )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels)

    axis.legend()
    axis.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    filename = os.path.join(
        RESULTS_DIR,
        "task2_performance.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(
        f"Performance graph saved to: {filename}"
    )


# ============================================================
# Main program
# ============================================================

def main():
    """
    Execute the complete Task 2 experiment.
    """
    print("=" * 80)
    print("LABORATORY WORK No. 2 - TASK 2")
    print("KD-TREE VS COMPLETE SEARCH")
    print("=" * 80)

    np.random.seed(42)

    create_results_directory()

    # --------------------------------------------------------
    # Step 1: Generate datasets
    # --------------------------------------------------------

    print("\n1. Generating datasets...")

    uniform_points = generate_uniform_points(
        NUM_POINTS
    )

    (
        clustered_points,
        cluster_labels,
        centers
    ) = generate_clustered_points()

    print(
        f"Uniform dataset: {len(uniform_points)} points"
    )

    print(
        f"Clustered dataset: {len(clustered_points)} points"
    )

    # --------------------------------------------------------
    # Step 2: Visualize datasets
    # --------------------------------------------------------

    print("\n2. Creating visualization...")

    visualize_datasets(
        uniform_points,
        clustered_points,
        cluster_labels,
        centers
    )

    # --------------------------------------------------------
    # Step 3: Build KD-Trees
    # --------------------------------------------------------

    print("\n3. Building KD-Trees...")

    (
        uniform_tree,
        uniform_build_time
    ) = measure_kd_tree_build(
        uniform_points
    )

    (
        clustered_tree,
        clustered_build_time
    ) = measure_kd_tree_build(
        clustered_points
    )

    print(
        f"Uniform KD-Tree build time: "
        f"{uniform_build_time:.8f} seconds"
    )

    print(
        f"Clustered KD-Tree build time: "
        f"{clustered_build_time:.8f} seconds"
    )

    # --------------------------------------------------------
    # Step 4: Generate test points
    # --------------------------------------------------------

    print("\n4. Generating test points...")

    test_points = generate_test_points()

    print(
        f"Generated {len(test_points)} test points."
    )

    # --------------------------------------------------------
    # Step 5: Correctness verification
    # --------------------------------------------------------

    print("\n5. Checking correctness...")

    uniform_correct = test_correctness(
        uniform_points,
        uniform_tree
    )

    clustered_correct = test_correctness(
        clustered_points,
        clustered_tree
    )

    print(
        f"Uniform dataset correctness: "
        f"{uniform_correct}"
    )

    print(
        f"Clustered dataset correctness: "
        f"{clustered_correct}"
    )

    # --------------------------------------------------------
    # Step 6: Measure KD-Tree queries
    # --------------------------------------------------------

    print("\n6. Measuring KD-Tree query time...")

    uniform_kd_time = measure_kd_tree_queries(
        uniform_tree,
        test_points
    )

    clustered_kd_time = measure_kd_tree_queries(
        clustered_tree,
        test_points
    )

    # --------------------------------------------------------
    # Step 7: Measure complete search
    # --------------------------------------------------------

    print("\n7. Measuring complete-search query time...")

    uniform_naive_time = measure_naive_queries(
        uniform_points,
        test_points
    )

    clustered_naive_time = measure_naive_queries(
        clustered_points,
        test_points
    )

    # --------------------------------------------------------
    # Step 8: Print results
    # --------------------------------------------------------

    print_results(
        uniform_build_time,
        uniform_kd_time,
        uniform_naive_time,
        clustered_build_time,
        clustered_kd_time,
        clustered_naive_time
    )

    # --------------------------------------------------------
    # Step 9: Save results
    # --------------------------------------------------------

    save_results(
        uniform_build_time,
        uniform_kd_time,
        uniform_naive_time,
        clustered_build_time,
        clustered_kd_time,
        clustered_naive_time
    )

    # --------------------------------------------------------
    # Step 10: Create performance graph
    # --------------------------------------------------------

    create_performance_graph(
        uniform_kd_time,
        uniform_naive_time,
        clustered_kd_time,
        clustered_naive_time
    )

    print("\n")
    print("=" * 80)
    print("TASK 2 COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
