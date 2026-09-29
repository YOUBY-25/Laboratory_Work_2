"""
Laboratory Work No. 2
Task 3 - R-tree

This program:
1. Generates 1000 random rectangles.
2. Builds an R-tree spatial index.
3. Performs intersection queries.
4. Performs containment queries.
5. Tests the required small example.
6. Visualizes the spatial query.
7. Measures query performance.
8. Saves results and figures in the results folder.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from rtree import index


# ============================================================
# Configuration
# ============================================================

NUMBER_OF_RECTANGLES = 1000

AREA_MIN = 0
AREA_MAX = 100

MIN_START = 0
MAX_START = 70

MIN_SIZE = 5
MAX_SIZE = 25

RANDOM_SEED = 42

QUERY_BBOX = (30, 30, 60, 60)


# ============================================================
# Utility
# ============================================================

def create_results_folder():
    """Create the results directory if it does not exist."""
    os.makedirs("results", exist_ok=True)


# ============================================================
# Rectangle generation
# ============================================================

def generate_rectangles(
    number_of_rectangles,
    seed=None
):
    """
    Generate random rectangles.

    Each rectangle has the format:

        (xmin, ymin, xmax, ymax)

    Returns
    -------
    list
        List of rectangle tuples.
    """

    rng = np.random.default_rng(seed)

    rectangles = []

    for _ in range(number_of_rectangles):

        xmin = rng.uniform(
            MIN_START,
            MAX_START
        )

        ymin = rng.uniform(
            MIN_START,
            MAX_START
        )

        width = rng.uniform(
            MIN_SIZE,
            MAX_SIZE
        )

        height = rng.uniform(
            MIN_SIZE,
            MAX_SIZE
        )

        xmax = min(
            xmin + width,
            AREA_MAX
        )

        ymax = min(
            ymin + height,
            AREA_MAX
        )

        rectangles.append(
            (
                float(xmin),
                float(ymin),
                float(xmax),
                float(ymax)
            )
        )

    return rectangles


# ============================================================
# R-tree construction
# ============================================================

def build_rtree(rectangles):
    """
    Build an R-tree index from the rectangle list.

    Each rectangle receives a unique integer ID.
    """

    spatial_index = index.Index()

    for rectangle_id, bbox in enumerate(rectangles):

        spatial_index.insert(
            rectangle_id,
            bbox
        )

    return spatial_index


# ============================================================
# Spatial queries
# ============================================================

def intersection_query(
    spatial_index,
    query_bbox
):
    """
    Return IDs of all rectangles whose bounding boxes
    intersect the query rectangle.
    """

    return list(
        spatial_index.intersection(
            query_bbox
        )
    )


def containment_query(
    rectangles,
    query_bbox
):
    """
    Return IDs of rectangles completely contained
    inside the query rectangle.

    The R-tree intersection operation first provides
    candidate rectangles. We then explicitly check
    whether each candidate is completely inside the
    query rectangle.
    """

    query_xmin, query_ymin, query_xmax, query_ymax = (
        query_bbox
    )

    contained_ids = []

    for rectangle_id, rectangle in enumerate(rectangles):

        xmin, ymin, xmax, ymax = rectangle

        if (
            xmin >= query_xmin
            and ymin >= query_ymin
            and xmax <= query_xmax
            and ymax <= query_ymax
        ):
            contained_ids.append(rectangle_id)

    return contained_ids


# ============================================================
# Small test from the laboratory manual
# ============================================================

def run_small_test():
    """
    Test the implementation using the five rectangles
    from the laboratory instructions.
    """

    rectangles = [
        (10, 10, 30, 30),  # ID 0
        (40, 20, 60, 50),  # ID 1
        (25, 40, 45, 70),  # ID 2
        (70, 10, 90, 40),  # ID 3
        (50, 60, 80, 90)   # ID 4
    ]

    query_bbox = (20, 20, 55, 55)

    spatial_index = build_rtree(
        rectangles
    )

    intersections = intersection_query(
        spatial_index,
        query_bbox
    )

    contained = containment_query(
        rectangles,
        query_bbox
    )

    print("\n" + "=" * 70)
    print("R-TREE SMALL TEST")
    print("=" * 70)

    print(
        f"Query rectangle: {query_bbox}"
    )

    print(
        f"Intersection IDs: {intersections}"
    )

    print(
        f"Contained IDs: {contained}"
    )

    expected_intersections = {0, 1, 2}
    expected_contained = {1}

    actual_intersections = set(
        intersections
    )

    actual_contained = set(
        contained
    )

    print("\nExpected intersection IDs:")
    print(expected_intersections)

    print("Expected contained IDs:")
    print(expected_contained)

    if actual_intersections == expected_intersections:
        print(
            "Intersection test: PASSED"
        )
    else:
        print(
            "Intersection test: FAILED"
        )

    if actual_contained == expected_contained:
        print(
            "Containment test: PASSED"
        )
    else:
        print(
            "Containment test: FAILED"
        )


# ============================================================
# Visualization
# ============================================================

def visualize_query(
    rectangles,
    query_bbox,
    intersection_ids,
    contained_ids
):
    """
    Visualize all rectangles and highlight:

    - intersection rectangles in blue;
    - contained rectangles in green;
    - query rectangle with a red border.
    """

    create_results_folder()

    figure, axis = plt.subplots(
        figsize=(10, 10)
    )

    intersection_set = set(
        intersection_ids
    )

    contained_set = set(
        contained_ids
    )

    # Draw all rectangles.
    for rectangle_id, bbox in enumerate(
        rectangles
    ):

        xmin, ymin, xmax, ymax = bbox

        width = xmax - xmin
        height = ymax - ymin

        # Default style.
        face_color = "lightgray"
        edge_color = "gray"
        alpha = 0.35
        line_width = 0.5

        # Contained rectangles are green.
        if rectangle_id in contained_set:

            face_color = "green"
            edge_color = "darkgreen"
            alpha = 0.65
            line_width = 1.5

        # Intersecting rectangles are blue.
        elif rectangle_id in intersection_set:

            face_color = "blue"
            edge_color = "darkblue"
            alpha = 0.45
            line_width = 1.0

        patch = Rectangle(
            (xmin, ymin),
            width,
            height,
            facecolor=face_color,
            edgecolor=edge_color,
            alpha=alpha,
            linewidth=line_width
        )

        axis.add_patch(patch)

    # Draw query rectangle.
    query_xmin, query_ymin, query_xmax, query_ymax = (
        query_bbox
    )

    query_patch = Rectangle(
        (query_xmin, query_ymin),
        query_xmax - query_xmin,
        query_ymax - query_ymin,
        fill=False,
        edgecolor="red",
        linewidth=3,
        label="Query area"
    )

    axis.add_patch(query_patch)

    # Legend elements.
    axis.plot(
        [],
        [],
        color="gray",
        linewidth=8,
        alpha=0.5,
        label="Other rectangles"
    )

    axis.plot(
        [],
        [],
        color="blue",
        linewidth=8,
        alpha=0.5,
        label="Intersecting"
    )

    axis.plot(
        [],
        [],
        color="green",
        linewidth=8,
        alpha=0.7,
        label="Contained"
    )

    axis.set_xlim(
        AREA_MIN,
        AREA_MAX
    )

    axis.set_ylim(
        AREA_MIN,
        AREA_MAX
    )

    axis.set_xlabel(
        "X coordinate"
    )

    axis.set_ylabel(
        "Y coordinate"
    )

    axis.set_title(
        "R-tree Spatial Query"
    )

    axis.set_aspect(
        "equal",
        adjustable="box"
    )

    axis.grid(
        alpha=0.2
    )

    axis.legend()

    plt.tight_layout()

    output_file = (
        "results/task3_rtree_visualization.png"
    )

    figure.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(figure)

    print(
        f"\nVisualization saved to: "
        f"{output_file}"
    )


# ============================================================
# Performance experiment
# ============================================================

def run_performance_experiment(
    rectangles,
    spatial_index,
    query_bbox
):
    """
    Measure the execution time of intersection
    and containment queries.
    """

    measurements = 10

    intersection_times = []
    containment_times = []

    for _ in range(measurements):

        start_time = time.perf_counter()

        intersection_query(
            spatial_index,
            query_bbox
        )

        intersection_times.append(
            time.perf_counter() - start_time
        )

        start_time = time.perf_counter()

        containment_query(
            rectangles,
            query_bbox
        )

        containment_times.append(
            time.perf_counter() - start_time
        )

    average_intersection = np.mean(
        intersection_times
    )

    average_containment = np.mean(
        containment_times
    )

    return (
        average_intersection,
        average_containment
    )


# ============================================================
# Save results
# ============================================================

def save_results(
    rectangles,
    query_bbox,
    intersection_ids,
    contained_ids,
    intersection_time,
    containment_time
):
    """Save Task 3 results to a text file."""

    create_results_folder()

    output_file = (
        "results/task3_results.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "TASK 3 - R-TREE RESULTS\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            f"Number of rectangles: "
            f"{len(rectangles)}\n"
        )

        file.write(
            f"Query bounding box: "
            f"{query_bbox}\n\n"
        )

        file.write(
            f"Number of intersecting rectangles: "
            f"{len(intersection_ids)}\n"
        )

        file.write(
            f"Number of contained rectangles: "
            f"{len(contained_ids)}\n\n"
        )

        file.write(
            "Average intersection query time "
            f"(10 measurements): "
            f"{intersection_time:.8f} seconds\n"
        )

        file.write(
            "Average containment query time "
            f"(10 measurements): "
            f"{containment_time:.8f} seconds\n"
        )

        file.write(
            "\nIntersection IDs:\n"
        )

        file.write(
            str(intersection_ids)
        )

        file.write(
            "\n\nContained IDs:\n"
        )

        file.write(
            str(contained_ids)
        )

    print(
        f"Results saved to: {output_file}"
    )


# ============================================================
# Main
# ============================================================

def main():
    """Run the complete Task 3 experiment."""

    create_results_folder()

    print("=" * 70)
    print("LABORATORY WORK No. 2 - TASK 3")
    print("R-TREE SPATIAL QUERIES")
    print("=" * 70)

    # --------------------------------------------------------
    # Run the required small test.
    # --------------------------------------------------------

    run_small_test()

    # --------------------------------------------------------
    # Generate 1000 rectangles.
    # --------------------------------------------------------

    print(
        "\nGenerating 1000 random rectangles..."
    )

    rectangles = generate_rectangles(
        NUMBER_OF_RECTANGLES,
        RANDOM_SEED
    )

    print(
        f"Generated {len(rectangles)} rectangles."
    )

    # --------------------------------------------------------
    # Build R-tree.
    # --------------------------------------------------------

    print(
        "Building R-tree index..."
    )

    start_time = time.perf_counter()

    spatial_index = build_rtree(
        rectangles
    )

    build_time = (
        time.perf_counter() - start_time
    )

    print(
        f"R-tree build time: "
        f"{build_time:.6f} seconds"
    )

    # --------------------------------------------------------
    # Perform queries.
    # --------------------------------------------------------

    print(
        f"\nQuery bounding box: "
        f"{QUERY_BBOX}"
    )

    intersection_ids = intersection_query(
        spatial_index,
        QUERY_BBOX
    )

    contained_ids = containment_query(
        rectangles,
        QUERY_BBOX
    )

    print(
        f"Intersecting rectangles: "
        f"{len(intersection_ids)}"
    )

    print(
        f"Contained rectangles: "
        f"{len(contained_ids)}"
    )

    # --------------------------------------------------------
    # Performance experiment.
    # --------------------------------------------------------

    (
        intersection_time,
        containment_time
    ) = run_performance_experiment(
        rectangles,
        spatial_index,
        QUERY_BBOX
    )

    print(
        "\nAverage intersection query time: "
        f"{intersection_time:.8f} seconds"
    )

    print(
        "Average containment query time: "
        f"{containment_time:.8f} seconds"
    )

    # --------------------------------------------------------
    # Visualization.
    # --------------------------------------------------------

    visualize_query(
        rectangles,
        QUERY_BBOX,
        intersection_ids,
        contained_ids
    )

    # --------------------------------------------------------
    # Save results.
    # --------------------------------------------------------

    save_results(
        rectangles,
        QUERY_BBOX,
        intersection_ids,
        contained_ids,
        intersection_time,
        containment_time
    )

    print("\n" + "=" * 70)
    print("TASK 3 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
