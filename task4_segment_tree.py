"""
Laboratory Work No. 2
Task 4 - Segment Tree for dynamic range minimum queries

The SegmentTreeMin class supports:
1. Construction from an array.
2. Minimum query on [left, right).
3. Point update.

The program also compares the Segment Tree
with the naive min(arr[left:right]) method.
"""

import os
import random
import time


class SegmentTreeMin:
    """
    Segment Tree for range minimum queries.

    The tree is stored in an array.

    Each internal node stores the minimum value
    of its two children.
    """

    def __init__(self, arr):
        """
        Build the Segment Tree from the input array.

        Parameters
        ----------
        arr : list
            Original array.
        """
        if not arr:
            raise ValueError("The input array cannot be empty.")

        self.n = len(arr)

        # Use a power-of-two tree size.
        self.size = 1

        while self.size < self.n:
            self.size *= 2

        # Infinity is used for unused leaves.
        self.tree = [float("inf")] * (2 * self.size)

        # Copy original values to the leaves.
        for i, value in enumerate(arr):
            self.tree[self.size + i] = value

        # Build internal nodes from bottom to top.
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = min(
                self.tree[2 * i],
                self.tree[2 * i + 1],
            )

    def update(self, index, value):
        """
        Replace the value at the specified index.

        Complexity: O(log n).
        """
        if index < 0 or index >= self.n:
            raise IndexError("Index out of range.")

        position = self.size + index

        self.tree[position] = value

        # Recalculate all ancestors.
        position //= 2

        while position >= 1:
            self.tree[position] = min(
                self.tree[2 * position],
                self.tree[2 * position + 1],
            )

            position //= 2

    def query(self, left, right):
        """
        Return the minimum value on [left, right).

        The right boundary is exclusive.

        Complexity: O(log n).
        """
        if left < 0 or right > self.n:
            raise IndexError("Query boundaries are out of range.")

        if left >= right:
            raise ValueError(
                "The query interval must be non-empty."
            )

        left += self.size
        right += self.size

        result = float("inf")

        while left < right:
            if left % 2 == 1:
                result = min(result, self.tree[left])
                left += 1

            if right % 2 == 1:
                right -= 1
                result = min(result, self.tree[right])

            left //= 2
            right //= 2

        return result


def create_results_directory():
    """Create the results directory if necessary."""
    os.makedirs("results", exist_ok=True)


def demonstrate_basic_example():
    """
    Demonstrate the required example from the laboratory manual.
    """
    print("=" * 60)
    print("TASK 4 - SEGMENT TREE")
    print("=" * 60)

    data = [3, 1, 4, 1, 5, 9, 2, 6]

    print("\nOriginal array:")
    print(data)

    segment_tree = SegmentTreeMin(data)

    result_before = segment_tree.query(1, 4)

    print(
        "\nquery(1, 4) before update:",
        result_before,
    )

    print("Expected result: 1")

    segment_tree.update(2, 0)

    print("\nAfter update(2, 0):")
    print(
        "Array conceptually becomes:",
        [3, 1, 0, 1, 5, 9, 2, 6],
    )

    result_after = segment_tree.query(1, 4)

    print(
        "\nquery(1, 4) after update:",
        result_after,
    )

    print("Expected result: 0")

    assert result_before == 1
    assert result_after == 0

    print("\nBasic example: PASSED")


def demonstrate_additional_example():
    """
    Verify the second example from the laboratory manual.
    """
    print("\n" + "=" * 60)
    print("ADDITIONAL TEST")
    print("=" * 60)

    data = [7, 2, 9, 4, 6, 1, 8, 3, 5]

    segment_tree = SegmentTreeMin(data)

    result = segment_tree.query(0, 9)

    print("query(0, 9):", result)
    print("Expected: 1")

    assert result == 1

    result = segment_tree.query(2, 5)

    print("query(2, 5):", result)
    print("Expected: 4")

    assert result == 4

    segment_tree.update(4, 0)

    result = segment_tree.query(2, 5)

    print("After update(4, 0):")
    print("query(2, 5):", result)
    print("Expected: 0")

    assert result == 0

    segment_tree.update(5, 10)

    result = segment_tree.query(0, 9)

    print("After update(5, 10):")
    print("query(0, 9):", result)
    print("Expected: 0")

    assert result == 0

    print("\nAdditional test: PASSED")


def generate_queries(n, m, seed=42):
    """
    Generate m random half-open intervals [left, right).
    """
    random.seed(seed)

    queries = []

    for _ in range(m):
        left = random.randint(0, n - 2)
        right = random.randint(left + 1, n)

        queries.append((left, right))

    return queries


def measure_naive(arr, queries):
    """
    Measure the total execution time of the naive approach.

    The naive method scans every element in each requested range.
    """
    start_time = time.perf_counter()

    total = 0

    for left, right in queries:
        total += min(arr[left:right])

    elapsed = time.perf_counter() - start_time

    return elapsed, total


def measure_segment_tree(arr, queries):
    """
    Measure the total execution time of Segment Tree queries.
    """
    segment_tree = SegmentTreeMin(arr)

    start_time = time.perf_counter()

    total = 0

    for left, right in queries:
        total += segment_tree.query(left, right)

    elapsed = time.perf_counter() - start_time

    return elapsed, total


def performance_experiment():
    """
    Compare the naive algorithm and Segment Tree
    for n=1000,m=1000 and n=10000,m=10000.
    """
    print("\n" + "=" * 60)
    print("PERFORMANCE EXPERIMENT")
    print("=" * 60)

    test_cases = [
        (1000, 1000),
        (10000, 10000),
    ]

    print(
        f"{'n':>10}"
        f"{'m':>10}"
        f"{'Naive (s)':>18}"
        f"{'Segment Tree (s)':>22}"
        f"{'Speedup':>15}"
    )

    print("-" * 75)

    results = []

    for n, m in test_cases:
        random.seed(42)

        arr = [
            random.randint(1, 100000)
            for _ in range(n)
        ]

        queries = generate_queries(n, m, seed=42)

        naive_time, naive_total = measure_naive(
            arr,
            queries,
        )

        tree_time, tree_total = measure_segment_tree(
            arr,
            queries,
        )

        # The total sums should be identical.
        assert naive_total == tree_total

        speedup = (
            naive_time / tree_time
            if tree_time > 0
            else float("inf")
        )

        print(
            f"{n:>10}"
            f"{m:>10}"
            f"{naive_time:>18.6f}"
            f"{tree_time:>22.6f}"
            f"{speedup:>15.2f}x"
        )

        results.append(
            (
                n,
                m,
                naive_time,
                tree_time,
            )
        )

    return results


def save_results(results):
    """
    Save the performance results to a text file.
    """
    create_results_directory()

    filename = "results/task4_results.txt"

    with open(filename, "w", encoding="utf-8") as file:
        file.write("Task 4 - Segment Tree Performance\n")
        file.write("=" * 70 + "\n")

        file.write(
            f"{'n':>10}"
            f"{'m':>10}"
            f"{'Naive(s)':>18}"
            f"{'SegmentTree(s)':>22}\n"
        )

        file.write("-" * 70 + "\n")

        for n, m, naive_time, tree_time in results:
            file.write(
                f"{n:>10}"
                f"{m:>10}"
                f"{naive_time:>18.6f}"
                f"{tree_time:>22.6f}\n"
            )


def main():
    """Run all Task 4 demonstrations and experiments."""
    create_results_directory()

    demonstrate_basic_example()

    demonstrate_additional_example()

    results = performance_experiment()

    save_results(results)

    print(
        "\nPerformance results saved to:"
        " results/task4_results.txt"
    )

    print("\nTask 4 completed successfully.")


if __name__ == "__main__":
    main()
