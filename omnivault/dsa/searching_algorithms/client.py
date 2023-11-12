from typing import List
from omnivault.dsa.searching_algorithms.context import SearchContext
from omnivault.dsa.searching_algorithms.strategies import (
    LinearSearchForLoop, IterativeBinarySearchExactMatch,
    RecursiveBinarySearchExactMatch)

def is_sorted(data: List[int]) -> bool:
    """Check if the given list is sorted."""
    return all(data[i] <= data[i + 1] for i in range(len(data) - 1))

def get_search_strategy(data: List[int]):
    """Determine the appropriate search strategy based on the dataset."""
    if is_sorted(data):
        # Use binary search for sorted data
        return IterativeBinarySearchExactMatch()
    else:
        # Use linear search for unsorted data
        return LinearSearchForLoop()

def main():
    # Example dataset
    data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    target = 9

    # Determine the appropriate strategy
    strategy = get_search_strategy(data)
    search_context = SearchContext(strategy=strategy)

    # Perform the search
    print(f"Searching for {target} in {data}")
    index = search_context.execute_search(data, target)

    if index != -1:
        print(f"Found {target} at index {index} using {type(strategy).__name__}")
    else:
        print(f"{target} not found in the dataset")

    # Optionally switch to a recursive binary search if data is sorted
    if is_sorted(data):
        print("Switching to Recursive Binary Search")
        search_context.strategy = RecursiveBinarySearchExactMatch()
        index = search_context.execute_search(data, target)
        print(f"Found {target} at index {index} using RecursiveBinarySearchExactMatch")

if __name__ == "__main__":
    main()
