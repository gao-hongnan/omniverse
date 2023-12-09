"""
This module provides a strategy design pattern implementation for searching
products by ID in an inventory system. It supports two search strategies:
linear search and binary search, chosen based on the specified traffic conditions.
The module includes classes for representing products and inventory,
a quicksort algorithm for sorting, and a command-line interface for user interaction.

Example
-------
To run the script with command-line arguments:
`python omnixamples/dsa/searching_algorithms/client.py --high-traffic --product-id 103`
"""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import rich
from rich.pretty import pprint

from omnivault.dsa.searching_algorithms.context import SearchContext
from omnivault.dsa.searching_algorithms.strategies import (
    IterativeBinarySearchExactMatch, LinearSearchForLoop)


@dataclass
class Product:
    """A class for representing a product."""

    id: int
    name: str
    price: float


@dataclass
class Inventory:
    """A class for representing an inventory of products."""

    items: List[Product] = field(default_factory=list)

    def add_new_item(self, product: Product) -> None:
        """Adds a new product to the inventory."""
        self.items.append(product)


def quicksort(items: List[Product], key: Callable[[Product], float]) -> List[Product]:
    """
    Sort a list of products using the quicksort algorithm.

    Parameters
    ----------
    items : List[Product]
        A list of Product objects to be sorted.
    key : Callable[[Product], float]
        A function that extracts the sorting key from a Product object.

    Returns
    -------
    List[Product]
        The sorted list of products.
    """
    if len(items) <= 1:
        return items

    pivot = items[0]
    less_than_pivot = [item for item in items[1:] if key(item) <= key(pivot)]
    greater_than_pivot = [item for item in items[1:] if key(item) > key(pivot)]
    return quicksort(less_than_pivot, key) + [pivot] + quicksort(greater_than_pivot, key)


def find_product_by_id(inventory: Inventory, product_id: int, high_traffic: bool = False) -> Optional[Product]:
    """Finds a product in the inventory by its ID."""
    if high_traffic:
        inventory.items = quicksort(inventory.items, key=lambda product: product.id)
        strategy = IterativeBinarySearchExactMatch()
    else:
        strategy = LinearSearchForLoop()  # type: ignore

    context = SearchContext(strategy=strategy)

    product_ids = [product.id for product in inventory.items]
    index = context.execute_search(container=product_ids, target=product_id)

    return inventory.items[index] if index != -1 else None


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--high-traffic",
        action="store_true",
        help="Use binary search for high traffic conditions.",
    )
    parser.add_argument(
        "--product-id",
        default=103,
        type=int,
        help="The ID of the product to search for.",
    )
    return parser.parse_args()


def main() -> Optional[Product]:
    """Main driver."""
    args = parse_args()
    # Creating an instance of Inventory
    inventory = Inventory()

    # Adding products to the inventory
    inventory.add_new_item(Product(id=103, name="Headphones", price=150.00))
    inventory.add_new_item(Product(id=101, name="Laptop", price=1200.00))
    inventory.add_new_item(Product(id=102, name="Smartphone", price=800.00))
    inventory.add_new_item(Product(id=104, name="Keyboard", price=100.00))
    inventory.add_new_item(Product(id=105, name="Mouse", price=50.00))

    pprint(inventory.items)

    # Product ID to search for
    search_id = args.product_id

    found_product = find_product_by_id(inventory, search_id, high_traffic=args.high_traffic)
    return found_product


if __name__ == "__main__":
    # python omnixamples/dsa/searching_algorithms/client.py --high-traffic --product-id 103
    found_product = main()
    if found_product:
        rich.print(f"Found product: {found_product.name}")
    else:
        rich.print("Product not found.")
