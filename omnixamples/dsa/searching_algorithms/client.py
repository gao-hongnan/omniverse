from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from typing import Callable, List

import rich
from rich.pretty import pprint

from omnivault.dsa.searching_algorithms.context import SearchContext
from omnivault.dsa.searching_algorithms.strategies import (
    IterativeBinarySearchExactMatch,
    LinearSearchForLoop,
)


@dataclass
class Product:
    id: int
    name: str
    price: float


@dataclass
class Inventory:
    items: List[Product] = field(default_factory=list)

    def add_new_item(self, product: Product) -> None:
        self.items.append(product)


def quicksort(items: List[Product], key: Callable[[Product], float]) -> List[Product]:
    """Sorts a list of items using the quicksort algorithm based on a given key."""
    if len(items) <= 1:
        return items
    else:
        pivot = items[0]
        less_than_pivot = [item for item in items[1:] if key(item) <= key(pivot)]
        greater_than_pivot = [item for item in items[1:] if key(item) > key(pivot)]
        return (
            quicksort(less_than_pivot, key)
            + [pivot]
            + quicksort(greater_than_pivot, key)
        )


def find_product_by_id(
    inventory: Inventory, product_id: int, high_traffic: bool = False
) -> Product:
    """Finds a product in the inventory by its ID."""
    if high_traffic:
        inventory.items = quicksort(inventory.items, key=lambda product: product.id)
        strategy = IterativeBinarySearchExactMatch()
    else:
        strategy = LinearSearchForLoop()

    context = SearchContext(strategy=strategy)

    # Assuming a method to extract product IDs from Product objects
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


def main() -> None:
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

    found_product = find_product_by_id(
        inventory, search_id, high_traffic=args.high_traffic
    )
    return found_product


if __name__ == "__main__":
    found_product = main()
    if found_product:
        rich.print(f"Found product: {found_product.name}")
    else:
        rich.print("Product not found.")
