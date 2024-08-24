import argparse


def parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control script execution mode.")
    parser.add_argument(
        "--async", dest="run_async", action="store_true", help="Run the script asynchronously (default)"
    )
    parser.add_argument("--sync", dest="run_async", action="store_false", help="Run the script synchronously")
    parser.set_defaults(run_async=True)

    args = parser.parse_args()
    return args
