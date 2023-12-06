"""Script to replace LaTeX math symbols in a markdown file."""
import argparse


def replace_delimiters(infile: str, outfile: str) -> None:
    with open(infile, "r", encoding="utf-8") as file:
        content = file.read()

    content = content.replace(r"\( ", "$")
    content = content.replace(r"\(", "$")
    content = content.replace(r" \)", "$")
    content = content.replace(r"\)", "$")

    content = content.replace(r"\[ ", "$$")
    content = content.replace(r"\[", "$$")
    content = content.replace(r" \]", "$$")
    content = content.replace(r"\]", "$$")

    with open(outfile, "w", encoding="utf-8") as file:
        file.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace LaTeX math symbols in a markdown file.")
    parser.add_argument("--infile", type=str, required=True, help="Input markdown file")
    parser.add_argument("--outfile", type=str, required=True, help="Output markdown file")

    args = parser.parse_args()

    replace_delimiters(args.infile, args.outfile)


if __name__ == "__main__":
    main()
