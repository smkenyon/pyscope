"""`python -m pyscope script.py` entry point. Delegates to the Typer app."""

from __future__ import annotations

from pyscope.cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
