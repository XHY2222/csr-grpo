"""Allow ``python -m csr_grpo ...`` as a shortcut for the CLI."""

from .cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
