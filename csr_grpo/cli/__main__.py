"""Allow ``python -m csr_grpo.cli ...``."""

from .main import main

if __name__ == "__main__":
    raise SystemExit(main())
