"""Public TinyStories training entry point."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _main() -> int:
    from small_scale_llm.training.entrypoint import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
