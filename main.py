from __future__ import annotations

import sys

from app.cli import build_parser, run_cli


def main() -> int:
    parser = build_parser()
    cli_mode = len(sys.argv) > 1 and not sys.argv[1].startswith("--gui")

    if cli_mode:
        args = parser.parse_args()
        return run_cli(args)

    from app.ui.app import run_gui

    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
