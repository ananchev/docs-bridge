"""CLI entrypoint.

Usage (design §8 / §15):
    ingest sync --subject aig
    ingest sync --subject all --verbose

Deployed as:
    docker run --rm ingest-worker sync --subject all
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingest", description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sync = sub.add_parser("sync", help="hash-delta sync one or all subjects")
    p_sync.add_argument(
        "--subject",
        required=True,
        help="subject name, or 'all' for every configured subject",
    )
    p_sync.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "sync":
        _setup_logging(args.verbose)
        # import here so --help stays fast and config errors surface first
        from . import sync as sync_mod

        cfg = config.load(args.config)
        names = (
            [s.name for s in cfg.subjects]
            if args.subject == "all"
            else [cfg.subject(args.subject).name]
        )
        results = sync_mod.sync(cfg, names)

        noop = all(r.is_noop for r in results)
        print("\n".join(str(r) for r in results))
        # Exit 0 always; the no-op line is the idempotency signal for §15.
        return 0 if results or noop else 1

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
