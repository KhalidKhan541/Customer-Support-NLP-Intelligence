"""CLI entry point for the Customer Support NLP Intelligence Pipeline.

Provides subcommands for running the full pipeline, generating synthetic
sample tickets, and producing reports from pre-computed enriched data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import yaml


def _load_config(config_path: str) -> dict:
    """Load configuration from a YAML file and return as a dictionary.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if config is None:
        config = {}
    return config


def _merge_config(config: dict, cli_args: argparse.Namespace) -> dict:
    """Merge CLI arguments into the loaded configuration.

    Non-None CLI values override their corresponding config keys.  Nested
    config sections (``pipeline``, ``output``) receive flattened CLI flags
    where applicable.
    """
    merged = {**config}

    # Top-level overrides
    for key in ("input", "output"):
        cli_val = getattr(cli_args, key, None)
        if cli_val is not None:
            merged[key] = cli_val

    # Pipeline section overrides
    pipeline = merged.setdefault("pipeline", {})
    if getattr(cli_args, "config", None) and "model" not in pipeline:
        pipeline["model"] = pipeline.get("model", "default")

    # Report-specific overrides
    if getattr(cli_args, "data", None):
        merged["data"] = cli_args.data

    if getattr(cli_args, "n", None) is not None:
        merged["n"] = cli_args.n

    return merged


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _cmd_run(config: dict) -> int:
    """Execute the full NLP intelligence pipeline.

    Loads raw ticket data, applies NLP enrichment (intent classification,
    sentiment analysis, entity extraction), and writes the enriched
    dataset to the configured output path.
    """
    from .pipeline import run_pipeline  # noqa: C0415 – lazy import

    input_path = config.get("input")
    output_path = config.get("output", "output/enriched_tickets.csv")

    if not input_path:
        print("[ERROR] No input file specified. Use --input or set 'input' in config.", file=sys.stderr)
        return 1

    if not Path(input_path).is_file():
        print(f"[ERROR] Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    print(f"[INFO] Loading tickets from: {input_path}")
    print(f"[INFO] Output will be written to: {output_path}")
    print("[INFO] Running NLP enrichment pipeline …")

    try:
        run_pipeline(
            input_path=input_path,
            output_path=output_path,
            config=config,
        )
    except Exception as exc:
        print(f"[ERROR] Pipeline failed: {exc}", file=sys.stderr)
        return 1

    print("[INFO] Pipeline completed successfully.")
    return 0


def _cmd_generate_sample(config: dict) -> int:
    """Generate synthetic support tickets for testing and demonstration.

    Writes a CSV file containing ``n`` randomly generated tickets with
    realistic field distributions.
    """
    from .synthetic import generate_tickets  # noqa: C0415 – lazy import

    n = config.get("n", 50)
    output_path = config.get("output", "sample_tickets.csv")

    print(f"[INFO] Generating {n} synthetic tickets → {output_path}")

    try:
        generate_tickets(n=n, output_path=output_path)
    except Exception as exc:
        print(f"[ERROR] Sample generation failed: {exc}", file=sys.stderr)
        return 1

    print(f"[INFO] Done. {n} tickets written to {output_path}")
    return 0


def _cmd_report_only(config: dict) -> int:
    """Produce a summary report from pre-computed enriched data.

    Expects the enriched CSV (with intent, sentiment, entities already
    present) and generates a structured report.
    """
    from .reporting import generate_report  # noqa: C0415 – lazy import

    data_path = config.get("data")
    output_path = config.get("output", "report.html")

    if not data_path:
        print("[ERROR] No data file specified. Use --data or set 'data' in config.", file=sys.stderr)
        return 1

    if not Path(data_path).is_file():
        print(f"[ERROR] Data file does not exist: {data_path}", file=sys.stderr)
        return 1

    print(f"[INFO] Reading enriched data from: {data_path}")
    print(f"[INFO] Report will be saved to: {output_path}")

    try:
        generate_report(data_path=data_path, output_path=output_path, config=config)
    except Exception as exc:
        print(f"[ERROR] Report generation failed: {exc}", file=sys.stderr)
        return 1

    print("[INFO] Report generated successfully.")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="customer-support-nlp",
        description="Customer Support NLP Intelligence Pipeline – analyse, enrich, and report on support tickets.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # -- run ---------------------------------------------------------------
    run_parser = subparsers.add_parser("run", help="Run the full NLP pipeline")
    run_parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="Path to the raw tickets CSV file.",
    )
    run_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path for the enriched output CSV file.",
    )
    run_parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to a YAML configuration file.",
    )

    # -- generate-sample ---------------------------------------------------
    gen_parser = subparsers.add_parser("generate-sample", help="Generate synthetic sample tickets")
    gen_parser.add_argument(
        "-n",
        type=int,
        default=50,
        help="Number of synthetic tickets to generate (default: 50).",
    )
    gen_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path for the generated tickets.",
    )
    gen_parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to a YAML configuration file.",
    )

    # -- report-only -------------------------------------------------------
    report_parser = subparsers.add_parser("report-only", help="Generate report from enriched data")
    report_parser.add_argument(
        "-d", "--data",
        type=str,
        default=None,
        help="Path to the pre-computed enriched CSV.",
    )
    report_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path for the generated report.",
    )
    report_parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to a YAML configuration file.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_COMMANDS = {
    "run": _cmd_run,
    "generate-sample": _cmd_generate_sample,
    "report-only": _cmd_report_only,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the CLI.

    Parses arguments, loads/merges configuration, and dispatches to the
    appropriate subcommand handler.

    Args:
        argv: Command-line arguments.  Defaults to ``sys.argv[1:]``.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    config: dict = {}
    if args.config:
        try:
            config = _load_config(args.config)
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        except yaml.YAMLError as exc:
            print(f"[ERROR] Invalid YAML in config file: {exc}", file=sys.stderr)
            return 1

    config = _merge_config(config, args)

    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(config)


if __name__ == "__main__":
    sys.exit(main())
