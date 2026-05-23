"""Main entry point for the context bias experiment."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.runner import run_experiment, generate_conditions
from analysis.analyze import generate_report


def main():
    parser = argparse.ArgumentParser(description="Context Bias Experiment")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run experiment
    run_parser = subparsers.add_parser("run", help="Run the experiment")
    run_parser.add_argument(
        "--domains",
        nargs="+",
        choices=["meals", "code_review", "content_moderation"],
        help="Domains to test (default: all)",
    )
    run_parser.add_argument(
        "--models",
        nargs="+",
        help="Models to test (default: all configured)",
    )
    run_parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        help="Context lengths to test (default: 5 10 20 50)",
    )
    run_parser.add_argument(
        "--polarities",
        nargs="+",
        choices=["no_saturated", "yes_saturated", "neutral"],
        help="Polarities to test (default: all)",
    )
    run_parser.add_argument(
        "--repetitions",
        type=int,
        help="Repetitions per condition (default: 10)",
    )
    run_parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent API calls (default: 4)",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for results",
    )
    run_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from previous results",
    )

    # Analyze results
    analyze_parser = subparsers.add_parser("analyze", help="Analyze results")
    analyze_parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("data/raw/results.jsonl"),
        help="Path to results JSONL file",
    )
    analyze_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for analysis",
    )

    # Count conditions
    count_parser = subparsers.add_parser("count", help="Count total conditions")
    count_parser.add_argument("--domains", nargs="+")
    count_parser.add_argument("--models", nargs="+")
    count_parser.add_argument("--context-lengths", nargs="+", type=int)
    count_parser.add_argument("--repetitions", type=int)

    # Quick test (small run)
    pilot_parser = subparsers.add_parser("pilot", help="Run a small pilot test")
    pilot_parser.add_argument(
        "--model",
        default="qwen3:4b",
        help="Model to test (default: qwen3:4b)",
    )
    pilot_parser.add_argument(
        "--domain",
        default="meals",
        choices=["meals", "code_review", "content_moderation"],
        help="Domain to test (default: meals)",
    )

    args = parser.parse_args()

    if args.command == "run":
        asyncio.run(
            run_experiment(
                output_dir=args.output_dir,
                domains=args.domains,
                models=args.models,
                context_lengths=args.context_lengths,
                polarities=args.polarities,
                repetitions=args.repetitions,
                concurrency=args.concurrency,
                resume=not args.no_resume,
            )
        )

    elif args.command == "analyze":
        generate_report(args.results_file, args.output_dir)

    elif args.command == "count":
        conditions = generate_conditions(
            domains=args.domains,
            models=args.models,
            context_lengths=args.context_lengths,
            repetitions=args.repetitions,
        )
        print(f"Total conditions: {len(conditions)}")

        # Breakdown
        from collections import Counter
        by_model = Counter(c[1] for c in conditions)
        by_domain = Counter(c[0].name for c in conditions)
        by_polarity = Counter(c[2] for c in conditions)

        print("\nBy model:")
        for m, n in sorted(by_model.items()):
            print(f"  {m}: {n}")
        print("\nBy domain:")
        for d, n in sorted(by_domain.items()):
            print(f"  {d}: {n}")
        print("\nBy polarity:")
        for p, n in sorted(by_polarity.items()):
            print(f"  {p}: {n}")

    elif args.command == "pilot":
        print(f"Running pilot: model={args.model}, domain={args.domain}")
        print("Config: 1 repetition, context lengths [5, 10], no_saturated + baseline")
        asyncio.run(
            run_experiment(
                output_dir=Path("data/pilot"),
                domains=[args.domain],
                models=[args.model],
                context_lengths=[5, 10],
                polarities=["no_saturated"],
                repetitions=1,
                concurrency=2,
                resume=False,
            )
        )
        print("\nRunning pilot analysis...")
        generate_report(Path("data/pilot/results.jsonl"), Path("results/pilot"))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
