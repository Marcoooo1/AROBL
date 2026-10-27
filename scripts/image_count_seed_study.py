
#!/usr/bin/env python3
"""Image count study with three training seeds."""

import argparse
from pathlib import Path

import image_count_study as base


# Experiment configuration
SCENES = ("scan24", "scan37", "scan55")
COUNTS = (10, 15, 20, 25, 30)
SEEDS = (0, 1, 2)

# Separate output directory
base.STUDY = (
    base.ROOT / "outputs" / "image_count_seed_study"
)

base.SCENES = SCENES
base.COUNTS = COUNTS


def experiment_name(scene, n, seed):
    return f"{scene}_n{n:02d}_s{seed}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=SCENES,
        default=list(SCENES)
    )

    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        choices=COUNTS,
        default=list(COUNTS)
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        choices=SEEDS,
        default=list(SEEDS)
    )

    parser.add_argument(
        "--exp",
        help="Only one experiment"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    parser.add_argument(
        "--dry-run",
        action="store_true"
    )

    args = parser.parse_args()

    # Build experiment matrix.
    experiments = [
        (scene, n, seed)
        for scene in args.scenes
        for n in args.counts
        for seed in args.seeds
    ]

    if args.exp:
        experiments = [
            x for x in experiments
            if experiment_name(*x) == args.exp
        ]

        if not experiments:
            parser.error(
                f"Unknown experiment: {args.exp}"
            )

    rows = base.load_rows()

    # Skip only successful experiments.
    pending = [
        x for x in experiments
        if rows.get(
            experiment_name(*x), {}
        ).get("status") != "ok"
    ]

    if args.limit is not None:
        pending = pending[:args.limit]

    print(
        f"Experiments: {len(experiments)}"
    )
    print(
        f"Already successful: "
        f"{len(experiments) - len(pending)}"
        if args.limit is None
        else f"Selected pending: {len(pending)}"
    )

    for scene, n, seed in pending:
        print(
            f"  {experiment_name(scene, n, seed)}"
        )

    if args.dry_run or not pending:
        return

    base.check_setup(args.scenes)

    # Run each experiment using the existing
    # training and evaluation pipeline.
    for index, (scene, n, seed) in enumerate(
        pending, start=1
    ):
        base.SEED = seed

        name = experiment_name(
            scene, n, seed
        )

        print(
            f"\n[{index}/{len(pending)}] {name}",
            flush=True
        )

        base.run_one(
            scene,
            n,
            rows
        )

    print("\nStudy finished.")
    print(
        "Results:",
        base.STUDY / "metrics.csv"
    )


if __name__ == "__main__":
    main()
