#!/usr/bin/env python
"""What the AlphaFold arm actually contains, and what it would cost to finish.

Exists because the state of this arm has been guessed at repeatedly, expensively
and wrongly. Every number here is read from files on disk rather than estimated:
which complexes came back, at which charge settings, which metrics are populated
and which are empty, and what remains.

Reads only local files. No Modal account, no network, no GPU, no cost.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal_app"))

import pandas as pd

from interface_charge.cli import fail
from interface_charge.config import DEFAULT_CONFIG

#: Metrics the analysis expects. Reported with their fill rate, because a column
#: that is present and entirely null is worse than one that is absent: it looks
#: like data. interface_rmsd_a was exactly that for every job ever run.
EXPECTED_METRICS = [
    "ipsae_d0res",
    "ipsae_d0dom",
    "ipsae_d0chn",
    "interface_ptm",
    "interface_pae",
    "interface_rmsd_a",
    "designed_chain_plddt",
    "mean_plddt",
    "jax_device_kind",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics", type=Path, default=Path("results/af2_metrics.csv"))
    parser.add_argument("--designs", type=Path, default=Path("data/raw/designs.csv"))
    parser.add_argument("--test-set", type=Path, default=Path("data/raw/test_set.csv"))
    parser.add_argument(
        "--definitions", type=Path, default=Path("results/interface_definitions.json")
    )
    parser.add_argument("--max-residues", type=int, default=450)
    parser.add_argument("--only-betas", default="-1.5,0,1.5")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.metrics.is_file():
        fail(
            f"{args.metrics} not found. Download and collect first:\n"
            "  modal volume get --force interface-charge-af2-results / C:\\af2out\n"
            "  python modal_app/collect.py --from-dir C:\\af2out"
        )

    from af2_multimer import build_job_list, estimate_cost

    table = pd.read_csv(args.metrics)
    jobs = build_job_list(args.designs, args.definitions, args.test_set)
    params = DEFAULT_CONFIG.af2

    print("=" * 68)
    print("WHAT IS DONE")
    print("=" * 68)
    print(f"jobs collected:      {len(table)}")
    print(f"complexes:           {table['pdb_id'].nunique()} of {len({j.pdb_id for j in jobs})}")
    print(f"grid size:           {len(jobs)} jobs")

    per_beta = table.groupby("beta").size().sort_index()
    print("\njobs per charge setting:")
    for beta in sorted({j.beta for j in jobs}):
        print(f"  beta {beta:+5.1f}  {int(per_beta.get(beta, 0)):>3}")

    counts = table.groupby("pdb_id")["beta"].nunique()
    for wanted in sorted({counts.max() if len(counts) else 0, 3}, reverse=True):
        if wanted:
            complete = sorted(counts[counts >= wanted].index)
            print(f"\ncomplexes with at least {wanted} charge setting(s): {len(complete)}")
            print(f"  {', '.join(complete)}")

    print("\n" + "=" * 68)
    print("WHICH METRICS ARE ACTUALLY POPULATED")
    print("=" * 68)
    print("A column that is present and entirely empty looks like data and is not.")
    for metric in EXPECTED_METRICS:
        if metric not in table.columns:
            print(f"  {metric:<24} ABSENT")
            continue
        filled = int(table[metric].notna().sum())
        flag = "  <-- never computed" if filled == 0 else ""
        print(f"  {metric:<24} {filled:>3} of {len(table)}{flag}")

    print("\n" + "=" * 68)
    print("WHAT REMAINS, AT THE CURRENT SCOPE")
    print("=" * 68)
    done = set(table["key"]) if "key" in table.columns else set()
    betas = {float(b) for b in args.only_betas.replace(" ", "").split(",")}
    scope = [
        job
        for job in jobs
        if job.key not in done and job.total_residues() <= args.max_residues and job.beta in betas
    ]
    print(f"filter:              <= {args.max_residues} residues, beta in {sorted(betas)}")
    print(f"outstanding:         {len(scope)} jobs over {len({j.pdb_id for j in scope})} complexes")
    if scope:
        lengths = sorted(j.total_residues() for j in scope)
        print(f"residues:            median {lengths[len(lengths) // 2]}, max {lengths[-1]}")
        estimate = estimate_cost(scope)
        print(f"hardware:            {params.gpu_type} at ${params.rate_usd_per_hour():.2f}/hour")
        print(f"estimated cost:      ${estimate['estimated_usd']:.2f}")
        print(f"estimated wall time: {estimate['estimated_wall_clock_hours']:.1f} h")
        print(
            f"\nCAVEAT: the per-job time behind that estimate is "
            f"{params.estimated_minutes_per_job} minutes,\nmeasured while JAX was "
            "silently running on CPU. No GPU-confirmed timing exists yet, so\ntreat "
            "it as an upper bound. The worst case for a single stuck job is "
            f"${params.timeout_s / 3600 * params.rate_usd_per_hour():.2f}."
        )

    excluded = sorted(
        {j.pdb_id for j in jobs} - {j.pdb_id for j in scope} - set(table["pdb_id"].unique())
    )
    if excluded:
        print(f"\ncomplexes excluded by the size filter and not yet run: {len(excluded)}")
        print(f"  {', '.join(excluded)}")

    print("\n" + "=" * 68)
    print("MACHINE READABLE")
    print("=" * 68)
    print(
        json.dumps(
            {
                "jobs_collected": len(table),
                "complexes_collected": int(table["pdb_id"].nunique()),
                "grid_jobs": len(jobs),
                "grid_complexes": len({j.pdb_id for j in jobs}),
                "betas_present": sorted(float(b) for b in table["beta"].unique()),
                "metric_fill": {
                    m: (int(table[m].notna().sum()) if m in table.columns else None)
                    for m in EXPECTED_METRICS
                },
                "outstanding_jobs_in_scope": len(scope),
                "outstanding_complexes_in_scope": len({j.pdb_id for j in scope}),
                "gpu_type": params.gpu_type,
                "usd_per_gpu_hour": params.rate_usd_per_hour(),
                "timeout_s": params.timeout_s,
                "max_containers": params.max_containers,
                "worst_case_usd_per_stuck_job": round(
                    params.timeout_s / 3600 * params.rate_usd_per_hour(), 2
                ),
                "per_job_minutes_assumption": params.estimated_minutes_per_job,
                "per_job_minutes_provenance": (
                    "measured while JAX ran on CPU; no GPU-confirmed timing exists yet"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
