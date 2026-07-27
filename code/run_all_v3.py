# -*- coding: utf-8 -*-
"""Sequential controller for the manuscript-final v5 experiment suite.

The filename is retained for compatibility with the rebuilt v3 data layout,
but the dispatched experiments are the final three-round, no-area-modulation
configuration reported in the manuscript.

Default sequence:
  data        metadata-constrained network and ERA5-Land forcing
  main        five seeds x ten folds for Regional LSTM, GAT, and controls
  supporting  physical-v4 ablation, component, and density evidence
  rounds      one-to-eight-round propagation-depth sweep
  pilrn       final three-round PILRN-S without area modulation
  components  full / no-area / topology-only comparison
  coupled     from-scratch and warm-start coupled controls
  baselines   nearest+DAR, IDW+DAR, and random forest
  density     end-to-end 100% / 50% / 25% training-gauge experiments
  audit       all-month canary audits for every reported batch
  evaluate    unified manuscript tables and paired evidence

All subprocesses run sequentially. Existing completed cells are reused by the
individual resume-safe runners; any failed step stops the pipeline.

Examples:
  python run_all_v3.py
  python run_all_v3.py --only pilrn,components,coupled
  python run_all_v3.py --only audit --audit-device cpu
"""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config_v2 as C


PYTHON = sys.executable
DEFAULT_STEPS = [
    "data",
    "main",
    "supporting",
    "rounds",
    "pilrn",
    "components",
    "coupled",
    "baselines",
    "density",
    "audit",
    "evaluate",
]
VALID_STEPS = set(DEFAULT_STEPS)
CORE_AUDIT_SCOPES = "main,ablation,density,round_sweep,components"
FINAL_AUDIT_SCOPES = (
    "final_pilrn,final_topology,final_coupled,final_density"
)


def run_script(arguments: list[str], label: str) -> None:
    command = [PYTHON, *arguments]
    print(
        f"\n{'=' * 72}\n"
        f">>> [{label}] {' '.join(arguments)}\n"
        f"{'=' * 72}",
        flush=True,
    )
    C.SP_V3.mkdir(parents=True, exist_ok=True)
    with (C.SP_V3 / "run_log.txt").open("a", encoding="utf-8") as log:
        log.write(
            f"\n[{datetime.datetime.now():%F %T}] "
            f"{label}: {' '.join(arguments)}\n"
        )
    result = subprocess.run(command, cwd=str(HERE), check=False)
    if result.returncode:
        raise SystemExit(
            f"[STOP] step {label} failed with return code "
            f"{result.returncode}"
        )


def parse_steps(text: str | None) -> list[str]:
    if text is None:
        return DEFAULT_STEPS.copy()
    steps = [value.strip() for value in text.split(",") if value.strip()]
    unknown = sorted(set(steps) - VALID_STEPS)
    if unknown:
        raise ValueError(
            f"Unknown steps {unknown}; choose from {sorted(VALID_STEPS)}"
        )
    return steps


def audit_arguments(
    scopes: str,
    device: str,
    output_directory: Path,
    resume: bool,
) -> list[str]:
    arguments = [
        "audit_leakage_physical_v4.py",
        "--scopes",
        scopes,
        "--months",
        "all",
        "--device",
        device,
        "--output-dir",
        str(output_directory),
    ]
    if resume:
        arguments.append("--resume")
    return arguments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "Comma-separated subset of data,main,rounds,pilrn,components,"
            "supporting,coupled,baselines,density,audit,evaluate."
        ),
    )
    parser.add_argument(
        "--audit-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="CPU is the release default for bit-exact sparse reductions.",
    )
    parser.add_argument(
        "--fresh-audit",
        action="store_true",
        help="Ignore completed audit rows instead of resuming them.",
    )
    parser.add_argument(
        "--force-data",
        action="store_true",
        help="Rebuild the network and forcing even when final files exist.",
    )
    args = parser.parse_args()
    steps = parse_steps(args.only)

    if "data" in steps:
        nodes = C.BASIN / "river_network_nodes_v3.csv"
        edges = C.BASIN / "river_network_edges_v3.csv"
        if args.force_data or not (nodes.exists() and edges.exists()):
            run_script(["rebuild_network_v3.py"], "network-rebuild")
        else:
            print("[data] final v3 node and edge tables exist; reuse", flush=True)
        if args.force_data or not C.FORCING_NPZ.exists():
            run_script(["prepare_forcing.py", "--extract"], "forcing")
        else:
            print("[data] final forcing matrix exists; reuse", flush=True)

    if "main" in steps:
        run_script(["run_stability_10x5.py"], "main-10fold-x-5seed")

    if "supporting" in steps:
        run_script(
            ["run_ablation_physical_v4.py"],
            "supporting-physical-ablation",
        )
        run_script(
            ["run_component_sweep_physical_v4.py"],
            "supporting-component-sweep",
        )
        run_script(
            ["run_density_physical_v4.py"],
            "supporting-density-evidence",
        )

    if "rounds" in steps:
        run_script(
            ["run_round_sweep_physical_v4.py"],
            "round-sweep-1-to-8",
        )

    if "pilrn" in steps:
        run_script(["run_final_pilrn_v5.py"], "final-pilrn-r3-no-area")

    if "components" in steps:
        run_script(
            ["run_final_components_v5.py"],
            "final-components-r3",
        )

    if "coupled" in steps:
        run_script(
            ["run_final_coupled_v5.py"],
            "final-coupled-controls",
        )

    if "baselines" in steps:
        run_script(["baseline_idw_v2.py"], "full-density-donor-baselines")
        run_script(["baseline_rf_v2.py"], "full-density-random-forest")

    if "density" in steps:
        run_script(["run_final_density_v5.py"], "density-pilrn-regional")
        run_script(["run_density_baselines_v6.py"], "density-baselines")
        run_script(
            ["run_density_coupled_warm_v6.py"],
            "density-coupled-warm",
        )
        run_script(
            ["run_density_coupled_scratch_v7.py"],
            "density-coupled-scratch",
        )

    if "audit" in steps:
        resume = not args.fresh_audit
        # Deterministic full-density baselines and the original main controls.
        run_script(
            [
                "audit_leakage.py",
                "--folds",
                "1-10",
                "--models",
                "all",
                "--months",
                "all",
                "--device",
                args.audit_device,
            ],
            "audit-full-density-methods",
        )
        # The density-baseline runner contains its own 140-cell all-month
        # canary audit and refreshes the summary when existing cells are reused.
        run_script(
            ["run_density_baselines_v6.py"],
            "audit-density-baselines",
        )
        audit_batches = [
            (
                CORE_AUDIT_SCOPES,
                C.SP_V3 / "Leakage_Audit_physical_v4",
                "audit-physical-v4-core",
            ),
            (
                FINAL_AUDIT_SCOPES,
                C.SP_V3 / "Leakage_Audit_final_v5",
                "audit-final-v5",
            ),
            (
                "final_density_coupled",
                C.SP_V3 / "Leakage_Audit_density_coupled_v6",
                "audit-density-coupled-warm",
            ),
            (
                "final_density_coupled_scratch",
                C.SP_V3 / "Leakage_Audit_density_coupled_scratch_v7",
                "audit-density-coupled-scratch",
            ),
        ]
        for scopes, output_directory, label in audit_batches:
            run_script(
                audit_arguments(
                    scopes,
                    args.audit_device,
                    output_directory,
                    resume,
                ),
                label,
            )

    if "evaluate" in steps:
        run_script(
            ["evaluate_final_manuscript_v5.py"],
            "final-manuscript-evaluation",
        )

    print(
        "\n[DONE] manuscript-final sequential pipeline completed; "
        f"outputs: {C.SP_V3}",
        flush=True,
    )


if __name__ == "__main__":
    main()
