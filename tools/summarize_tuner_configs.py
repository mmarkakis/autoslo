#!/usr/bin/env python3
"""
summarize_tuner_configs.py
==========================
For each named entry in a tuning manifest, reads the tuner run directory and
reports:
  - The selected initial RPUs (initial vs final)
  - The scheduled spinups (initial vs final)
  - The initial and final values of each swept autoscaler parameter

Usage
-----
    python tools/summarize_tuner_configs.py \\
        --tuning_manifest_path data/manifests/tuning/main_eval_v8.yml

    # Filter to specific runs (substring match on run name)
    python tools/summarize_tuner_configs.py \\
        --tuning_manifest_path data/manifests/tuning/main_eval_v8.yml \\
        --filter april15
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Force a wide render so the table is not squeezed on narrow terminals.
console = Console(width=220)

# ---------------------------------------------------------------------------
# YAML helpers (stdlib-only — no autoslo dependency)
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Nested-dict helpers
# ---------------------------------------------------------------------------


def _get_nested(d: dict[str, Any], dotted_key: str) -> Any:
    """Traverse *d* using a dot-separated key path; return None if missing."""
    cur: Any = d
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _shorten_param_key(key: str, all_keys: list[str]) -> str:
    """Return the shortest unique dot-suffix of *key* within *all_keys*."""
    parts = key.split(".")
    for n in range(1, len(parts) + 1):
        suffix = ".".join(parts[-n:])
        if sum(1 for k in all_keys if k.endswith(suffix)) == 1:
            return suffix
    return key


_MAX_COL_HEADER = 16


def _abbrev_header(s: str) -> str:
    """Truncate *s* to at most _MAX_COL_HEADER chars, appending '…' if cut."""
    if len(s) <= _MAX_COL_HEADER:
        return s
    return s[: _MAX_COL_HEADER - 1] + "…"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_rpus(rpus: Any) -> str:
    if rpus is None:
        return "[dim]—[/]"
    if isinstance(rpus, list):
        return "+".join(str(r) for r in rpus)
    return str(rpus)


def _fmt_spinups(spinups: Any) -> str:
    """Compact representation: '32@2366s, 16@6110s' or '(none)'."""
    if not spinups:
        return "[dim](none)[/]"
    parts = []
    for s in spinups:
        t = s.get("rel_time_s", "?")
        rpu = s.get("rpu", "?")
        t_str = f"{t:.0f}s" if isinstance(t, float) else str(t)
        parts.append(f"{rpu}@{t_str}")
    return ", ".join(parts)


def _fmt_changed(init_val: Any, final_val: Any, missing_final: bool) -> str:
    """Return a Rich markup cell showing 'init → final' (green if changed)."""
    if missing_final:
        return f"{_fmt_plain(init_val)} [dim]→ ?[/]"
    if init_val == final_val:
        return f"[dim]{_fmt_plain(init_val)}[/]"
    return (
        f"[dim]{_fmt_plain(init_val)}[/] [green]→ {_fmt_plain(final_val)}[/]"
    )


def _fmt_plain(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, list):
        return "+".join(str(x) for x in v)
    return str(v)


def _canonical_val(v: Any) -> Any:
    """Normalise int-valued floats to int so 1200.0 and 1200 are the same key."""
    if isinstance(v, float) and v == int(v):
        return int(v)
    return v


def _fmt_num(v: Any) -> str:
    """Human-readable number for table column headers and cells."""
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v == int(v) else str(v)
    return str(v)


# ---------------------------------------------------------------------------
# Per-run data extraction
# ---------------------------------------------------------------------------


def _extract_run_info(
    run_name: str, tuner_runs_dir: Path
) -> dict[str, Any]:
    """Return a dict with all relevant tuning outcomes for *run_name*."""
    run_dir = tuner_runs_dir / run_name

    result: dict[str, Any] = {
        "run_name": run_name,
        "run_dir_missing": not run_dir.exists(),
        "initial_config_missing": False,
        "final_config_missing": False,
        "init_rpus": None,
        "final_rpus": None,
        "init_spinups": [],
        "final_spinups": [],
        "swept": {},              # param_key → (init_val, final_val)
        "sweep_param_keys": [],
        "sweep_param_candidates": {},  # param_key → list of candidate values
    }

    if not run_dir.exists():
        return result

    init_cfg = _load_yaml(run_dir / "initial_execution_config.yml")
    final_cfg = _load_yaml(run_dir / "final_execution_config.yml")
    tuner_cfg = _load_yaml(run_dir / "tuner_config.yml")

    result["initial_config_missing"] = not (
        run_dir / "initial_execution_config.yml"
    ).exists()
    result["final_config_missing"] = not (
        run_dir / "final_execution_config.yml"
    ).exists()

    result["init_rpus"] = _get_nested(
        init_cfg, "managed_cluster_pool_config.initial_rpus"
    )
    result["final_rpus"] = _get_nested(
        final_cfg, "managed_cluster_pool_config.initial_rpus"
    )
    result["init_spinups"] = _get_nested(init_cfg, "scheduled_spinups") or []
    result["final_spinups"] = _get_nested(final_cfg, "scheduled_spinups") or []

    # Swept params — read from phase-5 initial/final configs when available so
    # we compare the values going *into* and *out of* the parameter sweep
    # (rather than the top-level initial config, which hasn't had spinups
    # applied yet).
    sweep_spec: dict[str, list] = (
        _get_nested(
            tuner_cfg,
            "autoscaling_param_sweep.param_sweep_config.params",
        )
        or {}
    )
    param_keys = list(sweep_spec.keys())
    result["sweep_param_keys"] = param_keys
    result["sweep_param_candidates"] = {
        k: [_canonical_val(v) for v in vals]
        for k, vals in sweep_spec.items()
    }

    phase5_dir = run_dir / "05_autoscaling_param_sweep"
    p5_init = _load_yaml(phase5_dir / "initial_config.yml")
    p5_final = _load_yaml(phase5_dir / "final_config.yml")

    # Fall back to top-level configs when phase-5 outputs are absent.
    src_init = p5_init if p5_init else init_cfg
    src_final = p5_final if p5_final else final_cfg

    for key in param_keys:
        result["swept"][key] = (
            _get_nested(src_init, key),
            _get_nested(src_final, key),
        )

    return result


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _build_table(
    infos: list[dict[str, Any]],
    all_param_keys: list[str],
) -> Table:
    """Construct the consolidated Rich Table for all runs."""
    # Build column-header abbreviations then truncate to _MAX_COL_HEADER.
    short_names = {k: _shorten_param_key(k, all_param_keys) for k in all_param_keys}
    col_headers = {k: _abbrev_header(short_names[k]) for k in all_param_keys}

    table = Table(
        show_header=True,
        header_style="bold",
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("Run", style="cyan", no_wrap=True, min_width=24)
    table.add_column("Status", justify="center", no_wrap=True, min_width=10)
    table.add_column("init_rpus", justify="right", no_wrap=True, min_width=10)
    table.add_column("final_rpus", justify="right", no_wrap=True, min_width=10)
    table.add_column("spinups (final)", no_wrap=False, min_width=22)
    for key in all_param_keys:
        table.add_column(col_headers[key], justify="right", no_wrap=True, min_width=12)

    for info in infos:
        name = info["run_name"]

        if info["run_dir_missing"]:
            table.add_row(
                name,
                "[red]dir missing[/]",
                *["[dim]—[/]"] * (4 + len(all_param_keys)),
            )
            continue

        if info["final_config_missing"]:
            status = "[yellow]incomplete[/]"
        else:
            status = "[green]✓[/]"

        missing_final = info["final_config_missing"]

        # init_rpus / final_rpus
        init_rpus_cell = _fmt_rpus(info["init_rpus"])
        final_rpus_cell = (
            _fmt_rpus(info["final_rpus"]) if not missing_final else "[dim]—[/]"
        )
        if (
            not missing_final
            and info["init_rpus"] != info["final_rpus"]
        ):
            final_rpus_cell = f"[green]{final_rpus_cell}[/]"

        # spinups
        spinups_cell = (
            _fmt_spinups(info["final_spinups"])
            if not missing_final
            else "[dim]—[/]"
        )

        # swept param cells
        param_cells: list[str] = []
        for key in all_param_keys:
            if key in info["swept"]:
                init_val, final_val = info["swept"][key]
                param_cells.append(
                    _fmt_changed(init_val, final_val, missing_final)
                )
            else:
                param_cells.append("[dim]—[/]")

        table.add_row(
            name,
            status,
            init_rpus_cell,
            final_rpus_cell,
            spinups_cell,
            *param_cells,
        )

    return table


def _build_value_distribution_table(
    infos: list[dict[str, Any]],
    all_param_keys: list[str],
) -> Table | None:
    """
    Summary table: one row per swept parameter, one column per candidate value
    (union across all runs).  Each cell shows count + row-percentage of
    completed runs whose final config selected that value for that parameter.
    Values that are not candidates for a given parameter are shown as '–'.
    """
    complete = [
        i for i in infos
        if not i["run_dir_missing"] and not i["final_config_missing"]
    ]
    if not complete:
        return None

    # Union of candidate values per param (canonical form).
    param_candidates: dict[str, list] = {k: [] for k in all_param_keys}
    for info in complete:
        for key, cands in info.get("sweep_param_candidates", {}).items():
            if key in param_candidates:
                for v in cands:
                    cv = _canonical_val(v)
                    if cv not in param_candidates[key]:
                        param_candidates[key].append(cv)

    # Sort each param's candidate list numerically.
    for key in all_param_keys:
        param_candidates[key].sort(key=float)

    # Final value per param per completed run.
    param_finals: dict[str, list] = {k: [] for k in all_param_keys}
    for info in complete:
        for key in all_param_keys:
            if key in info["swept"]:
                _, final_val = info["swept"][key]
                if final_val is not None:
                    param_finals[key].append(_canonical_val(final_val))

    # Global sorted column values = sorted union of all candidate values.
    all_vals: list = sorted(
        {v for cands in param_candidates.values() for v in cands},
        key=float,
    )
    if not all_vals:
        return None

    short_names = {k: _shorten_param_key(k, all_param_keys) for k in all_param_keys}
    col_headers = {k: _abbrev_header(short_names[k]) for k in all_param_keys}

    table = Table(
        title=(
            f"Swept-Parameter Value Distribution "
            f"({len(complete)} completed run{'s' if len(complete) != 1 else ''})"
        ),
        show_header=True,
        header_style="bold",
        padding=(0, 1),
    )
    table.add_column("Parameter", style="cyan", no_wrap=True, min_width=16)
    for val in all_vals:
        table.add_column(_fmt_num(val), justify="right", min_width=8)

    for key in all_param_keys:
        candidates = set(param_candidates[key])
        finals = param_finals[key]
        total = len(finals)

        # Count occurrences.
        counts: dict[Any, int] = {}
        for v in finals:
            counts[v] = counts.get(v, 0) + 1

        cells: list[str] = []
        for val in all_vals:
            if val not in candidates:
                cells.append("[dim]–[/]")
            else:
                n = counts.get(val, 0)
                pct = f"{n / total * 100:.0f}%" if total > 0 else "–"
                cells.append(f"{n} [dim]({pct})[/]")

        table.add_row(col_headers[key], *cells)

    return table


def _print_legend(all_param_keys: list[str]) -> None:
    """Print a numbered legend mapping column headers to full dotted paths."""
    if not all_param_keys:
        return
    short_names = {k: _shorten_param_key(k, all_param_keys) for k in all_param_keys}
    col_headers = {k: _abbrev_header(short_names[k]) for k in all_param_keys}
    lines = [
        f"  [{i + 1}] [cyan]{col_headers[k]}[/] = {k}"
        for i, k in enumerate(all_param_keys)
        if col_headers[k] != k  # only show when abbreviation differs
    ]
    if lines:
        console.print("[bold]Swept-parameter column legend:[/]")
        for line in lines:
            console.print(line)
        console.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _find_data_dir() -> Path:
    """Locate the repository's data/ directory relative to this script."""
    return Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize tuning outcomes for each run in a tuning manifest.\n"
            "Prints: initial RPUs, final RPUs, scheduled spinups, and\n"
            "initial→final values for all swept parameters."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tuning_manifest_path",
        required=True,
        metavar="PATH",
        help="Path to the YAML tuning manifest.",
    )
    parser.add_argument(
        "--filter",
        metavar="SUBSTRING",
        default="",
        help=(
            "Only include runs whose name contains SUBSTRING "
            "(case-insensitive). Omit to include all runs."
        ),
    )
    args = parser.parse_args()

    manifest_path = Path(args.tuning_manifest_path)
    if not manifest_path.exists():
        console.print(f"[red]Manifest not found:[/] {manifest_path}")
        raise SystemExit(1)

    with manifest_path.open(encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}

    main_content: dict[str, Any] = manifest.get("main_content", {})
    if not main_content:
        console.print("[yellow]No entries found in 'main_content'.[/]")
        return

    # Apply optional name filter.
    filt = args.filter.lower()
    run_names = [
        name for name in main_content if filt in name.lower()
    ]
    if not run_names:
        console.print(
            f"[yellow]No runs match filter '{args.filter}'.[/]"
        )
        return

    tuner_runs_dir = _find_data_dir() / "tuner_runs"

    console.print()
    console.print(
        Panel.fit(
            (
                f"[bold]Manifest:[/] {manifest_path}\n"
                f"[bold]Runs shown:[/] {len(run_names)} "
                f"(of {len(main_content)} total)\n"
                f"[bold]Tuner runs dir:[/] {tuner_runs_dir}"
            ),
            title="Tuner Config Summary",
            border_style="cyan",
        )
    )
    console.print()

    # Extract per-run information.
    infos = [_extract_run_info(name, tuner_runs_dir) for name in run_names]

    # Collect all unique swept param keys in first-seen order.
    seen: set[str] = set()
    all_param_keys: list[str] = []
    for info in infos:
        for k in info["sweep_param_keys"]:
            if k not in seen:
                all_param_keys.append(k)
                seen.add(k)

    _print_legend(all_param_keys)

    table = _build_table(infos, all_param_keys)
    console.print(table)
    console.print()

    # Summary counts.
    n_complete = sum(
        1 for i in infos if not i["run_dir_missing"] and not i["final_config_missing"]
    )
    n_incomplete = sum(
        1 for i in infos if not i["run_dir_missing"] and i["final_config_missing"]
    )
    n_missing = sum(1 for i in infos if i["run_dir_missing"])
    console.print(
        f"[green]{n_complete} complete[/]  "
        f"[yellow]{n_incomplete} incomplete[/]  "
        f"[red]{n_missing} dir missing[/]"
    )

    dist_table = _build_value_distribution_table(infos, all_param_keys)
    if dist_table is not None:
        console.print()
        console.print(dist_table)


if __name__ == "__main__":
    main()
