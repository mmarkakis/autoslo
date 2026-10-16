#!/usr/bin/env python3
"""plots_panel_summary_table.py
--------------------------------
Given a subdirectory of data/plots (as produced by plot.py), reads the plot
CSV there and writes a companion LaTeX table where each row is one figure
panel/scenario, with a VR + Cost column pair per method.

Unlike plot.py's own per-(lambda,kappa,C) table, VR values here are never
bold/underlined; instead a VR value is italicized whenever it is under the
scenario's SLO threshold, read from that panel's `threshold_lines` in the
plotting manifest (data/manifests/plotting/<dir_name>.yml) that produced the
directory. A summary row reports, per method, the number of scenarios with
an "okay" VR (e.g. 3/8) and the total cost across all scenarios.

Panel titles must follow the 'Cluster <id>, <Month> <day>, κ=<kappa>' format
(e.g. "Cluster 157, April 15, κ=4") — they are split into Cluster/Day/κ
columns, otherwise the tool errors out. Methods are grouped and reordered:
AutoSLO first, then any "Single"/"Round Robin" methods under a shared
"Round Robin" super-header, then any "RAIS" methods under a shared "RAIS"
super-header. Unrecognized methods are shown standalone with a warning.

Usage:
    python tools/plots_panel_summary_table.py data/plots/main_eval_v8_plus_extra
    python tools/plots_panel_summary_table.py main_eval_v8_plus_extra --live
"""

from __future__ import annotations

import argparse
import csv as csv_module
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console

import autoslo.filesystem.path_utils as pu
from autoslo.filesystem.yaml_helpers import load_yaml

console = Console()

AUTOSLO_LABEL = "AutoSLO"

_SCENARIO_TITLE_RE = re.compile(
    r"^Cluster\s+(?P<cluster>\S+),\s+(?P<month>[A-Za-z]+)\s+"
    r"(?P<day>\d+),\s+κ=(?P<kappa>\S+)$"
)


def _tex_escape(s: str) -> str:
    """Escape special LaTeX characters in plain text."""
    for char, repl in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]:
        s = s.replace(char, repl)
    return s


@dataclass
class _PanelRow:
    cluster: str
    day: str
    kappa: str
    threshold: float | None
    points: "OrderedDict[str, tuple[float, float]]"  # method -> (vr, cost)


@dataclass
class _ColumnGroup:
    """One top-level method header, spanning one or more (label, subheader)
    columns. A single member with subheader=None renders as a standalone
    header spanning both header rows (e.g. AutoSLO, or an unrecognized
    method); multiple members (or a subheader) render as a group with its
    own sub-header row (e.g. Round Robin -> 64 / 32+32).
    """

    header: str
    members: list[tuple[str, str | None]] = field(default_factory=list)

    @property
    def is_standalone(self) -> bool:
        return len(self.members) == 1 and self.members[0][1] is None


def _parse_scenario_title(title: str) -> tuple[str, str, str]:
    """Split a 'Cluster <id>, <Month> <day>, κ=<kappa>' panel title into
    (cluster, 'M/D' day, kappa). Raises ValueError if the title doesn't
    match that format.
    """
    match = _SCENARIO_TITLE_RE.match(title.strip())
    if not match:
        raise ValueError(
            f"Panel title {title!r} doesn't match the expected "
            "'Cluster <id>, <Month> <day>, κ=<kappa>' format."
        )
    month_num = datetime.strptime(match.group("month"), "%B").month
    day = f"{month_num}/{int(match.group('day'))}"
    return match.group("cluster"), day, match.group("kappa")


def _require_common_threshold(panel_rows: list[_PanelRow]) -> float:
    """Verify every scenario shares the same SLO violation-rate target and
    return it. Raises ValueError if thresholds differ or are missing.
    """
    thresholds = {panel_row.threshold for panel_row in panel_rows}
    if len(thresholds) != 1 or None in thresholds:
        raise ValueError(
            "All scenarios must share the same SLO violation rate target "
            f"(found: {sorted(t for t in thresholds if t is not None)}, "
            f"missing for {sum(1 for t in thresholds if t is None)} "
            "scenario(s))."
        )
    return thresholds.pop()


def _classify_method(label: str) -> tuple[str, str | None]:
    """Return (group_header, subheader) for a method label. AutoSLO and
    unrecognized methods get subheader=None (standalone column).
    """
    if label == AUTOSLO_LABEL:
        return AUTOSLO_LABEL, None
    if "Round Robin" in label:
        return "Round Robin", label.split("Round Robin", 1)[1].strip()
    if "Single" in label:
        return "Round Robin", label.split("Single", 1)[1].strip()
    if "RAIS" in label:
        return "RAIS", label.split("RAIS", 1)[1].lstrip(" -").strip()
    return label, None


def _build_column_groups(method_order: list[str]) -> list[_ColumnGroup]:
    """Group/reorder methods: AutoSLO first, then Round Robin/Single, then
    RAIS, then any unrecognized methods (each standalone, with a warning).
    """
    autoslo_members: list[tuple[str, str | None]] = []
    round_robin_members: list[tuple[str, str | None]] = []
    rais_members: list[tuple[str, str | None]] = []
    other_groups: list[_ColumnGroup] = []

    for method in method_order:
        group_header, subheader = _classify_method(method)
        if group_header == AUTOSLO_LABEL:
            autoslo_members.append((method, subheader))
        elif group_header == "Round Robin":
            round_robin_members.append((method, subheader))
        elif group_header == "RAIS":
            rais_members.append((method, subheader))
        else:
            console.print(
                f"[yellow]Warning: method '{method}' doesn't match the "
                "AutoSLO/Round Robin/Single/RAIS naming conventions — "
                "showing it as a standalone column.[/]"
            )
            other_groups.append(
                _ColumnGroup(header=method, members=[(method, None)])
            )

    groups: list[_ColumnGroup] = []
    if autoslo_members:
        groups.append(
            _ColumnGroup(header=AUTOSLO_LABEL, members=autoslo_members)
        )
    if round_robin_members:
        groups.append(
            _ColumnGroup(header="Round Robin", members=round_robin_members)
        )
    if rais_members:
        groups.append(_ColumnGroup(header="RAIS", members=rais_members))
    groups += other_groups
    return groups


def _resolve_csv_path(target: Path, live: bool | None) -> Path:
    if target.is_file():
        return target
    if not target.is_dir():
        raise FileNotFoundError(f"Not a file or directory: {target}")

    name = target.name
    live_csv = target / f"{name}#live.csv"
    sim_csv = target / f"{name}.csv"
    if live is True:
        if not live_csv.exists():
            raise FileNotFoundError(f"No live CSV found at {live_csv}")
        return live_csv
    if live is False:
        if not sim_csv.exists():
            raise FileNotFoundError(f"No simulation CSV found at {sim_csv}")
        return sim_csv
    # Auto-detect: only proceed if unambiguous.
    if live_csv.exists() and sim_csv.exists():
        raise FileNotFoundError(
            f"Both {live_csv.name} and {sim_csv.name} exist in {target} — "
            "pass --live or --sim to disambiguate."
        )
    if live_csv.exists():
        return live_csv
    if sim_csv.exists():
        return sim_csv
    raise FileNotFoundError(f"No plot CSV found in {target}")


def _load_panel_thresholds(
    dir_name: str,
) -> dict[tuple[int | None, int | None], float]:
    """Read each panel's SLO threshold (first threshold_lines value) from
    the plotting manifest that produced this plots directory, keyed by
    (row, col). Returns an empty dict if the manifest/thresholds aren't found.
    """
    manifest_path = (
        pu.get_data_dir() / "manifests" / "plotting" / f"{dir_name}.yml"
    )
    if not manifest_path.exists():
        return {}

    content = load_yaml(manifest_path).get("main_content", {})
    thresholds: dict[tuple[int | None, int | None], float] = {}
    if "panels" in content:
        for panel in content["panels"]:
            tls = panel.get("threshold_lines") or []
            if tls:
                thresholds[(panel["row"], panel["col"])] = float(
                    tls[0]["value"]
                )
    else:
        tls = content.get("threshold_lines") or []
        if tls:
            thresholds[(None, None)] = float(tls[0]["value"])
    return thresholds


def _load_panel_rows(
    csv_path: Path,
    thresholds: dict[tuple[int | None, int | None], float],
) -> tuple[list[_PanelRow], list[str]]:
    panel_rows: "OrderedDict[tuple[int | None, int | None], _PanelRow]" = (
        OrderedDict()
    )
    method_order: list[str] = []
    seen_methods: set[str] = set()

    with csv_path.open(newline="") as f:
        for row in csv_module.DictReader(f):
            key = (
                int(row["row"]) if row["row"] != "" else None,
                int(row["col"]) if row["col"] != "" else None,
            )
            method = row["label"]

            if key not in panel_rows:
                cluster, day, kappa = _parse_scenario_title(row["panel_title"])
                panel_rows[key] = _PanelRow(
                    cluster=cluster,
                    day=day,
                    kappa=kappa,
                    threshold=thresholds.get(key),
                    points=OrderedDict(),
                )
            panel_rows[key].points[method] = (float(row["x"]), float(row["y"]))

            if method not in seen_methods:
                method_order.append(method)
                seen_methods.add(method)

    return list(panel_rows.values()), method_order


def _write_latex_table(
    tex_path: Path,
    panel_rows: list[_PanelRow],
    method_order: list[str],
    threshold: float,
) -> None:
    groups = _build_column_groups(method_order)

    # --- Column spec: 3 scenario columns, then 2 columns per method,
    # with a "|" between members of a group and between groups. -----------
    col_parts = ["c", "c", "c", "||"]
    for gi, group in enumerate(groups):
        for mi, _ in enumerate(group.members):
            col_parts += ["c", "c"]
            if mi < len(group.members) - 1:
                col_parts += ["|"]
        if gi < len(groups) - 1:
            col_parts += ["|"]
    col_spec = " ".join(col_parts)

    lines: list[str] = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath}",
        r"\usepackage[landscape, margin=1cm]{geometry}",
        r"\begin{document}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        rf"\caption{{End-to-end evaluation with SLO violation rate target "
        rf"$\tau={threshold:.2f}$.}}\label{{tab:todo}}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
    ]

    # --- Header row 1: top-level group headers for multi-member groups
    # only (Round Robin, RAIS); standalone methods (e.g. AutoSLO) and the
    # scenario columns are left blank here and shown in row 2 instead. -----
    row1 = ["", "", ""]
    for gi, group in enumerate(groups):
        ncols = 2 * len(group.members)
        bar = "" if gi == len(groups) - 1 else "|"
        if group.is_standalone:
            row1 += ["", ""]
        else:
            header_text = _tex_escape(group.header)
            row1.append(rf"\multicolumn{{{ncols}}}{{c{bar}}}{{{header_text}}}")
    lines.append(" & ".join(row1) + r" \\")

    # --- Header row 2: scenario column labels, per-member sub-headers
    # (e.g. 64 / 32+32), and standalone method names (e.g. AutoSLO). -------
    row2 = ["Cluster", "Day", r"$\kappa$"]
    for gi, group in enumerate(groups):
        bar = "" if gi == len(groups) - 1 else "|"
        if group.is_standalone:
            header_text = _tex_escape(group.header)
            row2.append(rf"\multicolumn{{2}}{{c{bar}}}{{{header_text}}}")
        else:
            for mi, (_label, subheader) in enumerate(group.members):
                # Last member's trailing bar must match the group's own
                # trailing bar, so the group boundary isn't lost here.
                member_bar = "|" if mi < len(group.members) - 1 else bar
                row2.append(
                    rf"\multicolumn{{2}}{{c{member_bar}}}{{{_tex_escape(subheader or '')}}}"
                )
    lines.append(" & ".join(row2) + r" \\")

    # --- Header row 3: VR / Cost labels. ------------------------------------
    row3 = ["", "", ""]
    for group in groups:
        for _ in group.members:
            row3 += ["VR", "Cost"]
    lines.append(" & ".join(row3) + r" \\")
    lines.append(r"\hline")

    ok_counts: dict[str, int] = {m: 0 for m in method_order}
    totals: dict[str, int] = {m: 0 for m in method_order}
    total_cost: dict[str, float] = {m: 0.0 for m in method_order}

    for panel_row in panel_rows:
        cells = [
            _tex_escape(panel_row.cluster),
            _tex_escape(panel_row.day),
            _tex_escape(panel_row.kappa),
        ]
        for group in groups:
            for method, _subheader in group.members:
                if method in panel_row.points:
                    vr, cost = panel_row.points[method]
                    vr_cell = f"{vr:.4f}"
                    if (
                        panel_row.threshold is not None
                        and vr < panel_row.threshold
                    ):
                        vr_cell = rf"\textit{{{vr_cell}}}"
                        ok_counts[method] += 1
                    totals[method] += 1
                    total_cost[method] += cost
                    cells.append(vr_cell)
                    cells.append(rf"\${cost:.2f}")
                else:
                    cells += ["---", "---"]
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\hline")
    vr_summary_cells: list[str] = [r"\multicolumn{3}{c||}{VR Target Met}"]
    cost_summary_cells: list[str] = [r"\multicolumn{3}{c||}{Total cost}"]
    for group in groups:
        for method, _subheader in group.members:
            vr_summary_cells.append(f"{ok_counts[method]}/{totals[method]}")
            vr_summary_cells.append("")
            cost_summary_cells.append("")
            cost_summary_cells.append(rf"\${total_cost[method]:.2f}")
    lines.append(" & ".join(vr_summary_cells) + r" \\")
    lines.append(" & ".join(cost_summary_cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        r"\end{document}",
    ]

    tex_path.write_text("\n".join(lines) + "\n")
    console.print(f"[green]Saved:[/] {tex_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a per-scenario LaTeX summary table from a plot CSV: "
            "one row per panel, VR+Cost columns per method, VR italicized "
            "if under the scenario's SLO threshold."
        )
    )
    parser.add_argument(
        "plots_dir",
        help=(
            "Path to a data/plots/<name> subdirectory (or just <name>) "
            "containing the plot CSV to summarize."
        ),
    )
    live_group = parser.add_mutually_exclusive_group()
    live_group.add_argument(
        "--live", action="store_true", help="Use the '#live' CSV."
    )
    live_group.add_argument(
        "--sim", action="store_true", help="Use the (non-live) simulation CSV."
    )
    args = parser.parse_args()

    target = Path(args.plots_dir)
    if not target.exists():
        target = pu.get_data_dir() / "plots" / args.plots_dir

    live_flag = True if args.live else (False if args.sim else None)
    csv_path = _resolve_csv_path(target, live_flag)
    dir_name = csv_path.parent.name

    thresholds = _load_panel_thresholds(dir_name)
    try:
        panel_rows, method_order = _load_panel_rows(csv_path, thresholds)
        if not panel_rows:
            console.print(f"[yellow]No data rows found in {csv_path}[/]")
            return
        threshold = _require_common_threshold(panel_rows)
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)

    tex_path = csv_path.with_name(csv_path.stem + "_summary.tex")
    _write_latex_table(tex_path, panel_rows, method_order, threshold)


if __name__ == "__main__":
    main()
