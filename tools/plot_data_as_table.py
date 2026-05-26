from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


_PANEL_RE = re.compile(r"λ=([^,]+),\s*κ=([^,]+),\s*C=(.+)")


def _parse_panel(key: str) -> tuple[str, str, str]:
    """Split a panel title of the form 'λ=X, κ=Y, C=Z' into its three parts."""
    m = _PANEL_RE.match(key)
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    return key, "", ""


def _tex_escape(s: str) -> str:
    """Escape special LaTeX characters in a plain-text string."""
    for char, repl in [
        ("\\", r"\textbackslash{}"),
        ("&",  r"\&"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]:
        s = s.replace(char, repl)
    return s



def _build_latex(
    x_metric: str,
    panel_order: list[str],
    method_order: list[str],
    panels: dict[str, dict[str, tuple[float, float]]],
    reference: str | None,
) -> str:
    # Two sub-columns per method (VR, Cost); methods separated by ||
    col_parts = ["c", "c", "c", "||"]
    for i in range(len(method_order)):
        if i < len(method_order) - 1:
            col_parts += ["c", "c", "|"]
        else:
            col_parts += ["c", "c"]
    col_spec = " ".join(col_parts)

    lines: list[str] = []
    lines.append(r"\documentclass{article}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{amsmath}")
    lines.append(r"\usepackage[landscape, margin=1cm]{geometry}")
    lines.append(r"\begin{document}")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\columnwidth}{!}{")
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")

    # Top header: method names spanning 2 cols each
    top_header: list[str] = [r"$\lambda$", r"$\kappa$", "C"]
    for i, method in enumerate(method_order):
        mc_fmt = "c|" if i < len(method_order) - 1 else "c"
        top_header.append(rf"\multicolumn{{2}}{{{mc_fmt}}}{{{_tex_escape(method)}}}")
    lines.append(" & ".join(top_header) + r" \\")

    # Sub-header: VR and Cost labels
    sub_header: list[str] = ["", "", ""]
    for _ in method_order:
        sub_header += ["VR", "Cost"]
    lines.append(" & ".join(sub_header) + r" \\")
    lines.append(r"\hline")

    for panel_key in panel_order:
        lam, kap, c = _parse_panel(panel_key)
        present = [
            (m, panels[panel_key][m])
            for m in method_order
            if m in panels[panel_key]
        ]
        ranked = sorted(present, key=lambda t: t[1])  # lex (x, y)
        best   = ranked[0][0] if len(ranked) >= 1 else None
        second = ranked[1][0] if len(ranked) >= 2 else None
        cells: list[str] = [_tex_escape(lam), _tex_escape(kap), _tex_escape(c)]
        for method in method_order:
            if method in panels[panel_key]:
                x, y = panels[panel_key][method]
                vr_cell = f"{x:.4f}"
                cost_cell = rf"\${y:.2f}"
                if method == best:
                    vr_cell = rf"\textbf{{{vr_cell}}}"
                elif method == second:
                    vr_cell = rf"\underline{{{vr_cell}}}"
                cells.append(vr_cell)
                cells.append(cost_cell)
            else:
                cells += ["---", "---"]
        lines.append(" & ".join(cells) + r" \\")

    # Mean improvement row
    if reference and reference in method_order:
        # First pass: compute mean diffs for every non-reference method
        mean_vr: dict[str, float | None] = {}
        mean_cost: dict[str, float | None] = {}
        for method in method_order:
            if method == reference:
                mean_vr[method] = None
                mean_cost[method] = None
                continue
            x_diffs: list[float] = []
            y_diffs: list[float] = []
            for points in panels.values():
                if reference not in points or method not in points:
                    continue
                ref_x, ref_y = points[reference]
                x, y = points[method]
                if ref_x != 0:
                    x_diffs.append((x - ref_x) / ref_x * 100)
                if ref_y != 0:
                    y_diffs.append((y - ref_y) / ref_y * 100)
            mean_vr[method] = sum(x_diffs) / len(x_diffs) if x_diffs else None
            mean_cost[method] = sum(y_diffs) / len(y_diffs) if y_diffs else None

        # Rank non-reference methods by mean VR (ascending = most improvement)
        ranked_mean = sorted(
            [m for m in method_order if m != reference and mean_vr[m] is not None],
            key=lambda m: mean_vr[m],  # type: ignore[arg-type]
        )
        mean_best   = ranked_mean[0] if len(ranked_mean) >= 1 else None
        mean_second = ranked_mean[1] if len(ranked_mean) >= 2 else None

        lines.append(r"\hline")
        mean_cells: list[str] = [r"\multicolumn{3}{c||}{\textit{Mean~$\Delta$}}"]
        for method in method_order:
            if method == reference:
                mean_cells += ["---", "---"]
            else:
                vr_val = mean_vr[method]
                vr_str = rf"{vr_val:+.1f}\%" if vr_val is not None else "---"
                if method == mean_best:
                    vr_str = rf"\textbf{{{vr_str}}}"
                elif method == mean_second:
                    vr_str = rf"\underline{{{vr_str}}}"
                cost_val = mean_cost[method]
                cost_str = rf"{cost_val:+.1f}\%" if cost_val is not None else "---"
                mean_cells.append(vr_str)
                mean_cells.append(cost_str)
        lines.append(" & ".join(mean_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    # Caption: only show a short description if reference is given, otherwise TODO
    if reference:
        caption = f"Mean improvement is relative to {reference}."
    else:
        caption = "TODO"
    lines.append(rf"\caption{{{caption}}}")
    lines.append(r"\label{tab:TODO}")
    lines.append(r"\end{table}")
    lines.append(r"\end{document}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write a LaTeX tabular grid of a plotter-output CSV to a "
            "standalone .tex file: one row per panel, one column per method."
        )
    )
    parser.add_argument("csv_path", type=Path, help="Path to the plotter CSV.")
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        metavar="LABEL",
        help=(
            "Label of the reference method.  When given, the caption "
            "summarises each method's mean relative difference from the "
            "reference across all panels."
        ),
    )
    args = parser.parse_args()

    with args.csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("CSV is empty.")
        return

    x_metric: str = rows[0]["x_metric"]

    panels: dict[str, dict[str, tuple[float, float]]] = {}
    panel_order: list[str] = []
    method_order: list[str] = []
    seen_methods: set[str] = set()

    for row in rows:
        panel_key = row["panel_title"] or f"({row['row']},{row['col']})"
        if panel_key not in panels:
            panels[panel_key] = {}
            panel_order.append(panel_key)
        panels[panel_key][row["label"]] = (float(row["x"]), float(row["y"]))
        if row["label"] not in seen_methods:
            method_order.append(row["label"])
            seen_methods.add(row["label"])

    latex = _build_latex(x_metric, panel_order, method_order, panels, args.reference)
    out_path = args.csv_path.with_suffix(".tex")
    out_path.write_text(latex + "\n")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
