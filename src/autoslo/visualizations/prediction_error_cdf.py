from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	sorted_values = np.sort(values)
	cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
	return sorted_values, cdf


def _sorted_groups(values: Iterable[Any]) -> list[Any]:
	groups = list(values)
	try:
		return sorted(groups, key=lambda x: float(x))
	except (TypeError, ValueError):
		return sorted(groups, key=str)


def plot_grouped_cdf(
	ax: Axes,
	data: pd.DataFrame,
	value_col: str,
	group_col: str,
	*,
	palette: Mapping[Any, str] | None = None,
	title: str | None = None,
	xlabel: str | None = None,
	ylabel: str = "Cumulative probability",
	log_x: bool = True,
	include_overall: bool = True,
	overall_label: str = "Overall",
	perfect_x: float | None = 1.0,
	perfect_label: str = "Perfect",
	line_width: float = 1.8,
	legend_fontsize: int = 7,
	legend_title: str = "RPU",
	show_legend: bool = True,
) -> None:
	"""Plot grouped ECDF curves with optional overall/perfect overlays."""
	required_cols = [value_col, group_col]
	filtered = data[required_cols].dropna().copy()
	filtered[value_col] = pd.to_numeric(filtered[value_col], errors="coerce")
	filtered = filtered.dropna(subset=[value_col])

	if log_x:
		filtered = filtered[filtered[value_col] > 0]

	if filtered.empty:
		ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
		if title:
			ax.set_title(title)
		return

	groups = _sorted_groups(filtered[group_col].unique().tolist())
	for group in groups:
		sub = filtered[filtered[group_col] == group][value_col].to_numpy(dtype=float)
		if len(sub) == 0:
			continue
		xs, ys = _ecdf(sub)
		color = palette.get(group, "black") if palette else None
		ax.plot(
			xs,
			ys,
			color=color,
			lw=line_width,
			label=str(group),
		)

	if include_overall:
		all_vals = filtered[value_col].to_numpy(dtype=float)
		if len(all_vals) > 0:
			xs, ys = _ecdf(all_vals)
			ax.plot(
				xs,
				ys,
				color="black",
				lw=max(2.0, line_width + 0.2),
				label=overall_label,
			)

	if perfect_x is not None:
		ax.axvline(
			perfect_x,
			color="0.35",
			linestyle="--",
			linewidth=1.1,
			label=perfect_label,
		)

	if log_x:
		ax.set_xscale("log")

	ax.set_ylim(0, 1.02)
	if title:
		ax.set_title(title)
	if xlabel:
		ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.grid(True, which="major", linestyle="-", alpha=0.25)
	ax.grid(True, which="minor", linestyle=":", alpha=0.15)

	if show_legend:
		legend = ax.legend(fontsize=legend_fontsize)
		if legend is not None and legend_title:
			legend.set_title(legend_title)


def build_percentile_summary_lines(
	data: pd.DataFrame,
	group_col: str,
	value_col: str,
	*,
	quantiles: Sequence[float] = (0.50, 0.90, 0.95),
	include_overall: bool = True,
	group_header: str = "RPU",
	include_n: bool = True,
) -> list[str]:
	"""Build monospace-friendly percentile summary lines grouped by RPU."""
	group_width = max(3, len(group_header), len("ALL"))
	quantile_headers = [f"P{int(q * 100):02d}" for q in quantiles]
	header = f"{group_header:>{group_width}}  " + "  ".join(
		f"{h:>5}" for h in quantile_headers
	)
	if include_n:
		header += "     N"
	lines = [header]

	filtered = data[[group_col, value_col]].dropna().copy()
	filtered[value_col] = pd.to_numeric(filtered[value_col], errors="coerce")
	filtered = filtered.dropna(subset=[value_col])
	groups = _sorted_groups(filtered[group_col].unique().tolist())

	for group in groups:
		vals = filtered.loc[filtered[group_col] == group, value_col]
		qs = vals.quantile(list(quantiles))
		line = (
			f"{str(group):>{group_width}}  "
			+ "  ".join(f"{float(qs.loc[q]):>5.2f}" for q in quantiles)
		)
		if include_n:
			line += f"  {len(vals):>4}"
		lines.append(line)

	if include_overall and not filtered.empty:
		qs_all = filtered[value_col].quantile(list(quantiles))
		line = (
			f"{'ALL':>{group_width}}  "
			+ "  ".join(f"{float(qs_all.loc[q]):>5.2f}" for q in quantiles)
		)
		if include_n:
			line += f"  {len(filtered):>4}"
		lines.append(line)

	return lines


def build_direction_summary_lines(
	data: pd.DataFrame,
	group_col: str,
	actual_col: str,
	predicted_col: str,
	*,
	include_overall: bool = True,
	group_header: str = "RPU",
) -> list[str]:
	"""Build under/over prediction rate summary lines grouped by RPU."""
	group_width = max(3, len(group_header), len("ALL"))
	lines = [f"{group_header:>{group_width}}    Under     Over     N"]

	filtered = data[[group_col, actual_col, predicted_col]].dropna().copy()
	filtered[actual_col] = pd.to_numeric(filtered[actual_col], errors="coerce")
	filtered[predicted_col] = pd.to_numeric(filtered[predicted_col], errors="coerce")
	filtered = filtered.dropna(subset=[actual_col, predicted_col])

	groups = _sorted_groups(filtered[group_col].unique().tolist())
	for group in groups:
		sub = filtered[filtered[group_col] == group]
		under = (sub[actual_col] > sub[predicted_col]).mean() * 100
		over = (sub[actual_col] < sub[predicted_col]).mean() * 100
		lines.append(
			f"{str(group):>{group_width}}  {under:>6.2f}%  {over:>6.2f}%  {len(sub):>4}"
		)

	if include_overall and not filtered.empty:
		under_all = (filtered[actual_col] > filtered[predicted_col]).mean() * 100
		over_all = (filtered[actual_col] < filtered[predicted_col]).mean() * 100
		lines.append(
			f"{'ALL':>{group_width}}  {under_all:>6.2f}%  {over_all:>6.2f}%  {len(filtered):>4}"
		)

	return lines


def add_monospace_summary_box(
	ax: Axes,
	lines: Sequence[str],
	*,
	x: float = 0.97,
	y: float = 0.04,
	fontsize: int = 9,
) -> None:
	"""Add a compact summary table as a monospace annotation box."""
	ax.text(
		x,
		y,
		"\n".join(lines),
		transform=ax.transAxes,
		ha="right",
		va="bottom",
		fontsize=fontsize,
		family="monospace",
		bbox={
			"boxstyle": "round,pad=0.45",
			"facecolor": "white",
			"edgecolor": "0.8",
			"alpha": 0.85,
		},
	)
