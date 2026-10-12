"""Plot validation accuracy against memory context size for one experiment run.

The script uses only the Python standard library and writes an SVG file.
For runs containing multiple datasets, each system is represented by its
unweighted macro-average accuracy and average memory-context character count.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path


@dataclass(frozen=True)
class Point:
    name: str
    accuracy: float
    context_chars: float
    is_baseline: bool = False
    iteration: int | None = None


def load_points(
    summary_path: Path,
    datasets: list[str] | None = None,
    baselines: set[str] | None = None,
) -> tuple[list[Point], list[str]]:
    """Load complete systems from summary.json and macro-average datasets."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = datasets or list(summary)
    unknown = [dataset for dataset in selected if dataset not in summary]
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {', '.join(unknown)}. "
            f"Available: {', '.join(summary)}"
        )

    names = set.intersection(
        *(set(summary[dataset]) for dataset in selected)
    ) if selected else set()
    baseline_names = baselines or {"no_memory", "fewshot_all"}
    iteration_by_name: dict[str, int] = {}
    evolution_path = summary_path.with_name("evolution_summary.jsonl")
    if evolution_path.exists():
        for line in evolution_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("system") and row.get("iteration") is not None:
                iteration_by_name[row["system"]] = int(row["iteration"])
    points = []
    for name in sorted(names):
        rows = [summary[dataset][name] for dataset in selected]
        accuracies = [float(row["accuracy"]) * 100 for row in rows]
        contexts = [float(row.get("memory_context_chars", 0)) for row in rows]
        positive_contexts = [value for value in contexts if value > 0]
        points.append(
            Point(
                name=name,
                accuracy=sum(accuracies) / len(accuracies),
                context_chars=(
                    sum(positive_contexts) / len(positive_contexts)
                    if positive_contexts
                    else 0
                ),
                is_baseline=name in baseline_names,
                iteration=iteration_by_name.get(name),
            )
        )
    if not points:
        raise ValueError("No systems have complete results for all selected datasets")
    return points, selected


def pareto_front(points: list[Point]) -> list[Point]:
    """Return non-dominated points (higher accuracy, lower context is better)."""
    frontier = [
        point
        for point in points
        if not any(
            other.accuracy >= point.accuracy
            and other.context_chars <= point.context_chars
            and (
                other.accuracy > point.accuracy
                or other.context_chars < point.context_chars
            )
            for other in points
        )
    ]
    return sorted(frontier, key=lambda point: (point.context_chars, point.accuracy))


def _linear_ticks(low: float, high: float, count: int = 6) -> list[float]:
    span = max(high - low, 1e-9)
    raw = span / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    step = next(value for value in (1, 2, 5, 10) if value >= normalized) * magnitude
    start = math.floor(low / step) * step
    end = math.ceil(high / step) * step
    return [start + index * step for index in range(round((end - start) / step) + 1)]


def _log_ticks(low: float, high: float) -> list[float]:
    """Ticks for a log1p axis, including zero when it is in range."""
    ticks = {0.0}
    max_power = max(0, math.ceil(math.log10(max(high, 1))))
    for power in range(max_power + 1):
        base = 10**power
        for multiplier in (1, 2, 5):
            value = float(multiplier * base)
            if low <= value <= high:
                ticks.add(value)
    return sorted(value for value in ticks if low <= value <= high)


def render_svg(
    points: list[Point],
    datasets: list[str],
    output: Path,
    title: str,
    x_scale: str = "log",
) -> None:
    """Render points, a keyed method list, and the Pareto frontier as SVG."""
    frontier = pareto_front(points)
    frontier_names = {point.name for point in frontier}
    proposed = [point for point in points if not point.is_baseline]
    baselines = [point for point in points if point.is_baseline]
    ordered = baselines + sorted(proposed, key=lambda point: (-point.accuracy, point.context_chars))
    identifiers = {
        point.name: (f"B{index}" if point.is_baseline else f"P{index}")
        for group, prefix in ((baselines, "B"), (sorted(proposed, key=lambda p: (-p.accuracy, p.context_chars)), "P"))
        for index, point in enumerate(group, 1)
    }
    palette = (
        "#2563eb", "#16a34a", "#9333ea", "#ea580c", "#0891b2",
        "#db2777", "#65a30d", "#4f46e5", "#ca8a04", "#0f766e",
    )
    iteration_colors = {
        iteration: palette[index % len(palette)]
        for index, iteration in enumerate(
            sorted({point.iteration for point in proposed if point.iteration is not None})
        )
    }

    def point_color(point: Point) -> str:
        if point.is_baseline:
            return "#64748b"
        return iteration_colors.get(point.iteration, "#2563eb")

    width = 1420
    height = max(760, 230 + 25 * len(ordered))
    left, top, bottom = 105, 105, 95
    legend_width, gap = 430, 50
    chart_right = width - legend_width - gap
    chart_bottom = height - bottom
    chart_width = chart_right - left
    chart_height = chart_bottom - top

    x_values = [point.context_chars for point in points]
    if x_scale == "log":
        transform = lambda value: math.log10(value + 1)
        transformed = [transform(value) for value in x_values]
        pad = max((max(transformed) - min(transformed)) * 0.08, 0.08)
        tx_low, tx_high = min(transformed) - pad, max(transformed) + pad
        x_low = max(0.0, 10**tx_low - 1)
        x_high = max(x_values) * 1.12 + 1
        x_ticks = _log_ticks(x_low, x_high)
        x_axis_label = "Memory context (characters, log scale)"
    else:
        transform = float
        span = max(max(x_values) - min(x_values), 1)
        x_low = max(0.0, min(x_values) - span * 0.08)
        x_high = max(x_values) + span * 0.08
        tx_low, tx_high = transform(x_low), transform(x_high)
        x_ticks = _linear_ticks(x_low, x_high)
        x_axis_label = "Memory context (characters)"

    y_values = [point.accuracy for point in points]
    y_span = max(max(y_values) - min(y_values), 5)
    y_low = max(0.0, min(y_values) - y_span * 0.12)
    y_high = min(100.0, max(y_values) + y_span * 0.12)
    if y_high <= y_low:
        y_high = min(100.0, y_low + 5)
    y_ticks = [
        tick
        for tick in _linear_ticks(y_low, y_high)
        if y_low <= tick <= y_high
    ]

    def sx(value: float) -> float:
        return left + (transform(value) - tx_low) / (tx_high - tx_low) * chart_width

    def sy(value: float) -> float:
        return chart_bottom - (value - y_low) / (y_high - y_low) * chart_height

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'role="img" aria-labelledby="plot-title plot-desc">',
        f'<title id="plot-title">{escape(title)}</title>',
        '<desc id="plot-desc">Accuracy versus memory context characters for all methods. '
        'The red connected points form the Pareto frontier.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#172033}'
        '.tick{font-size:13px;fill:#526070}.axis-title{font-size:15px;font-weight:600}'
        '.method{font-size:13px}.small{font-size:12px;fill:#667085}</style>',
        f'<text x="{left}" y="42" font-size="24" font-weight="600">{escape(title)}</text>',
        f'<text x="{left}" y="68" class="small">Datasets: {escape(", ".join(datasets))} · '
        'Higher accuracy and lower context are better</text>',
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" '
        'fill="none" stroke="#9aa4b2" stroke-width="1"/>',
    ]

    for tick in y_ticks:
        y = sy(tick)
        lines.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{chart_right}" y2="{y:.2f}" '
                'stroke="#e4e7ec" stroke-width="1"/>',
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" class="tick">{tick:g}</text>',
            ]
        )
    for tick in x_ticks:
        if not (x_low <= tick <= x_high):
            continue
        x = sx(tick)
        lines.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{chart_bottom}" '
                'stroke="#eef0f3" stroke-width="1"/>',
                f'<text x="{x:.2f}" y="{chart_bottom + 24}" text-anchor="middle" class="tick">{tick:,.0f}</text>',
            ]
        )

    lines.extend(
        [
            f'<text x="{(left + chart_right) / 2:.2f}" y="{height - 33}" '
            f'text-anchor="middle" class="axis-title">{escape(x_axis_label)}</text>',
            f'<text x="28" y="{(top + chart_bottom) / 2:.2f}" text-anchor="middle" '
            'transform="rotate(-90 28 '
            f'{(top + chart_bottom) / 2:.2f})" class="axis-title">Accuracy (%)</text>',
        ]
    )

    if len(frontier) > 1:
        path = " ".join(
            ("M" if index == 0 else "L") + f" {sx(point.context_chars):.2f} {sy(point.accuracy):.2f}"
            for index, point in enumerate(frontier)
        )
        lines.append(
            f'<path d="{path}" fill="none" stroke="#dc2626" stroke-width="2.5"/>'
        )

    for point in points:
        x, y = sx(point.context_chars), sy(point.accuracy)
        identifier = identifiers[point.name]
        tooltip = (
            f"{point.name}: {point.accuracy:.1f}% accuracy, "
            f"{point.context_chars:,.0f} context chars"
        )
        if point.iteration is not None:
            tooltip += f", iteration {point.iteration}"
        if point.is_baseline:
            diamond = f"{x:.2f},{y - 7:.2f} {x + 7:.2f},{y:.2f} {x:.2f},{y + 7:.2f} {x - 7:.2f},{y:.2f}"
            lines.append(
                f'<polygon points="{diamond}" fill="#64748b" stroke="#ffffff" stroke-width="1.5"><title>{escape(tooltip)}</title></polygon>'
            )
        else:
            lines.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6.5" fill="{point_color(point)}" '
                f'stroke="#ffffff" stroke-width="1.5"><title>{escape(tooltip)}</title></circle>'
            )
        if point.name in frontier_names:
            lines.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="10" fill="none" '
                'stroke="#dc2626" stroke-width="2"/>'
            )
        lines.append(
            f'<text x="{x + 11:.2f}" y="{y - 9:.2f}" font-size="12" font-weight="600">{escape(identifier)}</text>'
        )

    legend_x = chart_right + 42
    lines.extend(
        [
            f'<text x="{legend_x}" y="{top}" font-size="16" font-weight="600">Methods</text>',
            f'<text x="{legend_x}" y="{top + 24}" class="small">Red ring = Pareto-optimal</text>',
        ]
    )
    if iteration_colors:
        iteration_legend = "  ".join(
            f'<tspan fill="{color}">● Iteration {iteration}</tspan>'
            for iteration, color in iteration_colors.items()
        )
        lines.append(
            f'<text x="{legend_x}" y="{top + 43}" class="small">{iteration_legend}</text>'
        )
    for index, point in enumerate(ordered):
        y = top + 72 + index * 25
        identifier = identifiers[point.name]
        marker = "◆" if point.is_baseline else "●"
        color = point_color(point)
        weight = "600" if point.name in frontier_names else "400"
        lines.append(
            f'<text x="{legend_x}" y="{y}" class="method" fill="{color}" font-weight="{weight}">'
            f'{marker} {escape(identifier)}  {escape(point.name)}'
            f'{"  [iter " + str(point.iteration) + "]" if point.iteration is not None else ""}</text>'
        )
        lines.append(
            f'<text x="{width - 25}" y="{y}" text-anchor="end" class="small">'
            f'{point.accuracy:.1f}% · {point.context_chars:,.0f} ch</text>'
        )

    lines.append('</svg>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot accuracy versus memory context and its Pareto frontier"
    )
    parser.add_argument("--run-name", required=True, help="Directory name under logs/")
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset to include; repeat for multiple. Default: all in summary.json",
    )
    parser.add_argument(
        "--x-scale", choices=("log", "linear"), default="log"
    )
    parser.add_argument("--output", type=Path, help="Output SVG path")
    parser.add_argument("--title", help="Plot title")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    run_dir = base / "logs" / args.run_name
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        parser.error(f"summary.json not found: {summary_path}")

    points, selected = load_points(summary_path, datasets=args.datasets)
    output = args.output or run_dir / "pareto_plot.svg"
    title = args.title or f"Harness accuracy–context trade-off · {args.run_name}"
    render_svg(points, selected, output, title, x_scale=args.x_scale)

    print(f"Wrote {output}")
    print("Pareto frontier:")
    for point in pareto_front(points):
        print(
            f"  {point.name}: accuracy={point.accuracy:.1f}%, "
            f"context={point.context_chars:,.0f} chars"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
