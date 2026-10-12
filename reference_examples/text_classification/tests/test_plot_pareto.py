from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from plot_pareto import Point, load_points, pareto_front, render_svg


class ParetoPlotTests(unittest.TestCase):
    def test_macro_average_and_complete_system_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "task-a": {
                            "shared": {"accuracy": 0.8, "memory_context_chars": 100},
                            "incomplete": {"accuracy": 1.0, "memory_context_chars": 1},
                        },
                        "task-b": {
                            "shared": {"accuracy": 0.6, "memory_context_chars": 300}
                        },
                    }
                ),
                encoding="utf-8",
            )

            points, datasets = load_points(summary)

        self.assertEqual(datasets, ["task-a", "task-b"])
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].accuracy, 70.0)
        self.assertEqual(points[0].context_chars, 200.0)

    def test_pareto_front_prefers_accuracy_and_smaller_context(self) -> None:
        points = [
            Point("accurate", 90, 1000),
            Point("compact", 80, 100),
            Point("dominated", 70, 1200),
        ]

        self.assertEqual(
            [point.name for point in pareto_front(points)],
            ["compact", "accurate"],
        )

    def test_svg_contains_methods_and_frontier_description(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "plot.svg"
            render_svg(
                [
                    Point("no_memory", 50, 100, is_baseline=True),
                    Point("candidate", 60, 200),
                ],
                ["task"],
                output,
                "Test plot",
            )
            svg = output.read_text(encoding="utf-8")

        self.assertIn("candidate", svg)
        self.assertIn("Pareto-optimal", svg)
        self.assertIn("Accuracy (%)", svg)

    def test_y_ticks_stay_inside_plot_domain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "plot.svg"
            render_svg(
                [Point("low", 40, 100), Point("high", 55, 200)],
                ["task"],
                output,
                "Bounded ticks",
            )
            svg = output.read_text(encoding="utf-8")

        self.assertNotIn('class="tick">30</text>', svg)
        self.assertNotIn('class="tick">60</text>', svg)

    def test_proposals_are_colored_by_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.json"
            summary.write_text(
                json.dumps({"D": {
                    "no_memory": {"accuracy": 0.5, "memory_context_chars": 10},
                    "first": {"accuracy": 0.6, "memory_context_chars": 20},
                    "second": {"accuracy": 0.7, "memory_context_chars": 30},
                }}),
                encoding="utf-8",
            )
            (root / "evolution_summary.jsonl").write_text(
                '{"iteration": 1, "system": "first"}\n'
                '{"iteration": 2, "system": "second"}\n',
                encoding="utf-8",
            )
            points, datasets = load_points(summary)
            output = root / "plot.svg"
            render_svg(points, datasets, output, "Test")
            svg = output.read_text(encoding="utf-8")

        self.assertIn("Iteration 1", svg)
        self.assertIn("Iteration 2", svg)
        self.assertIn("first  [iter 1]", svg)
        self.assertIn("second  [iter 2]", svg)
        self.assertIn('fill="#2563eb"', svg)
        self.assertIn('fill="#16a34a"', svg)


if __name__ == "__main__":
    unittest.main()
