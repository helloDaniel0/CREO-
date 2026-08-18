#!/usr/bin/env python3
"""Generate two compact CREO+ PDFs for a side-by-side IEEE column layout."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from plot_creo_linux_results import (
    DEFAULT_RESULTS,
    PROJECT_ROOT,
    PostScriptFigure,
    convert_to_pdf,
    read_series,
    step_points,
)


THROUGHPUT_OUTPUT = PROJECT_ROOT / "CREO_plus_Linux_Throughput.pdf"
RTT_OUTPUT = PROJECT_ROOT / "CREO_plus_Linux_RTT.pdf"

# Two 123 pt figures plus a 6 pt gap fit a 3.5 inch IEEE column exactly.
FIGURE_WIDTH = 123.0
FIGURE_HEIGHT = 100.0
PLOT_LEFT = 25.5
PLOT_RIGHT = 118.5
PLOT_BOTTOM = 18.5
PLOT_TOP = 83.0
X_MAX = 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--throughput-output", type=Path, default=THROUGHPUT_OUTPUT)
    parser.add_argument("--rtt-output", type=Path, default=RTT_OUTPUT)
    return parser.parse_args()


def x_coord(time_s: float) -> float:
    return PLOT_LEFT + (PLOT_RIGHT - PLOT_LEFT) * time_s / X_MAX


def y_coord(value: float, minimum: float, maximum: float) -> float:
    return PLOT_BOTTOM + (PLOT_TOP - PLOT_BOTTOM) * (value - minimum) / (maximum - minimum)


def begin_clipped_plot(figure: PostScriptFigure) -> None:
    figure.add("gsave")
    figure.add(
        f"newpath {PLOT_LEFT:.3f} {PLOT_BOTTOM:.3f} moveto "
        f"{PLOT_RIGHT:.3f} {PLOT_BOTTOM:.3f} lineto "
        f"{PLOT_RIGHT:.3f} {PLOT_TOP:.3f} lineto "
        f"{PLOT_LEFT:.3f} {PLOT_TOP:.3f} lineto closepath clip"
    )


def draw_axes(
    figure: PostScriptFigure,
    y_ticks: list[float],
    y_min: float,
    y_max: float,
    y_label: str,
) -> None:
    figure.color(0.80, 0.80, 0.80)
    figure.line_width(0.32)
    figure.add("[1.1 1.7] 0 setdash")
    for tick in y_ticks:
        y = y_coord(tick, y_min, y_max)
        figure.line(PLOT_LEFT, y, PLOT_RIGHT, y)
    for tick in (0.0, 20.0, 40.0, 60.0):
        x = x_coord(tick)
        figure.line(x, PLOT_BOTTOM, x, PLOT_TOP)
    figure.add("[] 0 setdash")

    figure.color(0.0, 0.0, 0.0)
    figure.line_width(0.62)
    figure.add(
        f"newpath {PLOT_LEFT:.3f} {PLOT_BOTTOM:.3f} moveto "
        f"{PLOT_RIGHT:.3f} {PLOT_BOTTOM:.3f} lineto "
        f"{PLOT_RIGHT:.3f} {PLOT_TOP:.3f} lineto "
        f"{PLOT_LEFT:.3f} {PLOT_TOP:.3f} lineto closepath stroke"
    )

    figure.font("Times-Roman", 7.2)
    for tick in (0.0, 20.0, 40.0, 60.0):
        x = x_coord(tick)
        figure.line(x, PLOT_BOTTOM, x, PLOT_BOTTOM - 2.2)
        figure.text(x, 10.3, f"{tick:g}", align="center")
    for tick in y_ticks:
        y = y_coord(tick, y_min, y_max)
        figure.line(PLOT_LEFT, y, PLOT_LEFT - 2.2, y)
        figure.text(PLOT_LEFT - 2.8, y - 2.25, f"{tick:g}", align="right")

    figure.font("Times-Bold", 7.8)
    figure.text((PLOT_LEFT + PLOT_RIGHT) / 2.0, 2.6, "Time (s)", align="center")
    figure.text(8.0, (PLOT_BOTTOM + PLOT_TOP) / 2.0, y_label, align="center", rotation=90.0)


def draw_legend_entry(
    figure: PostScriptFigure,
    x: float,
    label: str,
    color: tuple[float, float, float],
    dashed: bool = False,
) -> None:
    figure.color(*color)
    figure.line_width(1.05)
    figure.add("[3.2 1.8] 0 setdash" if dashed else "[] 0 setdash")
    figure.line(x, 93.3, x + 10.0, 93.3)
    figure.add("[] 0 setdash")
    figure.color(0.0, 0.0, 0.0)
    figure.font("Times-Roman", 6.7)
    figure.text(x + 12.5, 91.15, label)


def draw_throughput_figure(
    capacity: list[tuple[float, float]],
    throughput: list[tuple[float, float]],
    eps_path: Path,
) -> None:
    figure = PostScriptFigure(FIGURE_WIDTH, FIGURE_HEIGHT)
    capacity_steps = step_points(capacity, X_MAX)
    capacity_path = [(x_coord(t), y_coord(v, 0.0, 40.0)) for t, v in capacity_steps]
    throughput_path = [(x_coord(t), y_coord(v, 0.0, 40.0)) for t, v in throughput]

    begin_clipped_plot(figure)
    capacity_fill = [
        (x_coord(0.0), PLOT_BOTTOM),
        *capacity_path,
        (x_coord(X_MAX), PLOT_BOTTOM),
    ]
    figure.color(0.875, 0.925, 0.965)
    figure.polygon(capacity_fill)
    figure.add("grestore")

    draw_axes(figure, [0.0, 10.0, 20.0, 30.0, 40.0], 0.0, 40.0, "Rate (Mbps)")

    begin_clipped_plot(figure)
    figure.color(0.20, 0.46, 0.70)
    figure.line_width(0.90)
    figure.polyline(capacity_path)
    figure.color(0.78, 0.16, 0.13)
    figure.line_width(1.05)
    figure.polyline(throughput_path)
    figure.add("grestore")

    draw_legend_entry(figure, 12.0, "Capacity", (0.20, 0.46, 0.70))
    draw_legend_entry(figure, 64.0, "Rx Throughput", (0.78, 0.16, 0.13))
    figure.write(eps_path)


def draw_rtt_figure(
    rtt: list[tuple[float, float]],
    base_rtt: list[tuple[float, float]],
    eps_path: Path,
) -> None:
    figure = PostScriptFigure(FIGURE_WIDTH, FIGURE_HEIGHT)
    rtt_path = [(x_coord(t), y_coord(v, 15.0, 35.0)) for t, v in rtt]
    base_path = [
        (x_coord(t), y_coord(v, 15.0, 35.0))
        for t, v in step_points(base_rtt, X_MAX)
    ]

    draw_axes(figure, [15.0, 20.0, 25.0, 30.0, 35.0], 15.0, 35.0, "RTT (ms)")

    begin_clipped_plot(figure)
    figure.color(0.90, 0.00, 0.42)
    figure.line_width(0.95)
    figure.polyline(rtt_path)
    figure.color(0.31, 0.18, 0.52)
    figure.line_width(0.95)
    figure.add("[3.2 1.8] 0 setdash")
    figure.polyline(base_path)
    figure.add("[] 0 setdash")
    figure.add("grestore")

    draw_legend_entry(figure, 22.0, "TCP RTT", (0.90, 0.00, 0.42))
    draw_legend_entry(figure, 68.5, "Base RTT", (0.31, 0.18, 0.52), dashed=True)
    figure.write(eps_path)


def main() -> None:
    args = parse_args()
    results = args.results.resolve()
    throughput_output = args.throughput_output.resolve()
    rtt_output = args.rtt_output.resolve()

    capacity = read_series(results / "realbw.dat")
    throughput = read_series(results / "throughput.dat")
    rtt = read_series(results / "rtt.dat")
    base_rtt = read_series(results / "base-rtt.dat")

    with tempfile.TemporaryDirectory(prefix="creo-linux-split-") as temp_dir:
        temp_path = Path(temp_dir)
        throughput_eps = temp_path / "throughput.eps"
        rtt_eps = temp_path / "rtt.eps"
        draw_throughput_figure(capacity, throughput, throughput_eps)
        draw_rtt_figure(rtt, base_rtt, rtt_eps)
        convert_to_pdf(throughput_eps, throughput_output)
        convert_to_pdf(rtt_eps, rtt_output)

    print(f"Generated {throughput_output}")
    print(f"Generated {rtt_output}")


if __name__ == "__main__":
    main()
