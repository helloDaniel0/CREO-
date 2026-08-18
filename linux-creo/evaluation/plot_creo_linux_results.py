#!/usr/bin/env python3
"""Draw the real Linux CREO+ time series as a publication-style PDF."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "variable-capacity"
DEFAULT_OUTPUT = PROJECT_ROOT / "CREO_plus_Linux_Fig16.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_series(path: Path) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"{path}:{line_number}: expected at least two columns")
        points.append((float(fields[0]), float(fields[1])))
    if not points:
        raise ValueError(f"{path}: no data points")
    return points


def ps_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class PostScriptFigure:
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self.commands: list[str] = [
            "%!PS-Adobe-3.0 EPSF-3.0",
            f"%%BoundingBox: 0 0 {math.ceil(width)} {math.ceil(height)}",
            "%%LanguageLevel: 2",
            "%%Pages: 1",
            "%%EndComments",
            "1 setlinejoin 1 setlinecap",
            "1 1 1 setrgbcolor",
            f"newpath 0 0 moveto {width:.3f} 0 lineto {width:.3f} {height:.3f} lineto "
            f"0 {height:.3f} lineto closepath fill",
        ]

    def add(self, command: str) -> None:
        self.commands.append(command)

    def color(self, red: float, green: float, blue: float) -> None:
        self.add(f"{red:.4f} {green:.4f} {blue:.4f} setrgbcolor")

    def line_width(self, width: float) -> None:
        self.add(f"{width:.3f} setlinewidth")

    def font(self, name: str, size: float) -> None:
        self.add(f"/{name} findfont {size:.3f} scalefont setfont")

    def line(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.add(f"newpath {x1:.3f} {y1:.3f} moveto {x2:.3f} {y2:.3f} lineto stroke")

    def polyline(self, points: list[tuple[float, float]]) -> None:
        if len(points) < 2:
            return
        commands = [f"newpath {points[0][0]:.3f} {points[0][1]:.3f} moveto"]
        commands.extend(f"{x:.3f} {y:.3f} lineto" for x, y in points[1:])
        commands.append("stroke")
        self.add(" ".join(commands))

    def polygon(self, points: list[tuple[float, float]]) -> None:
        if len(points) < 3:
            return
        commands = [f"newpath {points[0][0]:.3f} {points[0][1]:.3f} moveto"]
        commands.extend(f"{x:.3f} {y:.3f} lineto" for x, y in points[1:])
        commands.append("closepath fill")
        self.add(" ".join(commands))

    def text(
        self,
        x: float,
        y: float,
        value: str,
        align: str = "left",
        rotation: float = 0.0,
    ) -> None:
        escaped = ps_text(value)
        offsets = {
            "left": "",
            "center": "dup stringwidth pop -2 div 0 rmoveto ",
            "right": "dup stringwidth pop neg 0 rmoveto ",
        }
        if align not in offsets:
            raise ValueError(f"unknown text alignment: {align}")
        self.add(
            f"gsave {x:.3f} {y:.3f} translate {rotation:.3f} rotate 0 0 moveto "
            f"({escaped}) {offsets[align]}show grestore"
        )

    def write(self, path: Path) -> None:
        path.write_text("\n".join([*self.commands, "showpage", "%%EOF", ""]))


def step_points(
    series: list[tuple[float, float]],
    end_time: float,
) -> list[tuple[float, float]]:
    points = [series[0]]
    for time_s, value in series[1:]:
        points.append((time_s, points[-1][1]))
        points.append((time_s, value))
    points.append((end_time, series[-1][1]))
    return points


def draw_figure(
    capacity: list[tuple[float, float]],
    throughput: list[tuple[float, float]],
    rtt: list[tuple[float, float]],
    base_rtt: list[tuple[float, float]],
    eps_path: Path,
) -> None:
    width, height = 540.0, 292.0
    left, right, bottom, top = 66.0, 473.0, 50.0, 250.0
    plot_width, plot_height = right - left, top - bottom

    capacity_period = capacity[1][0] - capacity[0][0] if len(capacity) > 1 else 0.5
    end_time = max(
        capacity[-1][0] + capacity_period,
        throughput[-1][0] + capacity_period / 2.0,
        rtt[-1][0] + capacity_period / 2.0,
        base_rtt[-1][0] + capacity_period,
    )
    # Receiver timestamps include sub-millisecond scheduling noise around 60 s.
    if abs(end_time - round(end_time)) < 0.01:
        end_time = float(round(end_time))
    x_max = max(10.0, math.ceil(end_time / 10.0) * 10.0)
    rate_max_value = max(value for series in (capacity, throughput) for _, value in series)
    rate_max = max(10.0, math.ceil(rate_max_value * 1.08 / 5.0) * 5.0)
    rtt_values = [value for series in (rtt, base_rtt) for _, value in series]
    rtt_min = math.floor((min(rtt_values) - 2.0) / 5.0) * 5.0
    rtt_max = math.ceil((max(rtt_values) + 2.0) / 5.0) * 5.0
    if rtt_max - rtt_min < 10.0:
        rtt_max = rtt_min + 10.0

    def x_coord(time_s: float) -> float:
        return left + plot_width * time_s / x_max

    def rate_coord(rate_mbps: float) -> float:
        return bottom + plot_height * rate_mbps / rate_max

    def rtt_coord(rtt_ms: float) -> float:
        return bottom + plot_height * (rtt_ms - rtt_min) / (rtt_max - rtt_min)

    capacity_step = step_points(capacity, x_max)
    capacity_path = [(x_coord(t), rate_coord(v)) for t, v in capacity_step]
    throughput_path = [(x_coord(t), rate_coord(v)) for t, v in throughput]
    rtt_path = [(x_coord(t), rtt_coord(v)) for t, v in rtt]
    base_rtt_path = [(x_coord(t), rtt_coord(v)) for t, v in step_points(base_rtt, x_max)]

    figure = PostScriptFigure(width, height)

    # Capacity is a light filled envelope, matching LeoCC's trace figures.
    figure.add("gsave")
    figure.add(
        f"newpath {left:.3f} {bottom:.3f} moveto {right:.3f} {bottom:.3f} lineto "
        f"{right:.3f} {top:.3f} lineto {left:.3f} {top:.3f} lineto closepath clip"
    )
    capacity_fill = [(x_coord(0.0), bottom), *capacity_path, (x_coord(x_max), bottom)]
    figure.color(0.875, 0.925, 0.965)
    figure.polygon(capacity_fill)
    figure.add("grestore")

    # Grid lines use the left rate scale, as in the referenced paper figures.
    figure.color(0.82, 0.82, 0.82)
    figure.line_width(0.45)
    figure.add("[1.5 2.5] 0 setdash")
    for tick in range(0, int(rate_max) + 1, 5):
        y = rate_coord(float(tick))
        figure.line(left, y, right, y)
    for tick in range(0, int(x_max) + 1, 10):
        x = x_coord(float(tick))
        figure.line(x, bottom, x, top)
    figure.add("[] 0 setdash")

    # Clip all data curves to the plotting rectangle.
    figure.add("gsave")
    figure.add(
        f"newpath {left:.3f} {bottom:.3f} moveto {right:.3f} {bottom:.3f} lineto "
        f"{right:.3f} {top:.3f} lineto {left:.3f} {top:.3f} lineto closepath clip"
    )
    figure.color(0.20, 0.46, 0.70)
    figure.line_width(1.35)
    figure.polyline(capacity_path)
    figure.color(0.78, 0.16, 0.13)
    figure.line_width(1.65)
    figure.polyline(throughput_path)
    figure.color(0.90, 0.00, 0.42)
    figure.line_width(1.10)
    figure.polyline(rtt_path)
    figure.color(0.31, 0.18, 0.52)
    figure.line_width(1.25)
    figure.add("[5 3] 0 setdash")
    figure.polyline(base_rtt_path)
    figure.add("[] 0 setdash")
    figure.add("grestore")

    # Frame and ticks.
    figure.color(0.0, 0.0, 0.0)
    figure.line_width(0.75)
    figure.add(
        f"newpath {left:.3f} {bottom:.3f} moveto {right:.3f} {bottom:.3f} lineto "
        f"{right:.3f} {top:.3f} lineto {left:.3f} {top:.3f} lineto closepath stroke"
    )
    figure.font("Times-Roman", 9.0)
    for tick in range(0, int(x_max) + 1, 10):
        x = x_coord(float(tick))
        figure.line(x, bottom, x, bottom - 3.5)
        figure.text(x, bottom - 14.0, str(tick), align="center")
    for tick in range(0, int(rate_max) + 1, 5):
        y = rate_coord(float(tick))
        figure.line(left, y, left - 3.5, y)
        figure.text(left - 6.0, y - 3.0, str(tick), align="right")
    rtt_tick = rtt_min
    while rtt_tick <= rtt_max + 1e-9:
        y = rtt_coord(rtt_tick)
        figure.line(right, y, right + 3.5, y)
        figure.text(right + 6.0, y - 3.0, f"{rtt_tick:g}")
        rtt_tick += 5.0

    figure.font("Times-Bold", 10.5)
    figure.text((left + right) / 2.0, 17.0, "Time (s)", align="center")
    figure.text(18.0, (bottom + top) / 2.0, "Rate (Mbps)", align="center", rotation=90.0)
    figure.text(523.0, (bottom + top) / 2.0, "RTT (ms)", align="center", rotation=-90.0)

    # Compact, paper-style legend above the axes.
    legend_y = 272.0
    legend_entries = [
        (75.0, (0.20, 0.46, 0.70), "Bottleneck BW", False),
        (183.0, (0.78, 0.16, 0.13), "CREO+ Rx throughput", False),
        (329.0, (0.90, 0.00, 0.42), "TCP RTT", False),
        (409.0, (0.31, 0.18, 0.52), "Base RTT", True),
    ]
    figure.font("Times-Roman", 8.6)
    for x, color, label, dashed in legend_entries:
        figure.color(*color)
        figure.line_width(1.5)
        figure.add("[5 3] 0 setdash" if dashed else "[] 0 setdash")
        figure.line(x, legend_y, x + 16.0, legend_y)
        figure.add("[] 0 setdash")
        figure.color(0.0, 0.0, 0.0)
        figure.text(x + 20.0, legend_y - 3.0, label)

    figure.write(eps_path)


def convert_to_pdf(eps_path: Path, output_path: Path) -> None:
    ps2pdf = shutil.which("ps2pdf")
    if not ps2pdf:
        raise RuntimeError("ps2pdf is required to generate the vector PDF")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ps2pdf, "-dEPSCrop", str(eps_path), str(output_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> None:
    args = parse_args()
    results = args.results.resolve()
    output = args.output.resolve()
    capacity = read_series(results / "realbw.dat")
    throughput = read_series(results / "throughput.dat")
    rtt = read_series(results / "rtt.dat")
    base_rtt = read_series(results / "base-rtt.dat")

    with tempfile.TemporaryDirectory(prefix="creo-linux-plot-") as temp_dir:
        eps_path = Path(temp_dir) / "creo-linux-fig16.eps"
        draw_figure(capacity, throughput, rtt, base_rtt, eps_path)
        convert_to_pdf(eps_path, output)

    print(f"Generated {output}")
    print(
        "Samples: "
        f"capacity={len(capacity)}, throughput={len(throughput)}, "
        f"rtt={len(rtt)}, base_rtt={len(base_rtt)}"
    )


if __name__ == "__main__":
    main()
