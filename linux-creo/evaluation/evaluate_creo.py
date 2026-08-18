#!/usr/bin/env python3
"""Run a trace-driven CREO TCP experiment on one Linux host.

The client, router, and server live in temporary network namespaces.  Only the
router namespace is shaped, so the host's physical interfaces and routes are
never modified.  The script restores the CREO module parameters and removes
all temporary namespaces in a finally block.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


CLIENT_NS = "creo-cli"
ROUTER_NS = "creo-rtr"
SERVER_NS = "creo-srv"
NAMESPACES = (CLIENT_NS, ROUTER_NS, SERVER_NS)
MODULE_PARAMETER_DIR = Path("/sys/module/tcp_creo/parameters")
RTT_PATTERN = re.compile(r"\brtt:([0-9.]+)/([0-9.]+)")
BYTES_ACKED_PATTERN = re.compile(r"\bbytes_acked:(\d+)")
PING_PATTERN = re.compile(
    r"(?:rtt|round-trip) min/avg/max/(?:mdev|stddev) = "
    r"([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)"
)


def command(
    args: list[str],
    *,
    check: bool = True,
    capture: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        timeout=timeout,
    )


def sudo(*args: str, **kwargs) -> subprocess.CompletedProcess[str]:
    return command(["sudo", "-n", *args], **kwargs)


def ns(namespace: str, *args: str, **kwargs) -> subprocess.CompletedProcess[str]:
    return sudo("ip", "netns", "exec", namespace, *args, **kwargs)


def read_text(args: list[str]) -> str:
    return command(args).stdout.strip()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def module_parameters() -> dict[str, str]:
    if not MODULE_PARAMETER_DIR.is_dir():
        raise RuntimeError("tcp_creo is not loaded")
    return {path.name: path.read_text(encoding="ascii").strip() for path in MODULE_PARAMETER_DIR.iterdir()}


def write_module_parameter(name: str, value: str | int) -> None:
    path = MODULE_PARAMETER_DIR / name
    sudo("sh", "-c", f"printf '%s' {value!s} > {path}")


def snapshot_host() -> dict[str, object]:
    return {
        "default_cc": read_text(["sysctl", "-n", "net.ipv4.tcp_congestion_control"]),
        "available_cc": read_text(["sysctl", "-n", "net.ipv4.tcp_available_congestion_control"]),
        "namespaces": read_text(["ip", "netns", "list"]),
        "qdisc": read_text(["tc", "qdisc", "show"]),
        "module_parameters": module_parameters(),
    }


def assert_namespaces_available() -> None:
    existing = {line.split()[0] for line in read_text(["ip", "netns", "list"]).splitlines() if line}
    overlap = existing.intersection(NAMESPACES)
    if overlap:
        raise RuntimeError(f"refusing to replace existing namespaces: {sorted(overlap)}")


def setup_topology(rate_mbps: float, one_way_delay_ms: float, queue_packets: int) -> None:
    for namespace in NAMESPACES:
        sudo("ip", "netns", "add", namespace)

    sudo("ip", "link", "add", "ceth0", "type", "veth", "peer", "name", "rcli0")
    sudo("ip", "link", "set", "ceth0", "netns", CLIENT_NS)
    sudo("ip", "link", "set", "rcli0", "netns", ROUTER_NS)
    sudo("ip", "link", "add", "rsrv0", "type", "veth", "peer", "name", "seth0")
    sudo("ip", "link", "set", "rsrv0", "netns", ROUTER_NS)
    sudo("ip", "link", "set", "seth0", "netns", SERVER_NS)

    for namespace in NAMESPACES:
        ns(namespace, "ip", "link", "set", "lo", "up")

    ns(CLIENT_NS, "ip", "addr", "add", "10.210.1.2/24", "dev", "ceth0")
    ns(CLIENT_NS, "ip", "link", "set", "ceth0", "up")
    ns(CLIENT_NS, "ip", "route", "add", "default", "via", "10.210.1.1")

    ns(ROUTER_NS, "ip", "addr", "add", "10.210.1.1/24", "dev", "rcli0")
    ns(ROUTER_NS, "ip", "addr", "add", "10.210.2.1/24", "dev", "rsrv0")
    ns(ROUTER_NS, "ip", "link", "set", "rcli0", "up")
    ns(ROUTER_NS, "ip", "link", "set", "rsrv0", "up")
    ns(ROUTER_NS, "sysctl", "-q", "-w", "net.ipv4.ip_forward=1")

    ns(SERVER_NS, "ip", "addr", "add", "10.210.2.2/24", "dev", "seth0")
    ns(SERVER_NS, "ip", "link", "set", "seth0", "up")
    ns(SERVER_NS, "ip", "route", "add", "default", "via", "10.210.2.1")

    for namespace, interface in (
        (CLIENT_NS, "ceth0"),
        (ROUTER_NS, "rcli0"),
        (ROUTER_NS, "rsrv0"),
        (SERVER_NS, "seth0"),
    ):
        ns(namespace, "ethtool", "-K", interface, "tso", "off", "gso", "off", "gro", "off")

    # sch_fq consumes sk_pacing_rate from the real CREO TCP socket.
    ns(CLIENT_NS, "tc", "qdisc", "replace", "dev", "ceth0", "root", "fq")

    # Forward path: HTB is the time-varying bottleneck and NetEm contributes
    # one-way propagation delay plus a bounded FIFO queue.
    ns(ROUTER_NS, "tc", "qdisc", "replace", "dev", "rsrv0", "root", "handle", "1:", "htb", "default", "10")
    ns(
        ROUTER_NS,
        "tc",
        "class",
        "add",
        "dev",
        "rsrv0",
        "parent",
        "1:",
        "classid",
        "1:10",
        "htb",
        "rate",
        f"{rate_mbps:.3f}mbit",
        "ceil",
        f"{rate_mbps:.3f}mbit",
        "burst",
        "32k",
        "cburst",
        "32k",
    )
    ns(
        ROUTER_NS,
        "tc",
        "qdisc",
        "add",
        "dev",
        "rsrv0",
        "parent",
        "1:10",
        "handle",
        "10:",
        "netem",
        "delay",
        f"{one_way_delay_ms:.3f}ms",
        "limit",
        str(queue_packets),
    )

    # ACK path uses the same propagation delay but no capacity bottleneck.
    ns(
        ROUTER_NS,
        "tc",
        "qdisc",
        "replace",
        "dev",
        "rcli0",
        "root",
        "netem",
        "delay",
        f"{one_way_delay_ms:.3f}ms",
        "limit",
        "1000",
    )


def cleanup_topology() -> None:
    for namespace in NAMESPACES:
        subprocess.run(
            ["sudo", "-n", "ip", "netns", "del", namespace],
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def variable_rate(time_s: float) -> float:
    """Deterministic 16-36 Mbps connected-phase capacity trace."""
    rate = 26.0 + 6.0 * math.sin(2.0 * math.pi * time_s / 11.0)
    rate += 3.5 * math.sin(2.0 * math.pi * time_s / 4.5 + 0.7)
    return min(max(rate, 16.0), 36.0)


def make_trace(duration_s: float, period_s: float) -> list[tuple[float, float]]:
    count = math.ceil(duration_s / period_s)
    return [(index * period_s, variable_rate(index * period_s)) for index in range(count)]


def play_trace(
    trace: list[tuple[float, float]],
    start: threading.Event,
    stop: threading.Event,
    rows: list[tuple[float, float]],
) -> None:
    start.wait()
    begin = time.monotonic()
    for scheduled_s, rate_mbps in trace:
        remaining = begin + scheduled_s - time.monotonic()
        if remaining > 0 and stop.wait(remaining):
            return
        if stop.is_set():
            return
        ns(
            ROUTER_NS,
            "tc",
            "class",
            "change",
            "dev",
            "rsrv0",
            "parent",
            "1:",
            "classid",
            "1:10",
            "htb",
            "rate",
            f"{rate_mbps:.3f}mbit",
            "ceil",
            f"{rate_mbps:.3f}mbit",
            "burst",
            "32k",
            "cburst",
            "32k",
        )
        rows.append((scheduled_s, rate_mbps))


def sample_rtt(
    start: threading.Event,
    stop: threading.Event,
    period_s: float,
    rows: list[tuple[float, float, float]],
) -> None:
    start.wait()
    begin = time.monotonic()
    while not stop.wait(period_s):
        output = ns(CLIENT_NS, "ss", "-tin", "dst", "10.210.2.2", check=False).stdout
        candidates: list[tuple[int, re.Match[str]]] = []
        for line in output.splitlines():
            match = RTT_PATTERN.search(line)
            if not match:
                continue
            bytes_match = BYTES_ACKED_PATTERN.search(line)
            candidates.append((int(bytes_match.group(1)) if bytes_match else 0, match))
        if candidates:
            # iperf3 creates a low-volume control socket and a data socket.
            # Select the connection that has acknowledged the most bytes.
            _, match = max(candidates, key=lambda candidate: candidate[0])
            rows.append((time.monotonic() - begin, float(match.group(1)), float(match.group(2))))


def measure_base_rtt() -> tuple[float, str]:
    completed = ns(
        CLIENT_NS,
        "ping",
        "-n",
        "-q",
        "-c",
        "10",
        "-i",
        "0.05",
        "10.210.2.2",
        timeout=5,
    )
    match = PING_PATTERN.search(completed.stdout)
    if not match:
        raise RuntimeError(f"could not parse base RTT from:\n{completed.stdout}")
    # The first ICMP can include one-time qdisc/neighbor warmup.  Congestion
    # control defines base RTT as the minimum empty-path sample, not the mean.
    return float(match.group(1)), completed.stdout


def parse_iperf_intervals(
    payload: dict[str, object], expected_period_s: float
) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    for interval in payload.get("intervals", []):
        summary = interval.get("sum") or interval.get("sum_received")
        if not summary:
            streams = interval.get("streams", [])
            summary = streams[0] if streams else None
        if not summary:
            continue
        start_s = float(summary.get("start", 0.0))
        end_s = float(summary.get("end", start_s))
        if end_s - start_s < 0.9 * expected_period_s:
            continue
        rows.append(((start_s + end_s) / 2.0, float(summary["bits_per_second"]) / 1e6))
    return rows


def aggregate_rtt(
    raw_rows: list[tuple[float, float, float]], duration_s: float, period_s: float
) -> list[tuple[float, float]]:
    bins: dict[int, list[float]] = {}
    for timestamp, rtt_ms, _ in raw_rows:
        index = int(timestamp / period_s)
        bins.setdefault(index, []).append(rtt_ms)
    result: list[tuple[float, float]] = []
    for index in range(math.ceil(duration_s / period_s)):
        values = bins.get(index)
        if values:
            result.append(((index + 0.5) * period_s, statistics.fmean(values)))
    return result


def write_dat(path: Path, rows: list[tuple[float, ...]], header: str) -> None:
    with path.open("w", encoding="ascii") as handle:
        handle.write(f"# {header}\n")
        for row in rows:
            handle.write(" ".join(f"{value:.6f}" for value in row) + "\n")


def summarize(
    throughput: list[tuple[float, float]],
    realbw: list[tuple[float, float]],
    rtt_raw: list[tuple[float, float, float]],
    base_rtt_ms: float,
    warmup_s: float,
) -> dict[str, float | bool]:
    throughput_values = [value for timestamp, value in throughput if timestamp >= warmup_s]
    capacity_values = [value for timestamp, value in realbw if timestamp >= warmup_s]
    rtt_values = [rtt for timestamp, rtt, _ in rtt_raw if timestamp >= warmup_s]
    mean_throughput = statistics.fmean(throughput_values)
    mean_capacity = statistics.fmean(capacity_values)
    mean_rtt = statistics.fmean(rtt_values)
    utilization = mean_throughput / mean_capacity
    inflation = mean_rtt / base_rtt_ms - 1.0
    return {
        "mean_realbw_mbps": mean_capacity,
        "mean_throughput_mbps": mean_throughput,
        "link_utilization": utilization,
        "base_rtt_ms": base_rtt_ms,
        "mean_rtt_ms": mean_rtt,
        "p95_rtt_ms": percentile(rtt_values, 0.95),
        "rtt_inflation": inflation,
        "target_utilization_met": 0.80 <= utilization <= 0.90,
        "target_rtt_inflation_met": 0.10 <= inflation <= 0.20,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--trace-period", type=float, default=0.5)
    parser.add_argument("--throughput-period", type=float, default=0.5)
    parser.add_argument("--rtt-period", type=float, default=0.1)
    parser.add_argument("--base-rtt", type=float, default=20.0)
    parser.add_argument("--queue-packets", type=int, default=100)
    parser.add_argument("--action-index", type=int, default=3)
    parser.add_argument(
        "--action-gain-q10",
        type=int,
        default=0,
        help="Q10 model action override; zero uses action-index",
    )
    parser.add_argument("--update-interval-us", type=int, default=100000)
    parser.add_argument("--probe-cycle-steps", type=int, default=0)
    parser.add_argument("--probe-gain-q10", type=int, default=1280)
    parser.add_argument("--cruise-gain-q10", type=int, default=799)
    parser.add_argument("--output", type=Path, default=Path("results/variable-capacity"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration <= args.warmup or args.base_rtt <= 0 or args.trace_period <= 0:
        raise SystemExit("duration/base-rtt/period arguments are invalid")
    if not 0 <= args.action_index <= 6:
        raise SystemExit("action-index must be in [0, 6]")
    if args.action_gain_q10 and not 512 <= args.action_gain_q10 <= 2048:
        raise SystemExit("action-gain-q10 must be zero or in [512, 2048]")
    if os.geteuid() == 0:
        raise SystemExit("run as the normal workspace user; the script invokes sudo -n itself")
    if shutil.which("iperf3") is None or shutil.which("ss") is None:
        raise SystemExit("iperf3 and ss are required")

    sudo("true")
    assert_namespaces_available()
    before = snapshot_host()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "host-state-before.json").write_text(
        json.dumps(before, indent=2, sort_keys=True), encoding="utf-8"
    )

    trace = make_trace(args.duration, args.trace_period)
    realbw_rows: list[tuple[float, float]] = []
    rtt_rows: list[tuple[float, float, float]] = []
    start = threading.Event()
    stop = threading.Event()
    old_signal_handlers: dict[int, object] = {}
    server_process: subprocess.Popen[str] | None = None
    restored = False

    def interrupt(_signum, _frame) -> None:
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_signal_handlers[signum] = signal.signal(signum, interrupt)

    try:
        write_module_parameter("action_index", args.action_index)
        if (MODULE_PARAMETER_DIR / "action_gain_q10").exists():
            write_module_parameter("action_gain_q10", args.action_gain_q10)
        write_module_parameter("update_interval_us", args.update_interval_us)
        for name, value in (
            ("probe_cycle_steps", args.probe_cycle_steps),
            ("probe_gain_q10", args.probe_gain_q10),
            ("cruise_gain_q10", args.cruise_gain_q10),
        ):
            if (MODULE_PARAMETER_DIR / name).exists():
                write_module_parameter(name, value)
        write_module_parameter("debug", "N")
        setup_topology(trace[0][1], args.base_rtt / 2.0, args.queue_packets)
        base_rtt_ms, ping_output = measure_base_rtt()
        (args.output / "base-rtt-ping.txt").write_text(ping_output, encoding="utf-8")

        trace_thread = threading.Thread(
            target=play_trace, args=(trace, start, stop, realbw_rows), daemon=True
        )
        rtt_thread = threading.Thread(
            target=sample_rtt,
            args=(start, stop, args.rtt_period, rtt_rows),
            daemon=True,
        )
        trace_thread.start()
        rtt_thread.start()

        server_process = subprocess.Popen(
            [
                "sudo",
                "-n",
                "ip",
                "netns",
                "exec",
                SERVER_NS,
                "iperf3",
                "-s",
                "-1",
                "-i",
                f"{args.throughput_period:.3f}",
                "-J",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        time.sleep(0.25)
        start.set()
        client = ns(
            CLIENT_NS,
            "iperf3",
            "-c",
            "10.210.2.2",
            "-C",
            "creo",
            "-t",
            f"{args.duration:.3f}",
            "-i",
            f"{args.throughput_period:.3f}",
            "-w",
            "4M",
            "-J",
            timeout=args.duration + 20,
            check=False,
        )
        stop.set()
        trace_thread.join(timeout=2)
        rtt_thread.join(timeout=2)
        if client.returncode != 0:
            raise RuntimeError(f"iperf3 client failed:\n{client.stdout}")
        (args.output / "iperf-client.json").write_text(client.stdout, encoding="utf-8")

        server_stdout, _ = server_process.communicate(timeout=10)
        if server_process.returncode != 0:
            raise RuntimeError(f"iperf3 server failed:\n{server_stdout}")
        (args.output / "iperf-server.json").write_text(server_stdout, encoding="utf-8")
        server_payload = json.loads(server_stdout)
        throughput_rows = parse_iperf_intervals(server_payload, args.throughput_period)
        if not throughput_rows or not rtt_rows or not realbw_rows:
            raise RuntimeError("one or more metric streams are empty")

        rtt_aggregated = aggregate_rtt(rtt_rows, args.duration, args.throughput_period)
        write_dat(args.output / "realbw.dat", realbw_rows, "time_s real_bandwidth_mbps")
        write_dat(args.output / "throughput.dat", throughput_rows, "time_s receiver_throughput_mbps")
        write_dat(args.output / "rtt.dat", rtt_aggregated, "time_s mean_tcp_rtt_ms")
        write_dat(args.output / "rtt-raw.dat", rtt_rows, "time_s tcp_rtt_ms rttvar_ms")
        write_dat(
            args.output / "base-rtt.dat",
            [(timestamp, base_rtt_ms) for timestamp, _ in realbw_rows],
            "time_s measured_idle_base_rtt_ms",
        )

        summary = summarize(throughput_rows, realbw_rows, rtt_rows, base_rtt_ms, args.warmup)
        summary.update(
            {
                "algorithm": "creo",
                "action_index": args.action_index,
                "action_gain_q10": args.action_gain_q10,
                "action_gain": (
                    args.action_gain_q10 / 1024.0 if args.action_gain_q10 else None
                ),
                "update_interval_us": args.update_interval_us,
                "probe_cycle_steps": args.probe_cycle_steps,
                "probe_gain_q10": args.probe_gain_q10,
                "cruise_gain_q10": args.cruise_gain_q10,
                "duration_s": args.duration,
                "warmup_s": args.warmup,
                "trace_period_s": args.trace_period,
                "throughput_period_s": args.throughput_period,
                "queue_packets": args.queue_packets,
                "implementation_scope": (
                    "connected-phase kernel online DRL"
                    if before["module_parameters"].get("drl_enabled") == "Y"
                    else "connected-phase kernel fallback policy"
                ),
            }
        )
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        stop.set()
        if server_process is not None and server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                server_process.kill()
        cleanup_topology()
        for name, value in before["module_parameters"].items():
            write_module_parameter(name, value)
        current_cc = read_text(["sysctl", "-n", "net.ipv4.tcp_congestion_control"])
        if current_cc != before["default_cc"]:
            sudo("sysctl", "-q", "-w", f"net.ipv4.tcp_congestion_control={before['default_cc']}")
        after = snapshot_host()
        restored = (
            after["default_cc"] == before["default_cc"]
            and after["namespaces"] == before["namespaces"]
            and after["qdisc"] == before["qdisc"]
            and after["module_parameters"] == before["module_parameters"]
        )
        restore_report = {
            "restored": restored,
            "default_cc_unchanged": after["default_cc"] == before["default_cc"],
            "namespaces_restored": after["namespaces"] == before["namespaces"],
            "host_qdisc_restored": after["qdisc"] == before["qdisc"],
            "module_parameters_restored": after["module_parameters"] == before["module_parameters"],
        }
        (args.output / "host-state-after.json").write_text(
            json.dumps(after, indent=2, sort_keys=True), encoding="utf-8"
        )
        (args.output / "restore-report.json").write_text(
            json.dumps(restore_report, indent=2, sort_keys=True), encoding="utf-8"
        )
        for signum, handler in old_signal_handlers.items():
            signal.signal(signum, handler)

    if not restored:
        raise RuntimeError(f"host state restoration check failed; inspect {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
