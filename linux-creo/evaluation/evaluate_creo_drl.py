#!/usr/bin/env python3
"""Orchestrate a reproducible kernel-to-model-to-kernel CREO+ evaluation."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = Path("/sys/module/tcp_creo/parameters")


def command(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, text=True, **kwargs)


def read_parameter(name: str) -> str:
    return (PARAMETERS / name).read_text(encoding="ascii").strip()


def write_parameter(name: str, value: str) -> None:
    command(
        [
            "sudo",
            "-n",
            "sh",
            "-c",
            f"printf '%s' {value!s} > {PARAMETERS / name}",
        ],
        stdout=subprocess.DEVNULL,
    )


def default_checkpoint() -> Path:
    return (
        ROOT.parent
        / "ns3-creo/ns3-overlay/contrib/ai/examples/rl-tcp/use-gym/checkpoints/example-smoke.pt"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=3.0)
    parser.add_argument("--base-rtt", type=float, default=20.0)
    parser.add_argument("--update-interval-us", type=int, default=100000)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument("--capacity-trace", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/drl-closed-loop"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.geteuid() == 0:
        raise SystemExit("run as the workspace user; this script invokes sudo -n")
    command(["sudo", "-n", "true"])
    if not PARAMETERS.is_dir() or not (PARAMETERS / "drl_enabled").exists():
        raise SystemExit("the online-DRL tcp_creo.ko module is not loaded")
    if not Path("/dev/creo_drl").exists():
        raise SystemExit("/dev/creo_drl is missing")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")

    args.output.mkdir(parents=True, exist_ok=True)
    model_output = args.output / "model"
    network_output = args.output / "network"
    stop_file = args.output / ".stop-model-daemon"
    if stop_file.exists():
        stop_file.unlink()

    default_before = command(
        ["sysctl", "-n", "net.ipv4.tcp_congestion_control"],
        capture_output=True,
    ).stdout.strip()
    drl_before = read_parameter("drl_enabled")
    daemon_command = [
        "sudo",
        "-n",
        sys.executable,
        str(ROOT / "deployment/creo_drl_daemon.py"),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--state-dir",
        str(model_output.resolve()),
        "--stop-file",
        str(stop_file.resolve()),
        "--duration",
        str(args.duration + 30.0),
    ]
    if args.capacity_trace:
        daemon_command.extend(
            ["--capacity-trace", str(args.capacity_trace.resolve())]
        )

    daemon_log = (args.output / "model-daemon.log").open("w", encoding="utf-8")
    daemon = subprocess.Popen(
        daemon_command,
        cwd=ROOT,
        stdout=daemon_log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    evaluation_error: str | None = None
    network_returncode = -1

    try:
        time.sleep(1.0)
        if daemon.poll() is not None:
            raise RuntimeError("model daemon exited before the network test")
        write_parameter("drl_enabled", "Y")
        network_command = [
            sys.executable,
            str(ROOT / "evaluation/evaluate_creo.py"),
            "--duration",
            str(args.duration),
            "--warmup",
            str(args.warmup),
            "--base-rtt",
            str(args.base_rtt),
            "--update-interval-us",
            str(args.update_interval_us),
            "--probe-cycle-steps",
            "0",
            "--output",
            str(network_output),
        ]
        completed = subprocess.run(network_command, cwd=ROOT, text=True)
        network_returncode = completed.returncode
        if completed.returncode:
            evaluation_error = f"network evaluator exited {completed.returncode}"
    except Exception as error:  # restoration must still run
        evaluation_error = str(error)
    finally:
        stop_file.write_text("stop\n", encoding="ascii")
        try:
            daemon.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(daemon.pid, signal.SIGTERM)
            daemon.wait(timeout=5)
        daemon_log.close()
        write_parameter("drl_enabled", drl_before)
        default_after = command(
            ["sysctl", "-n", "net.ipv4.tcp_congestion_control"],
            capture_output=True,
        ).stdout.strip()
        if default_after != default_before:
            command(
                [
                    "sudo",
                    "-n",
                    "sysctl",
                    "-q",
                    "-w",
                    f"net.ipv4.tcp_congestion_control={default_before}",
                ]
            )
        stop_file.unlink(missing_ok=True)

    model_summary_path = model_output / "online-summary.json"
    network_summary_path = network_output / "summary.json"
    model_summary = (
        json.loads(model_summary_path.read_text(encoding="utf-8"))
        if model_summary_path.exists()
        else {}
    )
    network_summary = (
        json.loads(network_summary_path.read_text(encoding="utf-8"))
        if network_summary_path.exists()
        else {}
    )
    restored_default = command(
        ["sysctl", "-n", "net.ipv4.tcp_congestion_control"],
        capture_output=True,
    ).stdout.strip()
    result = {
        "status": "ok"
        if not evaluation_error
        and network_returncode == 0
        and daemon.returncode == 0
        and model_summary.get("closed_loop_verified")
        else "failed",
        "error": evaluation_error,
        "network_returncode": network_returncode,
        "daemon_returncode": daemon.returncode,
        "closed_loop_verified": model_summary.get("closed_loop_verified", False),
        "kernel_action_matches": model_summary.get("kernel_action_matches", 0),
        "kernel_action_mismatches": model_summary.get(
            "kernel_action_mismatches", 0
        ),
        "mean_inference_us": model_summary.get("mean_inference_us"),
        "link_utilization": network_summary.get("link_utilization"),
        "mean_throughput_mbps": network_summary.get("mean_throughput_mbps"),
        "mean_rtt_ms": network_summary.get("mean_rtt_ms"),
        "default_cc_before": default_before,
        "default_cc_after": restored_default,
        "drl_enabled_before": drl_before,
        "drl_enabled_after": read_parameter("drl_enabled"),
    }
    (args.output / "closed-loop-summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    command(
        [
            "sudo",
            "-n",
            "chown",
            "-R",
            f"{os.getuid()}:{os.getgid()}",
            str(args.output.resolve()),
        ]
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
