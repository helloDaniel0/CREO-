#!/usr/bin/env python3
"""Verify CREO+ DRL on a real outbound HTTPS upload without changing default CC."""

from __future__ import annotations

import argparse
import collections
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from evaluate_creo_drl import (
    PARAMETERS,
    ROOT,
    command,
    default_checkpoint,
    read_parameter,
    write_parameter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=20_000_000)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/drl-outbound-upload"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.geteuid() == 0:
        raise SystemExit("run as the workspace user; this script invokes sudo -n")
    command(["sudo", "-n", "true"])
    if not PARAMETERS.is_dir() or not Path("/dev/creo_drl").exists():
        raise SystemExit("the online-DRL tcp_creo.ko module is not loaded")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")

    args.output.mkdir(parents=True, exist_ok=True)
    model_output = args.output / "model"
    upload_output = args.output / "upload.json"
    stop_file = args.output / ".stop-model-daemon"
    stop_file.unlink(missing_ok=True)
    drl_before = read_parameter("drl_enabled")
    default_before = command(
        ["sysctl", "-n", "net.ipv4.tcp_congestion_control"],
        capture_output=True,
    ).stdout.strip()

    daemon_log = (args.output / "model-daemon.log").open("w", encoding="utf-8")
    daemon = subprocess.Popen(
        [
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
            "90",
        ],
        cwd=ROOT,
        stdout=daemon_log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    upload_returncode = -1
    error_text: str | None = None
    try:
        time.sleep(1.0)
        if daemon.poll() is not None:
            raise RuntimeError("model daemon exited before upload")
        write_parameter("drl_enabled", "Y")
        upload = subprocess.run(
            [
                sys.executable,
                str(ROOT / "evaluation/cloudflare_upload.py"),
                "--bytes",
                str(args.bytes),
                "--cc",
                "creo",
                "--output",
                str(upload_output),
            ],
            cwd=ROOT,
            text=True,
        )
        upload_returncode = upload.returncode
        if upload.returncode:
            error_text = f"upload exited {upload.returncode}"
    except Exception as error:  # restoration must still run
        error_text = str(error)
    finally:
        stop_file.write_text("stop\n", encoding="ascii")
        try:
            daemon.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(daemon.pid, signal.SIGTERM)
            daemon.wait(timeout=5)
        daemon_log.close()
        write_parameter("drl_enabled", drl_before)
        stop_file.unlink(missing_ok=True)

    model_summary_path = model_output / "online-summary.json"
    model_summary = (
        json.loads(model_summary_path.read_text(encoding="utf-8"))
        if model_summary_path.exists()
        else {}
    )
    upload_summary = (
        json.loads(upload_output.read_text(encoding="utf-8"))
        if upload_output.exists()
        else {}
    )
    rows = []
    control_log = model_output / "online-control.jsonl"
    if control_log.exists():
        rows = [json.loads(line) for line in control_log.read_text().splitlines()]
    action_counts = collections.Counter(int(row["action_q10"]) for row in rows)
    default_after = command(
        ["sysctl", "-n", "net.ipv4.tcp_congestion_control"],
        capture_output=True,
    ).stdout.strip()
    result = {
        "status": "ok"
        if error_text is None
        and upload_returncode == 0
        and daemon.returncode == 0
        and model_summary.get("closed_loop_verified")
        and upload_summary.get("selected_cc") == "creo"
        and default_after == default_before
        else "failed",
        "error": error_text,
        "selected_socket_cc": upload_summary.get("selected_cc"),
        "bytes_uploaded": upload_summary.get("bytes_uploaded"),
        "upload_mbps": upload_summary.get("upload_mbps"),
        "remote_ip": upload_summary.get("remote_ip"),
        "http_status": upload_summary.get("http_status"),
        "closed_loop_verified": model_summary.get("closed_loop_verified", False),
        "states_received": model_summary.get("states_received", 0),
        "kernel_action_matches": model_summary.get("kernel_action_matches", 0),
        "kernel_action_mismatches": model_summary.get(
            "kernel_action_mismatches", 0
        ),
        "mean_inference_us": model_summary.get("mean_inference_us"),
        "action_q10_counts": {str(key): value for key, value in action_counts.items()},
        "default_cc_before": default_before,
        "default_cc_after": default_after,
        "drl_enabled_before": drl_before,
        "drl_enabled_after": read_parameter("drl_enabled"),
    }
    (args.output / "outbound-summary.json").write_text(
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
