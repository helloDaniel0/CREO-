# Linux Evaluation

- `evaluate_creo_drl.py`: local iperf3 flow through namespaces, HTB, and NetEm;
  verifies kernel-acknowledged model actions.
- `evaluate_creo_drl_upload.py`: real outbound TLS upload on one socket selected
  with `TCP_CONGESTION=creo`.
- `evaluate_creo.py`: fallback-controller trace replay.
- `cloudflare_upload.py`: socket-level upload helper.
- `plot_creo_linux_*.py`: throughput/capacity and RTT/base-RTT figures.

The namespace tests require root networking privileges, `iperf3`, `tc`, and a
built module. They record restoration checks and leave the host default CCA
unchanged.
