# CREO+ Linux TCP Module

`tcp_creo.c` builds as an out-of-tree loadable CCA named `creo`. It registers
through `tcp_congestion_ops`, exports `/dev/creo_drl` mode `0600`, queues
sequenced per-flow telemetry without blocking ACK processing, and applies the
latest valid action to kernel pacing and cwnd.

```bash
make
sudo insmod tcp_creo.ko drl_enabled=1 update_interval_us=100000
sysctl net.ipv4.tcp_available_congestion_control
sudo python3 deployment/creo_drl_daemon.py --checkpoint /path/model.pt
iperf3 -C creo -c SERVER
sudo rmmod tcp_creo
```

Use an `fq` egress qdisc for pacing. Select `creo` per socket with
`TCP_CONGESTION`; changing the host-wide default is unnecessary. The current
module implements the connected phase only.
