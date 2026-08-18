# LeoCC Experiments

`leocc-connected-test.cc` is the common ns-3 topology and measurement driver.
`run_leocc_connected.py` runs generated and Starlink BP/ISL cases.
`run_leocc_handover.py` injects the configured interruption and schedules the
post-outage controller notification. `run_leocc_weakness.py` exercises rapid
capacity variation and long-RTT conditions.

Outputs include receiver goodput, sender throughput, capacity, RTT, minimum RTT,
queue state, and aggregate summaries. The handover driver models ICMP detection
time; it does not exchange actual ICMP packets.
