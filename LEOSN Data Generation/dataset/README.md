# Trace Files

- `bw.txt`, `latency.txt`: topology-generated capacity and propagation-delay
  traces.
- `SIGCOMMbw.txt`, `SIGCOMMlatency.txt`: local Starlink traces consumed by the
  LeoCC and CREO+ experiment drivers.

Trace rows are replayed by sample index; the ns-3 drivers control the replay
interval and optional capacity jitter. Verify the license and provenance of
third-party traces before redistribution.
