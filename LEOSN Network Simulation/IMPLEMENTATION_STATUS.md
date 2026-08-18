# Implementation Status

The paper checked for this release is `LEOCCA.pdf` in the working directory.
No file named exactly `LEOCC.pdf` was present. The reference LeoCC paper was
checked separately against `tcp-leocc.cc`.

## CREO+

| Component | Source | Status |
| --- | --- | --- |
| Burst-pacing capacity sampling | `tcp-rl.cc`, `tcp-sim-creo-*.cc` | Implemented for the connected-phase ns-3 path. |
| DWT denoising | `ablation/ablation_agent.py` | Periodic db2 DWT, MAD noise estimate, soft threshold, and inverse transform. |
| Capacity decomposition | `ablation/ablation_agent.py` | Denoised capacity is separated into moving trend and residual fluctuation. |
| PDPA | `ablation/ablation_agent.py` | Trace-derived candidates, Pareto pruning, and a three-down/one-neutral/three-up action set. |
| DRL | `ablation/ablation_agent.py` | Three LSTMs, metric CNN, actor, twin critics, replay buffer, and discrete SAC updates. |
| ns3-ai exchange | `tcp-rl-env.*`, `tcp-rl.*` | ns-3 observations are sent to Python and returned actions update the TCP control state. |
| Training and testing | `train_creo_*.py`, `test_creo.py`, `ablation/*.py` | Single-flow, multi-flow, component tests, training, and deterministic evaluation are present. |
| UDP handover notification | `scratch/creo-handover-notification/` | Runnable packet header, sequence matching, ACK, RTT-spaced retry, and absolute `tHO` callback. |
| Full stop-wait-resume integration | - | The UDP callback is not yet wired into the complete TCP handover state machine. |

The included checkpoint is a control-path sample with 22 learning updates. It
is not the converged 8,000-10,000-epoch model described by the paper. Paper
performance claims should be released with the corresponding converged weights,
training configuration, seeds, and raw logs.

## Linux prototype

`tcp_creo.c` is a loadable `tcp_congestion_ops` module. It collects delivery
rate, capacity estimate, pacing rate, RTT/minimum RTT, RTT gradient, loss,
inflight, cwnd, BDP, and app-limited state. A root-only `/dev/creo_drl` binary
interface exchanges sequenced per-flow states and actions with the Python
service. Returned actions control `sk_pacing_rate` and `snd_cwnd`; the ACK path
does not wait for Python. The model service implements shared batched inference,
DWT, PDPA, the LSTM/CNN actor, bounded replay, and asynchronous fine-tuning.

This supports the paper's wording "preliminary Linux prototype" and
"state-inference-action path." It does not implement the CREO+ handover phase
inside Linux and should not be described as a complete production deployment.

## LeoCC reproduction

`tcp-leocc.cc` implements startup/drain, aggressive maximum and moderate
bandwidth estimates, RTT-range-guided target selection, the 1.25/0.75 dynamic
cruise cycle, probe RTT, and post-reconfiguration adaptation. The experiment
driver replays both generated and Starlink traces and can inject a handover
outage.

The current reproduction does not implement the paper's real ICMP response-
interval detector, Kalman bandwidth/RTT estimators, MPF RTT filtering, eBPF TCP
option propagation, or full bottleneck-shifting logic. Detection is scheduled
by the experiment driver and the moderate estimator is an EWMA. Therefore the
sentence "we reproduce the complete algorithm flow of LeoCC" is stronger than
the available code. A technically accurate replacement is: "we reproduce
LeoCC's core rate-control, dynamic-cruise, and reconfiguration-adaptation
behavior in ns-3."
