# Training and Evaluation

- `creo_agent.py`: compact discrete SAC agent used by the basic drivers.
- `train_creo_single.py`: single-flow training.
- `train_creo_multi.py`: multi-flow training with per-flow agents.
- `test_creo.py`: deterministic checkpoint evaluation.
- `tcp-rl-env.*`: Gym observation/action bridge.
- `tcp-rl.*`: TCP hooks, burst-pacing state, and action application.
- `tcp-sim-creo-single.cc`: single-flow trace-driven topology.
- `tcp-sim-creo-multi.cc`: multi-flow trace-driven topology.
- `ablation/`: complete connected-phase DWT/PDPA/LSTM/CNN/SAC implementation.

The basic agent is useful for interface tests. Use `ablation_agent.py` when the
full connected-phase feature pipeline is required.
