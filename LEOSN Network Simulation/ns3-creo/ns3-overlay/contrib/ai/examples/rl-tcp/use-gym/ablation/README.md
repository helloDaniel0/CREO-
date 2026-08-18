# Connected-Phase Components

`ablation_agent.py` contains the full feature and learner path: db2 DWT
denoising, trend/residual decomposition, PDPA action construction, three LSTM
branches, a metric CNN, twin critics, and discrete SAC training.

The variants change one component while preserving topology, trace, seed,
reward, and training budget: `full`, `no_dwt`, `no_burst_pacing`, `no_pdpa`,
`no_lstm`, and `no_cnn`.

```bash
python3 test_components.py
python3 run_ablation_suite.py --smoke
python3 run_ablation_suite.py --episodes 20 --duration 200 \
  --sim_seeds 7,19,31,43,59 \
  --bw_trace dataset/SIGCOMMbw.txt \
  --latency_trace dataset/SIGCOMMlatency.txt
```

Models, logs, and summaries are generated locally and excluded from source
control.
