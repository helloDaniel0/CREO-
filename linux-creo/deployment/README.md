# User-Space Model Service

`creo_drl_daemon.py` translates the packed ABI in `creo_drl_uapi.h` into batched
model requests and returns flow/sequence-matched Q10 actions.
`creo_model_service.py` implements db2 DWT, PDPA, three LSTMs, the metric CNN,
shared inference, bounded SQLite replay, and a coalescing fine-tune worker.

```bash
python3 creo_model_service.py --self-test --checkpoint /path/model.pt \
  --state-dir /tmp/creo-model-test
sudo python3 creo_drl_daemon.py --checkpoint /path/model.pt \
  --state-dir /var/lib/creo-drl
```

The supplied systemd unit assumes the repository is installed at
`/opt/creo-plus`; adjust `WorkingDirectory` and `ExecStart` for another path.
