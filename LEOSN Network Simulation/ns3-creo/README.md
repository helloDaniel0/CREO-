# CREO+ on ns-3.41

`ns3-overlay` preserves the paths expected by ns-3.41. Install it into a clean
ns-3.41 source tree:

```bash
rsync -a ns3-overlay/ /path/to/ns-3.41/
cd /path/to/ns-3.41
python3 -m pip install -r contrib/ai/requirements.txt
python3 -m pip install -r contrib/ai/examples/rl-tcp/requirements.txt
./ns3 configure --enable-examples
./ns3 build ai ns3ai_creo_single ns3ai_creo_multi ns3ai_creo_ablation
```

Main targets:

- `ns3ai_creo_single`: one TCP flow and one Python agent.
- `ns3ai_creo_multi`: multiple independently controlled TCP flows.
- `creo-udp-handover-notification`: UDP notification/ACK/retry example.

The generated and Starlink traces are installed under `dataset/`. The Python
programs accept explicit trace and output paths for reproducible runs.
