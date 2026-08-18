# CREO+ TCP Examples

The C++ side derives TCP observations, samples burst-pacing delivery, and
applies Python actions. The Python side provides single-flow training,
multi-flow training, checkpoint evaluation, and controlled ablations.

```bash
python3 use-gym/train_creo_single.py --help
python3 use-gym/train_creo_multi.py --help
python3 use-gym/test_creo.py --help
python3 use-gym/ablation/run_ablation_suite.py --help
```

Run commands from the ns-3 root so trace and executable paths resolve against
the installed tree.
