# Checkpoints

`example-smoke.pt` verifies serialization and the ns-3/Linux action path. Its
metadata records `variant=full`, a 32-sample capacity window, and 22 learner
updates. It is not a converged performance checkpoint.

Training scripts can write a replacement checkpoint to any path. Pass that path
explicitly to `test_creo.py`, `test_ablation.py`, or the Linux model daemon.
