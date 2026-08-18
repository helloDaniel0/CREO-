# CREO+ Research Prototype

This repository contains the implementation sources used to study CREO+ in
ns-3 and Linux, together with an ns-3 reproduction of the LeoCC control path.

## Source tree

- `ns3-creo/`: ns3-ai runtime, CREO+ training and evaluation programs, DWT,
  PDPA, discrete SAC, burst-pacing hooks, and UDP handover notification.
- `linux-creo/`: out-of-tree Linux TCP congestion-control module, user-space
  shared-model service, binary device ABI, and single-host evaluators.
- `leocc-ns3/`: LeoCC controller, ns-3 integration patch, and trace-driven
  connected/handover experiment drivers.
- `datasets/`: generated capacity/latency traces and the local copy of the
  Starlink traces used by the experiment scripts.
- `IMPLEMENTATION_STATUS.md`: correspondence between paper components and
  source files, including known reproduction limits.

The ns-3 trees are overlays for ns-3.41. Copy an overlay into a clean ns-3.41
tree before configuring it. Each component README contains its build command.

## License

The code is distributed under GPL-2.0. The ns3-ai and Linux-derived files retain
their original license terms. Confirm redistribution permission for third-party
measurement traces before publishing them in a public repository.
