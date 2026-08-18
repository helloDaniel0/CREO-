# ns3-ai Runtime for CREO+

This module contains only the Gym/message runtime and the CREO+ TCP examples.
Unrelated upstream examples are intentionally omitted. `CMakeLists.txt` builds
the protobuf interface, Python binding, and `examples/rl-tcp` targets.

The runtime is based on ns3-ai and remains GPL-2.0 licensed. Protobuf, pybind11,
Boost.Program_options, Python development headers, NumPy, Gymnasium, and PyTorch
are required.
