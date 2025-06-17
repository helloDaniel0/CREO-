Satellite Time-Varying Link Capacity and Handover Interruption Events Generator
======

To accelerate the simulation, we precompute the data for the two core challenges in LEOSNs: time-varying link capacity and handover interruptions, decoupling them from ns-3, avoiding the overhead of online generation within ns-3 that could degrade simulation speed. **The precomputed data is directly imported into the link configuration of ns-3.**

## Step 1: Satellites and GSs Postion Generation
We adopt the node placement methodology used in [SaTCP](https://github.com/XuyangCaoUCSD/LeoEM) and [StarPerf](https://github.com/SpaceNetLab/StarPerf_Simulator) for constructing the positions of satellites and GS in LEOSNs.

Specifically, we store the location of each node at every second over one orbital period to capture the time-varying topology of the satellite constellation. For satellite positions, here we demonstrate data generated with [Starlink constellation parameters](https://www.dropbox.com/scl/fo/jmfet91n4za2c6j5fwkuj/AOr6_cDeZrHAga7jQVyj_J0?rlkey=wvdr5c3bw7ddld091l08bvdjx&e=1&dl=0), as implemented in StarPerf. For ground stations, we adopt all [GS locations](https://github.com/XuyangCaoUCSD/LeoEM/blob/main/ground_stations.xlsx) `ground_stations.xlsx` mentioned in StarPerf, based on the official Starlink deployment.

This precomputed positional data enables accurate modeling of dynamic satellite-to-GS and inter-satellite connectivity in simulations.

## Step 2: Distance and Delay Generation
Based on the node positions and a predefined connectivity threshold (i.e., the maximum distance at which two nodes can communicate), we calculate the distance and corresponding propagation delay between all feasible node pairs.

## Step 3: Routing Generation (Including Bandwitdh and Delay Calculation)
Given the positions of dish and PoP, we compute the routing paths based on the shortest-path criterion.

We demonstrate BPL routing computation process here. To run the routing algorithm, execute: `python bp_routing.py`. To visualize the actual routing results, you can run: `python bp_routing_plot.py`. 

We compute the bandwidth of each segment along the routing path using the Shannon formula and store the results. The computations for bandwidth and propagation delay are implemented in `utility.py`. Here, we only illustrate large-scale fading (e.g., free-space path loss). You are welcome to extend the script by incorporating additional fading models such as small-scale fading or shadowing, depending on your simulation needs.
