set(CREO_COMMON_SOURCES
        use-gym/tcp-rl-env.cc
        use-gym/tcp-rl.cc
)

set(CREO_COMMON_LIBRARIES
        ${libai}
        ${libcore}
        ${libpoint-to-point}
        ${libnetwork}
        ${libapplications}
        ${libinternet}
        ${libtraffic-control}
        ${libflow-monitor}
)

build_lib_example(
        NAME ns3ai_creo_single
        SOURCE_FILES
            use-gym/tcp-sim-creo-single.cc
            ${CREO_COMMON_SOURCES}
        LIBRARIES_TO_LINK
            ${CREO_COMMON_LIBRARIES}
)

build_lib_example(
        NAME ns3ai_creo_multi
        SOURCE_FILES
            use-gym/tcp-sim-creo-multi.cc
            ${CREO_COMMON_SOURCES}
        LIBRARIES_TO_LINK
            ${CREO_COMMON_LIBRARIES}
)

# Dedicated target used by the connected-phase ablation scripts.  It shares the
# single-flow topology so every variant sees exactly the same network process.
build_lib_example(
        NAME ns3ai_creo_ablation
        SOURCE_FILES
            use-gym/tcp-sim-creo-single.cc
            ${CREO_COMMON_SOURCES}
        LIBRARIES_TO_LINK
            ${CREO_COMMON_LIBRARIES}
)
