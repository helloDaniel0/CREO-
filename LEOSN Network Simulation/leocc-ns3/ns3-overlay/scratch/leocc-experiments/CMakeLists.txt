build_exec(
  EXECNAME leocc-connected-eval
  SOURCE_FILES ../leocc-connected-test.cc
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libapplications}
    ${libinternet}
    ${libtraffic-control}
    ${libflow-monitor}
  EXECUTABLE_DIRECTORY_PATH ${CMAKE_OUTPUT_DIRECTORY}/scratch/leocc-experiments
)
