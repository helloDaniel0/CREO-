build_exec(
  EXECNAME creo-udp-handover-notification
  SOURCE_FILES creo-udp-handover-notification.cc
  LIBRARIES_TO_LINK
    ${libapplications}
    ${libcore}
    ${libinternet}
    ${libnetwork}
    ${libpoint-to-point}
  EXECUTABLE_DIRECTORY_PATH ${CMAKE_OUTPUT_DIRECTORY}/scratch/creo-handover-notification
)
