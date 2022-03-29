add_subdirectory(transfer)

set(HEADER_LIST
  ${HEADER_LIST}
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_types_concept.h
  ${CMAKE_CURRENT_SOURCE_DIR}/device.h
  ${CMAKE_CURRENT_SOURCE_DIR}/stream.h
  ${CMAKE_CURRENT_SOURCE_DIR}/stream_factory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/event.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_adapter.h
  PARENT_SCOPE
)