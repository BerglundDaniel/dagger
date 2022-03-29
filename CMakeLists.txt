cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(DAGGER VERSION 1.0 LANGUAGES CUDA CXX)

include(FetchContent)

#Options
option(DAGGER_TESTS "Enable tests" ON)
option(VERBOSE "Set CMake and compilation to verbose" on)
option(DAGGER_CUDA_ARCHITECTURES "Which cuda architecture to compile for (min 60)" 80)

if(${DAGGER_CUDA_ARCHITECTURES} LESS 60) #TODO incorrect somehow
    message( FATAL_ERROR "Requires DAGGER_CUDA_ARCHITECTURES 60 or more" )
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(DAGGER_LIB DAGGER_CC)
set(DAGGER_NVCC_LIB DAGGER_NVCC)
set(DAGGER_TEST_LIB DAGGER_TEST_CC)

if(${VERBOSE})
  set (CMAKE_VERBOSE_MAKEFILE true)
  set (CUDA_VERBOSE_BUILD ON)
endif(${VERBOSE})

MESSAGE( STATUS "Checking for CUDAToolkit" )
find_package(CUDAToolkit REQUIRED)

#Print some info
MESSAGE( STATUS "change a configuration variable with: cmake -D<Variable>=<Value>" )
MESSAGE( STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}" )
MESSAGE( STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}" )
MESSAGE( STATUS "CUDA_CUBLAS_LIBRARIES = ${CUDA_CUBLAS_LIBRARIES}" )
MESSAGE( STATUS "CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}" )
MESSAGE( STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES = ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" )
MESSAGE( STATUS "CUDA_PATH = ${CUDA_PATH}" )

MESSAGE( STATUS "CUDA_LIBRARIES = ${${CUDA_LIBRARIES}}" )


#Add all the relevant sub directories
add_subdirectory(include)
add_subdirectory(src)

#Testing only available if this is the main app
#Override with DAGGER_BUILD_TESTING
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR DAGGER_BUILD_TESTING)
   AND DAGGER_TESTS)
  MESSAGE( STATUS "Dagger tests enabled" )
  add_subdirectory(test)
else()
  MESSAGE( STATUS "Dagger tests disabled" )
endif()
