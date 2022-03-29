MESSAGE( STATUS "Building Dagger" )
set(CMAKE_CXX_EXTENSIONS Off)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")  TODO add it to the rest in the correct way
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall")

add_subdirectory(util)
add_subdirectory(cuda)
add_subdirectory(container)
add_subdirectory(kernel)

add_library(${DAGGER_LIB} SHARED ${SOURCE} ${HEADER_LIST}) #TODO is header list actually needed?
#add_library(${DAGGER_LIB} SHARED ${SOURCE})

target_include_directories(${DAGGER_LIB} PUBLIC ${PROJECT_SOURCE_DIR}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

MESSAGE( STATUS "Seting compiler options" )

set_target_properties(${DAGGER_LIB} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_features(${DAGGER_LIB} PUBLIC cxx_std_20)
#TODO cuda c++11
set_target_properties(${DAGGER_LIB} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${DAGGER_LIB} PROPERTIES CUDA_ARCHITECTURES ${DAGGER_CUDA_ARCHITECTURES})

target_compile_options(${DAGGER_LIB} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --extended-lambda
                       --use_fast_math
                       -O3
                       >)

target_link_libraries(${DAGGER_LIB} PUBLIC CUDA::cublas)
target_link_libraries(${DAGGER_LIB} PUBLIC CUDA::cudart)

target_compile_definitions(${DAGGER_LIB} PUBLIC DAGGER_TENSOR_MULT=8)

#TODO outdated?
set(CUDA_PROPAGATE_HOST_FLAGS off) #Some c++ flags that gets passed to NVCC -Xcompiler can cause problems

#TODO wrong place outdated?
if(CMAKE_BUILD_TYPE MATCHES DEBUG)
  MESSAGE( STATUS "Debug build." )
  #target_compile_options(JSONUtils PRIVATE -Werror)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g") #TODO fix
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g) #TODO fix
  target_compile_definitions(${DAGGER_LIB} PUBLIC DAGGER_DEBUG)
  #target_compile_definitions(${DAGGER_NVCC_LIB} PUBLIC DAGGER_DEBUG)
endif()


source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST}
)
