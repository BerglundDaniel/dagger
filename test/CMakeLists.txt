MESSAGE( STATUS "Fetching gtest" )
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
  #GIT_REPOSITORY https://github.com/google/googletest.git
  #GIT_TAG        master
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

add_subdirectory(mocks)
add_subdirectory(util)
add_subdirectory(container)
add_subdirectory(cuda)

add_executable(${DAGGER_TEST_LIB} ${TEST_SOURCE})

target_include_directories(${DAGGER_TEST_LIB} PUBLIC ${PROJECT_SOURCE_DIR}/test ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${DAGGER_TEST_LIB} PRIVATE ${DAGGER_LIB} gtest_main gmock_main)

#add_test(NAME daggertests COMMAND ${DAGGER_TEST_LIB})

include(GoogleTest)
gtest_discover_tests(${DAGGER_TEST_LIB})
