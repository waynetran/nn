cmake_minimum_required(VERSION 3.4)
project(nn)

set(SOURCE_FILES
        src/Net.cpp
        )

add_library(nn ${SOURCE_FILES})

# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
target_include_directories(nn PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)


# 'make install' to the correct location
install(TARGETS nn EXPORT nnConfig
    ARCHIVE  DESTINATION lib
    LIBRARY  DESTINATION lib
    RUNTIME  DESTINATION bin)  # This is for Windows

install(DIRECTORY include/ DESTINATION include
    PATTERN ".svn" EXCLUDE
    )

# This makes the project importable from the install directory
# Put config file in per-project dir (name MUST match), can also
# just go into <prefix>/cmake.
install(EXPORT nnConfig DESTINATION share/nn/cmake)

# This makes the project importable from the build directory
export(TARGETS nn FILE nnConfig.cmake)

add_executable(test_nn
    test/main.cpp)

target_link_libraries(test_nn
    nn)

add_test(test_nn test_nn)
