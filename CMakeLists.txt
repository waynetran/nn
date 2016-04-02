cmake_minimum_required(VERSION 3.4)
project(nn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
        Common.h
        Net.h
        Utils.h

        main.cpp
        Net.cpp
        )
add_executable(nn ${SOURCE_FILES})