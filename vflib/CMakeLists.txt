cmake_minimum_required(VERSION 3.8)
project(vflib_3_0_1)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        include/argedit.hpp
        include/argloader.hpp
        include/argraph.hpp
        include/benchmark.hpp
        include/degmsorter.hpp
        include/dict.h
        include/error.hpp
        include/match.hpp
        include/memory.hpp
        include/nodeclassifier.hpp
        include/nodesorter.hpp
        include/probability_strategy.hpp
        include/vf2_state.hpp
        include/vf2_sub_state.hpp
        include/vf3_state.hpp
        include/vf3_sub_state.hpp
        include/vf3l_sub_state.hpp
        src/main.cpp)

add_executable(vflib_3_0_1 ${SOURCE_FILES})