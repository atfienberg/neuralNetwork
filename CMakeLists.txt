cmake_minimum_required (VERSION 2.6)

project (neuralNetTest)

execute_process(COMMAND root-config --cflags OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE ROOTCFLAGS)
execute_process(COMMAND root-config --libs OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE ROOT_LIBRARIES)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -std=c++11 ${ROOTCFLAGS}")

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cxx)
file(GLOB headers ${PROJECT_SOURCE_DIR}/src/*.hh)

list(APPEND sources json11/json11.cpp)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/json11)


add_executable(pulseDiscriminator ${PROJECT_SOURCE_DIR}/src/executables/pulseDiscriminationTest.cxx ${sources} ${headers})

add_executable(pileupSeparator ${PROJECT_SOURCE_DIR}/src/executables/pileupSeparator.cxx ${sources} ${headers})

add_executable(spatialSeparator ${PROJECT_SOURCE_DIR}/src/executables/spatialSeparator.cxx ${sources} ${headers})

target_link_libraries (pulseDiscriminator ${ROOT_LIBRARIES})

target_link_libraries (pileupSeparator ${ROOT_LIBRARIES})

target_link_libraries (spatialSeparator ${ROOT_LIBRARIES})