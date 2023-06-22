cmake_minimum_required(VERSION 3.13)
project(VCGLib_point2mesh)

option(VCG_BUILD_EXAMPLES "Build a set of examples of the library" OFF)

add_subdirectory(/home/ggutow/eclipse-workspace/vcglib/ vcgbuild)
if (VCG_HEADER_ONLY)
    set(SOURCES
        vcglib_point2mesh.cpp
        ${VCG_INCLUDE_DIRS}/wrap/ply/plylib.cpp)
endif()

add_executable(vcglib_point2mesh
	${SOURCES})

target_link_libraries(
	vcglib_point2mesh
	PUBLIC
		vcglib
	)
