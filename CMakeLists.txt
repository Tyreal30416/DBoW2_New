cmake_minimum_required(VERSION 2.8)
project(DBoW2)

option(WITH_OPENMP   "Enable OpenMP multithreading"     ON)
option(BUILD_DBoW2   "Build DBoW2"            ON)
option(BUILD_Demo    "Build demo application" ON)
option(BUILD_DEBUG "BUILD demo with debug mode" OFF)

IF(BUILD_DEBUG)
	SET(CMAKE_BUILD_TYPE "Debug") # Release, RelWithDebInfo
	SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
	SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
ELSE()
	SET(CMAKE_BUILD_TYPE "Release")
ENDIF()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")



set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/.cmake")
include(UpdateCacheVariable)

set(HDRS
	include/DBoW2/BowVector.h           include/DBoW2/FBrief.h              include/DBoW2/FSurf64.h
	include/DBoW2/QueryResults.h        include/DBoW2/TemplatedDatabase.h   include/DBoW2/FORB.h
	include/DBoW2/DBoW2.h               include/DBoW2/FClass.h              include/DBoW2/FeatureVector.h
	include/DBoW2/ScoringObject.h       include/DBoW2/TemplatedVocabulary.h include/DBoW2/FSift.h)
set(SRCS
	src/BowVector.cpp     src/FBrief.cpp        src/FSurf64.cpp       src/FORB.cpp
	src/FeatureVector.cpp src/QueryResults.cpp  src/ScoringObject.cpp src/FSift.cpp)

if (WITH_OPENMP)
    find_package(OpenMP QUIET)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    else()
        message("")
        UPDATE_CACHE_VARIABLE(WITH_OPENMP OFF)
    endif(OPENMP_FOUND)
endif(WITH_OPENMP)

#find_package(OpenCV REQUIRED)
#find_package(DLib REQUIRED)
find_package(Boost REQUIRED)
#set opencv2.4
set(OpenCV_INCLUDE_DIRS /usr/local/include ${OpenCV_INCLUDE_DIRS})
set(OpenCV_LIB_DIRS /usr/local/lib)
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_calib3d opencv_video opencv_features2d opencv_ml opencv_highgui opencv_objdetect opencv_contrib opencv_legacy opencv_nonfree)

message(${OpenCV_LIBS})

#set dlib
set(DLIB_INCLUDE_DIRS /usr/local/include/ ${DLIB_INCLUDE_DIRS})
set(DLIB_LIB_DIR /usr/local/lib/ )
set(DLIB_LIBRARIES DLib)

link_directories(${OpenCV_LIB_DIRS} ${DLIB_LIB_DIR})
if(BUILD_DBoW2)
	include_directories(include/DBoW2/ ${OpenCV_INCLUDE_DIRS} ${DLIB_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
	add_library(${PROJECT_NAME} SHARED ${SRCS})
	target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${DLIB_LIBRARIES})
endif(BUILD_DBoW2)

if(BUILD_Demo)
	add_executable(demo demo/demo.cpp)
	target_link_libraries(demo ${PROJECT_NAME} ${OpenCV_LIBS} ${DLIB_LIBRARIES})

	#build trainVoc
	set(BUILD_FILES
		demo/ORBExtractor.h
		demo/ORBExtractor.cpp
		demo/trainVoc.h 
		demo/trainVoc.cpp 
		demo/main.cpp)

	add_executable(trainVoc ${BUILD_FILES})
	target_link_libraries(trainVoc ${PROJECT_NAME} ${OpenCV_LIBS} ${DLIB_LIBRARIES})

	file(COPY demo/images DESTINATION ${CMAKE_BINARY_DIR}/)
endif(BUILD_Demo)

configure_file(src/DBoW2.cmake.in
	"${PROJECT_BINARY_DIR}/DBoW2Config.cmake" @ONLY)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
if(BUILD_DBoW2)
	install(DIRECTORY include/DBoW2 DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
endif()
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DBoW2Config.cmake"
	DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(FILES "${PROJECT_BINARY_DIR}/DBoW2Config.cmake"
	DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/DBoW2/)
