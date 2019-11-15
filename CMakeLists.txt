cmake_minimum_required(VERSION 3.0.0)
project(puckshootai VERSION 0.1.0)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set( CMAKE_VERBOSE_MAKEFILE ON )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

#set(CMAKE_EXE_LINKER_FLAGS "-Wl,-pie -fPIE -fPIC -O2 -fmessage-length=0 -D_FORTIFY_SOURCE=2 -fstack-protector -funwind-tables -fasynchronous-unwind-tables")
set(CMAKE_EXE_LINKER_FLAGS "-fPIE -fPIC -O2 -fmessage-length=0 -D_FORTIFY_SOURCE=2 -fstack-protector -funwind-tables -fasynchronous-unwind-tables")
set(CMAKE_MODULE_LINKER_FLAGS "-Wl,-pie -fPIE -fPIC -O2 -fmessage-length=0 -D_FORTIFY_SOURCE=2 -fstack-protector -funwind-tables -fasynchronous-unwind-tables")
set(CMAKE_SHARED_LINKER_FLAGS "-fPIE -fPIC -O2 -fmessage-length=0 -D_FORTIFY_SOURCE=2 -fstack-protector -funwind-tables -fasynchronous-unwind-tables")
set(CMAKE_STATIC_LINKER_FLAGS "-fPIE -fPIC -O2 -fmessage-length=0 -D_FORTIFY_SOURCE=2 -fstack-protector -funwind-tables -fasynchronous-unwind-tables")
add_compile_options(-Wall -Wextra -fPIE -fPIC -O2 -fmessage-length=0 -D_FORTIFY_SOURCE=2 -fstack-protector -funwind-tables -fasynchronous-unwind-tables)

#set(THREADS_PREFER_PTHREAD_FLAG ON)
#find_package(Threads REQUIRED)
find_package(Armadillo REQUIRED)

find_program(SCONS_BIN NAMES scons)
if (SCONS_BIN)
message("Found scons: ${SCONS_BIN}")
else()
message(FATAL_ERROR "Error: Godot builder scons is required (found: ${SCONS_BIN})")
endif()

if (EXISTS "${CMAKE_SOURCE_DIR}/godot-cpp/bin/libgodot-cpp.linux.debug.64.a")
message("Found godot: ${CMAKE_SOURCE_DIR}/godot-cpp/bin/libgodot-cpp.linux.debug.64.a")
else()
message("Not found ${CMAKE_SOURCE_DIR}/godot-cpp/bin/libgodot-cpp.linux.debug.64.a - will run build")
add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/godot-cpp/bin/libgodot-cpp.linux.debug.64.a COMMAND ${SCONS_BIN} generate_bindings=yes WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/godot-cpp)
endif()

include_directories(${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/Catch2/single_include/catch2 ${CMAKE_SOURCE_DIR}/godot-cpp/include ${CMAKE_SOURCE_DIR}/godot-cpp/include/core ${CMAKE_SOURCE_DIR}/godot-cpp/include/gen ${CMAKE_SOURCE_DIR}/godot-cpp/godot_headers ${ARMADILLO_INCLUDE_DIRS})
link_directories(${PROJECT_BINARY_DIR} Catch2 ${CMAKE_SOURCE_DIR}/godot-cpp/bin)

add_custom_target(libgodot DEPENDS ${CMAKE_SOURCE_DIR}/godot-cpp/bin/libgodot-cpp.linux.debug.64.a)

add_library(${PROJECT_NAME} SHARED src/godot_lib.cpp src/${PROJECT_NAME}.cpp)
add_dependencies(${PROJECT_NAME} libgodot)
# set_target_properties(${PROJECT_NAME} PROPERTIES
#     VERSION ${PROJECT_VERSION}
#     SOVERSION 0
# )
target_link_libraries(${PROJECT_NAME} godot-cpp.linux.debug.64 ${ARMADILLO_LIBRARIES})

set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/Catch2;${CMAKE_SOURCE_DIR}/Catch2/contrib;${CMAKE_PREFIX_PATH}")
set(CMAKE_INCLUDE_PATH "${CMAKE_SOURCE_DIR}/Catch2;${CMAKE_SOURCE_DIR}/Catch2/contrib;${CMAKE_INCLUDE_PATH}")
set(CATCH_INSTALL_DOCS OFF)
set(CATCH_INSTALL_HELPERS ON)
add_subdirectory(Catch2 Catch2)

include(CTest)
enable_testing()

if (BUILD_TESTING)
include_directories(${CMAKE_SOURCE_DIR}/tests)

add_executable(test_normalize tests/normalize.cpp tests/test.cpp)
add_dependencies(test_normalize ${PROJECT_NAME})
target_link_libraries(test_normalize Catch2::Catch2 ${ARMADILLO_LIBRARIES} ${PROJECT_NAME})

add_executable(test_sigmoid tests/sigmoid.cpp tests/test.cpp)
add_dependencies(test_sigmoid ${PROJECT_NAME})
target_link_libraries(test_sigmoid Catch2::Catch2 ${ARMADILLO_LIBRARIES} ${PROJECT_NAME})

add_executable(test_train tests/train.cpp tests/test.cpp)
add_dependencies(test_train ${PROJECT_NAME})
target_link_libraries(test_train Catch2::Catch2 ${ARMADILLO_LIBRARIES} ${PROJECT_NAME})

add_executable(test_predict tests/predict.cpp tests/test.cpp)
add_dependencies(test_predict ${PROJECT_NAME})
target_link_libraries(test_predict Catch2::Catch2 ${ARMADILLO_LIBRARIES} ${PROJECT_NAME})

include(${CMAKE_SOURCE_DIR}/Catch2/contrib/ParseAndAddCatchTests.cmake)

ParseAndAddCatchTests(test_normalize)
ParseAndAddCatchTests(test_sigmoid)
ParseAndAddCatchTests(test_train)
ParseAndAddCatchTests(test_predict)
endif()
