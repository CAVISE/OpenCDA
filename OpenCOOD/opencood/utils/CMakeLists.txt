find_program(CYTHON_EXECUTABLE NAMES cython cython3)

if(NOT CYTHON_EXECUTABLE)
    message(FATAL_ERROR "Cython not found. Install with: pip install cython")
endif()

set(CYTHON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CYTHON_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(BOX_OVERLAPS_PYX ${CYTHON_SOURCE_DIR}/box_overlaps.pyx)
set(BOX_OVERLAPS_CPP ${CYTHON_BINARY_DIR}/box_overlaps.cpp)

add_custom_command(
    OUTPUT ${BOX_OVERLAPS_CPP}
    COMMAND ${CYTHON_EXECUTABLE} -3 --cplus -o ${BOX_OVERLAPS_CPP} ${BOX_OVERLAPS_PYX}
    DEPENDS ${BOX_OVERLAPS_PYX}
    WORKING_DIRECTORY ${CYTHON_SOURCE_DIR}
    VERBATIM
)

add_library(box_overlaps MODULE ${BOX_OVERLAPS_CPP})

set_target_properties(box_overlaps PROPERTIES
    PREFIX ""
    OUTPUT_NAME "box_overlaps"
    SUFFIX "${Python3_EXTENSION_MODULE_SUFFIX}"
)

target_include_directories(box_overlaps PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(box_overlaps PRIVATE ${Python3_LIBRARIES})
target_compile_options(box_overlaps PRIVATE $<$<CONFIG:Release>:-O3 -DNDEBUG> $<$<CONFIG:Debug>:-g -O0>)

install(TARGETS box_overlaps LIBRARY DESTINATION opencood/utils)