find_package(Cython MODULE REQUIRED)
include(UseCython)

Cython_transpile(${CMAKE_CURRENT_SOURCE_DIR}/box_overlaps.pyx
    LANGUAGE CXX
    OUTPUT_VARIABLE BOX_OVERLAPS_CPP
)

add_library(box_overlaps MODULE ${BOX_OVERLAPS_CPP})

set_target_properties(box_overlaps PROPERTIES
    PREFIX ""
    OUTPUT_NAME "box_overlaps"
    SUFFIX ".${Python3_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}"
)

target_include_directories(box_overlaps PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

target_link_libraries(box_overlaps PRIVATE Python3::Module)

target_compile_options(box_overlaps PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:Debug>:-g -O0>
)

set_target_properties(box_overlaps PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

install(TARGETS box_overlaps LIBRARY DESTINATION opencood/utils)
