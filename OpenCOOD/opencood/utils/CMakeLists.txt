find_package(Cython MODULE REQUIRED)
include(UseCython)

Cython_transpile(${CMAKE_CURRENT_SOURCE_DIR}/box_overlaps.pyx
    LANGUAGE CXX
    OUTPUT_VARIABLE BOX_OVERLAPS_CPP
)

python_add_library(box_overlaps MODULE WITH_SOABI ${BOX_OVERLAPS_CPP})

target_include_directories(box_overlaps PRIVATE
    ${Python3_NumPy_INCLUDE_DIRS}
)

target_compile_options(box_overlaps PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:Debug>:-g -O0>
)

install(TARGETS box_overlaps LIBRARY DESTINATION opencood/utils)
