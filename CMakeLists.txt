###
# CMakeLists for kwimage C/C++/Cuda Python extention modules
#
# Notes:
# https://github.com/Erotemic/netharn/issues/7
# https://github.com/amueller/word_cloud/pull/42ci

cmake_minimum_required(VERSION 3.11.0)

project(kwimage LANGUAGES C CXX CUDA)

# Setup basic python stuff and ensure we have skbuild
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/CMake")
include( kwimage-setup-python )

option(USE_CUDA "Build with CUDA" "True")
if (USE_CUDA)
    find_package(CUDA)
    if (CUDA_FOUND)
        message(STATUS "CUDA found (include: ${CUDA_INCLUDE_DIRS}, lib: ${CUDA_LIBRARIES})")
        include_directories(${CUDA_INCLUDE_DIRS})
    else(CUDA_FOUND)
        message(STATUS "CUDA not found, CUDA library will not be built")
        set(USE_CUDA OFF)
    endif(CUDA_FOUND)
endif(USE_CUDA)


#if (USE_CUDA)
#  project(kwimage LANGUAGES C CXX CUDA)
#else()
#  project(kwimage LANGUAGES C CXX)
#endif()


###
# Status string for debugging
#
set(PYTHON_SETUP_STATUS "
  * PYTHON_EXECUTABLE = \"${PYTHON_EXECUTABLE}\"

  * PYTHON_INCLUDE_DIR = \"${PYTHON_INCLUDE_DIR}\"
  * PYTHON_LIBRARY = \"${PYTHON_LIBRARY}\"
  * PYTHON_LIBRARY_DEBUG = \"${PYTHON_LIBRARY_DEBUG}\"

  * skbuild_location = \"${skbuild_location}\"
  * skbuild_cmake_dir = \"${skbuild_cmake_dir}\"
")




option(KWIMAGE_VERBOSE "Print extra info" False)

find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)


add_subdirectory("kwimage/structs/_boxes_backend")
add_subdirectory("kwimage/structs/_mask_backend")
add_subdirectory("kwimage/algo/_nms_backend")


set(KWIMAGE_CONFIG_STATUS "
PYTHON_CONFIG_STATUS
====================

Include Dirs:
  * CUDA_INCLUDE_DIRS = \"${CUDA_INCLUDE_DIRS}\"
  * NumPy_INCLUDE_DIRS = \"${NumPy_INCLUDE_DIRS}\"
  * PYTHON_INCLUDE_DIRS = \"${PYTHON_INCLUDE_DIRS}\"

status(Cython):
  * CYTHON_FOUND = \"${CYTHON_FOUND}\"
  * CYTHON_EXECUTABLE = \"${CYTHON_EXECUTABLE}\"
  * CYTHON_VERSION = \"${CYTHON_VERSION}\"

status(NumPy):
  * NumPy_FOUND = \"${NumPy_FOUND}\"
  * NumPy_INCLUDE_DIRS = \"${NumPy_INCLUDE_DIRS}\"
  * NumPy_VERSION = \"${NumPy_VERSION}\"
  * NumPy_CONV_TEMPLATE_EXECUTABLE = \"${NumPy_CONV_TEMPLATE_EXECUTABLE}\"
  * NumPy_FROM_TEMPLATE_EXECUTABLE = \"${NumPy_FROM_TEMPLATE_EXECUTABLE}\"

status(PythonExtensions):
  * PYTHON_PREFIX = \"${PYTHON_PREFIX}\"
  * PYTHON_SITE_PACKAGES_DIR = \"${PYTHON_SITE_PACKAGES_DIR}\"
  * PYTHON_RELATIVE_SITE_PACKAGES_DIR = \"${PYTHON_RELATIVE_SITE_PACKAGES_DIR}\"
  * PYTHON_SEPARATOR = \"${PYTHON_SEPARATOR}\"
  * PYTHON_PATH_SEPARATOR = \"${PYTHON_PATH_SEPARATOR}\"
  * PYTHON_EXTENSION_MODULE_SUFFIX = \"${PYTHON_EXTENSION_MODULE_SUFFIX}\"

status(python-setup)
${PYTHON_SETUP_STATUS}
")

if (KWIMAGE_VERBOSE)
  message(STATUS ${KWIMAGE_CONFIG_STATUS})
endif ()

