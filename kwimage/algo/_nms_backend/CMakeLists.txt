

function(cpu_cython_module cython_source module_name)
  # Translate Cython into C/C++
  add_cython_target(${module_name} "${cython_source}" C OUTPUT_VAR sources)
  # Create C++ library. Specify include dirs and link libs as normal
  add_library(${module_name} MODULE ${sources})
  target_include_directories(
    ${module_name}
    PUBLIC
      ${NumPy_INCLUDE_DIRS}
      ${PYTHON_INCLUDE_DIRS}
  )
  #target_link_libraries(${module_name} ${PYTHON_LIBRARIES})
  #target_link_libraries(${module_name})
  #${PYTHON_LIBRARIES})

  target_compile_definitions(${module_name} PUBLIC
    "NPY_NO_DEPRECATED_API"
    #"NPY_1_7_API_VERSION=0x00000007"
    )

  # Transform the C++ library into an importable python module
  python_extension_module(${module_name})
  # Install the C++ module to the correct relative location
  # (this will be an inplace build if you use `pip install -e`)
  file(RELATIVE_PATH _install_dest "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
  install(TARGETS ${module_name} LIBRARY DESTINATION "${_install_dest}")
endfunction(cpu_cython_module)

option(BUILD_CPU_NMS "Cython CPU-based NMS" True)
if (BUILD_CPU_NMS)

  #set(cython_source "cpu_nms.pyx")
  #set(module_name "cpu_nms")
  cpu_cython_module("cpu_nms.pyx" "cpu_nms")

  #set(cython_source "cpu_soft_nms.pyx")
  #set(module_name "cpu_soft_nms.pyx")
  cpu_cython_module("cpu_soft_nms.pyx" "cpu_soft_nms")

endif()



option(BUILD_GPU_NMS "Cython GPU-based NMS" True)
if (USE_CUDA AND BUILD_GPU_NMS)
  set(cython_source "gpu_nms.pyx")
  set(module_name "gpu_nms")

  # Translate Cython into C/C++
  add_cython_target(${module_name} "${cython_source}" CXX OUTPUT_VAR sources)

  # Add any other non-cython dependencies to the sources
  list(APPEND sources
    "nms_kernel.cu"
    "gpu_nms.hpp"
  )
  message(STATUS "THE sources = ${sources}")

  # Create C++ library. Specify include dirs and link libs as normal
  add_library(${module_name} MODULE ${sources})
  target_include_directories(${module_name} PUBLIC
    ${NumPy_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  #target_link_libraries(${module_name} ${CUDA_LIBRARIES} ${PYTHON_LIBRARIES})
  target_link_libraries(${module_name} ${CUDA_LIBRARIES})

  target_compile_definitions(${module_name} PUBLIC
    "NPY_NO_DEPRECATED_API"
    #"NPY_1_7_API_VERSION=0x00000007"
    )

  # Transform the C++ library into an importable python module
  python_extension_module(${module_name})

  # Install the C++ module to the correct relative location
  # (this will be an inplace build if you use `pip install -e`)
  file(RELATIVE_PATH _install_dest "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
  install(TARGETS ${module_name} LIBRARY DESTINATION "${_install_dest}")
endif()

