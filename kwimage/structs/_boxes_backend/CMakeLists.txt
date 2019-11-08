
option(BUILD_BOXES_CYTHON "Are your IoU ops TOO SLOW? Try Cython!" True)
if (BUILD_BOXES_CYTHON)

  set(cython_source "cython_boxes.pyx")
  set(module_name "cython_boxes")

  # Translate Cython into C/C++
  add_cython_target(${module_name} "${cython_source}" C OUTPUT_VAR sources)

  # Create C++ library. Specify include dirs and link libs as normal
  add_library(${module_name} MODULE ${sources})
  target_include_directories(${module_name} PUBLIC ${NumPy_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
  #target_link_libraries(${module_name} ${PYTHON_LIBRARIES})

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
