cmake_minimum_required(VERSION 3.18)
project(llmengine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# LLMENGINE_MODE selects the kernel/compile profile.
# - CORRECTNESS: -O2, no -ffast-math, FP32 scalar reference path. Deterministic.
# - PERF (default): -O3, -ffast-math, NEON FP16/INT8. Approximations allowed.
if(NOT LLMENGINE_MODE)
  set(LLMENGINE_MODE "PERF")
endif()

if(LLMENGINE_MODE STREQUAL "CORRECTNESS")
  message(STATUS "llmengine build mode: CORRECTNESS (deterministic FP32 reference)")
  add_compile_options(-O2 -fno-fast-math -fno-finite-math-only)
  add_compile_definitions(LLMENGINE_CORRECTNESS=1)
  # Debug binding (_debug_weight_ptr) is on by default in correctness builds
  # because the Phase 0 alias tests need it. Production-style builds should
  # disable it via -DLLMENGINE_DEBUG_BINDINGS=OFF at configure time.
  if(NOT DEFINED LLMENGINE_DEBUG_BINDINGS)
    set(LLMENGINE_DEBUG_BINDINGS ON)
  endif()
else()
  message(STATUS "llmengine build mode: PERF (NEON + fast-math)")
  add_compile_options(-O3 -ffast-math -funroll-loops -fno-omit-frame-pointer)
  add_compile_definitions(LLMENGINE_PERF=1)
  if(NOT DEFINED LLMENGINE_DEBUG_BINDINGS)
    set(LLMENGINE_DEBUG_BINDINGS OFF)
  endif()
endif()

if(LLMENGINE_DEBUG_BINDINGS)
  add_compile_definitions(LLMENGINE_DEBUG_BINDINGS=1)
  message(STATUS "  LLMENGINE_DEBUG_BINDINGS: ON (exposes _debug_weight_ptr)")
endif()

# Apple Silicon already enables NEON; we detect via __ARM_NEON in code.
# std::thread for parallelism, not OpenMP (avoids macOS libomp duplication issues).
find_package(Threads REQUIRED)

# Prefer the venv python over a system one. Without this, CMake's default
# Python discovery may pick a higher-version system python that lacks
# pybind11/torch.
set(Python3_FIND_VIRTUALENV ONLY)
set(Python3_FIND_STRATEGY    LOCATION)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Discover pybind11 from the active Python env.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
find_package(pybind11 CONFIG REQUIRED)

# Sources grow as phases land:
#   P0: weights.cpp, engine.cpp
#   P1: + kernels.cpp, model.cpp
#   P3: + kv_cache.cpp
#   P4: (more kernels in same file)
#   P7: + sampler.cpp, scheduler.cpp
set(CORE_SRC
  src/weights.cpp
  src/engine.cpp
  src/kernels.cpp
  src/model.cpp
  src/kv_cache.cpp
  src/paged_kv.cpp
  src/scheduler.cpp
  src/parallel.cpp
)

add_library(llmengine_core STATIC ${CORE_SRC})
target_include_directories(llmengine_core PUBLIC include)
target_include_directories(llmengine_core PRIVATE third_party)
target_compile_options(llmengine_core PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(llmengine_core PUBLIC Threads::Threads)

pybind11_add_module(_llmengine src/bindings.cpp)
target_link_libraries(_llmengine PRIVATE llmengine_core)
target_include_directories(_llmengine PRIVATE include)

# Drop the compiled module next to the Python package for in-place use in PERF
# mode (the default `import llmengine` path). For CORRECTNESS mode keep it
# inside the build tree so the two coexist; tests prepend the correctness
# build dir to sys.path before importing. The build tree needs an __init__.py
# in the package dir so Python actually treats it as the `llmengine` package
# when sys.path[0] = build/correctness/python.
if(LLMENGINE_MODE STREQUAL "CORRECTNESS")
  set_target_properties(_llmengine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/llmengine
  )
  # Mirror the Python package into the correctness build tree so its
  # _llmengine.so loads as part of a real `llmengine` package. Use
  # add_custom_command (not configure_file) so edits to __init__.py /
  # server.py propagate on every `cmake --build` rather than requiring a
  # re-configure, previously stale copies caused server tests to fail in
  # correctness while passing in perf.
  set(LLMENGINE_PY_FILES __init__.py server.py)
  set(LLMENGINE_PY_OUTPUTS "")
  foreach(f ${LLMENGINE_PY_FILES})
    add_custom_command(
      OUTPUT  ${CMAKE_BINARY_DIR}/python/llmengine/${f}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${CMAKE_SOURCE_DIR}/python/llmengine/${f}
              ${CMAKE_BINARY_DIR}/python/llmengine/${f}
      DEPENDS ${CMAKE_SOURCE_DIR}/python/llmengine/${f}
      COMMENT "Mirroring python/llmengine/${f} into correctness build tree"
    )
    list(APPEND LLMENGINE_PY_OUTPUTS ${CMAKE_BINARY_DIR}/python/llmengine/${f})
  endforeach()
  add_custom_target(llmengine_py ALL DEPENDS ${LLMENGINE_PY_OUTPUTS})
  add_dependencies(_llmengine llmengine_py)
else()
  set_target_properties(_llmengine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/llmengine
  )
endif()
