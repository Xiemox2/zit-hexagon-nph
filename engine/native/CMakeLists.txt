# sdbench native engine: stable-diffusion.cpp (happyyzy fork, MIT) + ggml-hexagon,
# wrapped by our own small JNI layer. Built standalone with the NDK toolchain file.
cmake_minimum_required(VERSION 3.22)
project(sdbench_native C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(SDCPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../sdcpp")

if(ANDROID)
  set(GGML_HEXAGON ON CACHE BOOL "" FORCE)
  set(PREBUILT_LIB_DIR "android_aarch64" CACHE STRING "" FORCE)
endif()
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(SD_WEBP OFF CACHE BOOL "" FORCE)
set(SD_WEBM OFF CACHE BOOL "" FORCE)

# NPH 0.29 NPG: ggml-opencl (upstream ggml, MIT) for the experimental GPU VAE. It links against our
# own dlopen trampoline shim (no hard dependency on the vendor libOpenCL.so) and is NOT auto-registered;
# libnph registers it only when the NPG toggle is on.
# 0.30.1: NPG (GPU VAE) removed from the default build; -DNPH_OPENCL=ON restores the 0.29/0.30 code (kept in git).
option(NPH_OPENCL "build ggml-opencl for the experimental NPG GPU VAE" OFF)
if(ANDROID AND NPH_OPENCL)
  set(NPH_OCL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../../npg/third_party/OpenCL-Headers-main")
  add_library(nph_ocl_shim STATIC nph_ocl_shim.c)
  target_include_directories(nph_ocl_shim PUBLIC "${NPH_OCL_HEADERS}")
  target_compile_definitions(nph_ocl_shim PRIVATE CL_TARGET_OPENCL_VERSION=300)
  set(GGML_OPENCL ON CACHE BOOL "" FORCE)
  set(GGML_OPENCL_EMBED_KERNELS ON CACHE BOOL "" FORCE)
  set(GGML_OPENCL_USE_ADRENO_KERNELS ON CACHE BOOL "" FORCE)
  set(OpenCL_INCLUDE_DIR "${NPH_OCL_HEADERS}" CACHE PATH "" FORCE)
  set(OpenCL_LIBRARY nph_ocl_shim CACHE STRING "" FORCE)
endif()

add_subdirectory("${SDCPP_DIR}" sdcpp EXCLUDE_FROM_ALL)
if(ANDROID AND NPH_OPENCL AND TARGET ggml)
  target_compile_definitions(ggml PRIVATE GGML_OPENCL_NO_AUTO_REGISTER)
endif()

if(ANDROID)
  add_library(sdbench SHARED sdbench_jni.cpp)
  target_include_directories(sdbench PRIVATE "${SDCPP_DIR}/include")
  target_link_libraries(sdbench PRIVATE stable-diffusion log)
  target_compile_options(sdbench PRIVATE -fvisibility=hidden)
  target_link_options(sdbench PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/sdbench.map" "-Wl,-s"
                      "-Wl,-z,max-page-size=16384")
  foreach(v v79 v81)
    if(TARGET htp-${v})
      add_dependencies(sdbench htp-${v})
    endif()
  endforeach()
endif()

# NPH engine for the main ZIT-HEXAGON app (same sd.cpp, residency-aware JNI).
if(ANDROID)
  add_library(nph SHARED nph_jni.cpp npg_selftest.cpp)
  target_include_directories(nph PRIVATE "${SDCPP_DIR}/include")
  target_include_directories(nph PRIVATE "${SDCPP_DIR}/ggml/include")
  target_link_libraries(nph PRIVATE stable-diffusion log)
  if(NPH_OPENCL)
    target_compile_definitions(nph PRIVATE NPH_HAVE_OPENCL=1)
    target_link_libraries(nph PRIVATE nph_ocl_shim dl)
  endif()
  target_compile_options(nph PRIVATE -fvisibility=hidden)
  target_link_options(nph PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/nph.map" "-Wl,-s"
                      "-Wl,-z,max-page-size=16384")
  foreach(v v79 v81)
    if(TARGET htp-${v} AND NOT NPH_NO_SKELS)
      add_dependencies(nph htp-${v})
    endif()
  endforeach()
endif()

# HV15 smoke bench (HunyuanVideo 1.5 FP8 on NPH)
if(ANDROID)
  add_library(hv15 SHARED hv15_jni.cpp)
  target_include_directories(hv15 PRIVATE "${SDCPP_DIR}/include")
  target_link_libraries(hv15 PRIVATE stable-diffusion log)
  target_compile_options(hv15 PRIVATE -fvisibility=hidden)
  target_link_options(hv15 PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/hv15.map" "-Wl,-s"
                      "-Wl,-z,max-page-size=16384")
  foreach(v v79 v81)
    if(TARGET htp-${v})
      add_dependencies(hv15 htp-${v})
    endif()
  endforeach()
endif()
