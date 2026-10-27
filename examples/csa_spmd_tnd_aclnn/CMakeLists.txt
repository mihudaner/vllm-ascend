cmake_minimum_required(VERSION 3.20)
project(csa_spmd_tnd_pto_aclnn LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CSA_VARIANT -1 CACHE STRING "-1 builds all variants; otherwise build only selected case")
add_compile_definitions(CSA_VARIANT=${CSA_VARIANT})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PTO_ISA_ROOT "" CACHE PATH "PTO-ISA checkout")
if(NOT EXISTS "${PTO_ISA_ROOT}/include/pto/pto-inst.hpp")
  message(FATAL_ERROR "Pass -DPTO_ISA_ROOT=/path/to/pto-isa")
endif()
set(ASCEND_COMPUTE_UNIT ascend910_93 CACHE STRING "Supported target")
set(vendor_name csa_spmd_tnd_pto CACHE STRING "Isolated OPP vendor")
find_package(ASC REQUIRED CONFIG PATHS "$ENV{ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
set(package_name ${vendor_name})
npu_op_package(${package_name} TYPE RUN CONFIG
  ENABLE_DEFAULT_PACKAGE_NAME_RULE False
  INSTALL_PATH ${CMAKE_BINARY_DIR}/)
add_subdirectory(op_host)
add_subdirectory(op_kernel)

# npugraph_ex recompiles the installed kernel sources for SuperKernel. Keep
# the same relative include layout as the source tree in the OPP package.
install(DIRECTORY include generated
  DESTINATION "packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl")
