cmake_minimum_required(VERSION 3.20)
project(dsf_csa_pto_bnn_aclnn LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PTO_ISA_ROOT "" CACHE PATH "PTO-ISA checkout")
if(NOT EXISTS "${PTO_ISA_ROOT}/include/pto/pto-inst.hpp")
  message(FATAL_ERROR "Pass -DPTO_ISA_ROOT=/path/to/pto-isa")
endif()
set(ASCEND_COMPUTE_UNIT ascend910_93 CACHE STRING "Supported target")
set(vendor_name dsf_csa_pto_bnn CACHE STRING "Isolated OPP vendor")
find_package(ASC REQUIRED CONFIG PATHS "$ENV{ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
set(package_name ${vendor_name})
npu_op_package(${package_name} TYPE RUN CONFIG
  ENABLE_DEFAULT_PACKAGE_NAME_RULE False
  INSTALL_PATH ${CMAKE_BINARY_DIR}/)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
# The original project is the only build entry. This stage contains only
# compiler inputs, never another standalone ACLNN project or host policy route.
execute_process(COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/prepare_device_sources.py"
  --output "${CMAKE_BINARY_DIR}/device_sources"
  RESULT_VARIABLE device_source_result)
if(NOT device_source_result EQUAL 0)
  message(FATAL_ERROR "Failed to prepare complete device-dispatch sources")
endif()
add_subdirectory(op_host)
add_subdirectory(op_kernel)

# npugraph_ex recompiles the installed kernel sources for SuperKernel. Keep
# the same relative include layout as the source tree in the OPP package.
install(DIRECTORY "${CMAKE_BINARY_DIR}/device_sources/include" "${CMAKE_BINARY_DIR}/device_sources/generated"
  DESTINATION "packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl")
