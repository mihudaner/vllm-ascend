npu_op_kernel_options(ascendc_kernels ALL OPTIONS
  "-I${HCPRE_ISA_ROOT}/include" "-O3" "-DMEMORY_BASE"
  "-mllvm" "-cce-aicore-stack-size=0x8000"
  "-mllvm" "-cce-aicore-function-stack-size=0x8000"
  "-mllvm" "-cce-aicore-record-overflow=false")
npu_op_kernel_sources(ascendc_kernels KERNEL_DIR ./)
npu_op_kernel_library(ascendc_kernels SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/ TILING_LIBRARY cust_optiling)
npu_op_package_add(${package_name} LIBRARY ascendc_kernels)

# 将 SK 静态重编译需要的源码和头文件纳入安装规则，运行脚本无需再补文件。
set(hcpre_impl "packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl")
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/" DESTINATION "${hcpre_impl}/include")
install(DIRECTORY "${PROJECT_SOURCE_DIR}/generated/" DESTINATION "${hcpre_impl}/generated")
# ISA 使用 CANN Python 入口自带的 ../ascendc/common include 路径，随包迁移。
install(DIRECTORY "${HCPRE_ISA_ROOT}/include/" DESTINATION "${hcpre_impl}/ascendc/common")
install(FILES "${HCPRE_ISA_ROOT}/LICENSE" DESTINATION "${hcpre_impl}/ascendc/common/pto")
configure_file("${PROJECT_SOURCE_DIR}/cmake/PortableSourcePackage.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/PortableSourcePackage.cmake" @ONLY)
install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/PortableSourcePackage.cmake")
