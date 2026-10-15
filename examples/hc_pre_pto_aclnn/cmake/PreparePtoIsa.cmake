# CMake 配置阶段：从原始 ISA 复制头文件，并只在构建目录的副本中应用 SK 补丁。
# 普通二进制编译和源码打包共用此副本；不修改共享 ISA 仓库。
# 重新配置时先从原始头文件覆盖副本，再应用补丁，避免重复打补丁。
find_package(Git REQUIRED)
set(HCPRE_ISA_ROOT "${CMAKE_BINARY_DIR}/dependencies/pto-isa")
file(GLOB_RECURSE isa_headers CONFIGURE_DEPENDS "${PTO_ISA_ROOT}/include/*")
foreach(header IN LISTS isa_headers)
  file(RELATIVE_PATH relative "${PTO_ISA_ROOT}" "${header}")
  configure_file("${header}" "${HCPRE_ISA_ROOT}/${relative}" COPYONLY)
endforeach()
# 使用确定的工作目录，避免 git apply 按上层仓库的子目录范围跳过补丁。
execute_process(
  COMMAND "${GIT_EXECUTABLE}" apply --unsafe-paths --directory "${HCPRE_ISA_ROOT}"
          "${CMAKE_CURRENT_SOURCE_DIR}/patches/pto_isa_sk.patch"
  WORKING_DIRECTORY / RESULT_VARIABLE patch_status ERROR_VARIABLE patch_error)
if(NOT patch_status EQUAL 0)
  message(FATAL_ERROR "PTO-ISA SK patch failed: ${patch_error}")
endif()
configure_file("${PTO_ISA_ROOT}/LICENSE" "${HCPRE_ISA_ROOT}/LICENSE" COPYONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             "${CMAKE_CURRENT_SOURCE_DIR}/patches/pto_isa_sk.patch")
