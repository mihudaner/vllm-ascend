// SPDX-License-Identifier: Apache-2.0
// Direct ACLNN fixture runner. No Torch, vLLM, PyPTO or Simpler dependency.
#include <acl/acl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "invoke.h"
namespace fs = std::filesystem;
namespace {
constexpr size_t kGuard = 512;
void Check(int code, const char* stage) {
  if (code != 0) {
    const char* detail = aclGetRecentErrMsg();
    throw std::runtime_error(std::string(stage) + ": " + std::to_string(code) + " " + (detail ? detail : ""));
  }
}
std::vector<std::string> Split(const std::string& value, char separator) {
  std::vector<std::string> result;
  std::stringstream input(value);
  std::string part;
  while (std::getline(input, part, separator)) result.push_back(part);
  return result;
}
std::vector<int64_t> Shape(const std::string& value) {
  std::vector<int64_t> shape;
  for (const auto& part : Split(value, ',')) {
    const auto n = std::stoll(part);
    if (n <= 0) throw std::runtime_error("Nonpositive dimension");
    shape.push_back(n);
  }
  if (shape.empty() || shape.size() > 5) throw std::runtime_error("Invalid rank");
  return shape;
}
std::vector<uint8_t> Read(const fs::path& path, size_t expected) {
  if (fs::file_size(path) != expected) throw std::runtime_error("Size mismatch: " + path.string());
  std::vector<uint8_t> data(expected);
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(data.data()), expected)) throw std::runtime_error("Read failed");
  return data;
}
struct Tensor {
  std::string name, dtype;
  std::vector<int64_t> shape, storageShape, strides;
  std::vector<uint8_t> initial;
  size_t bytes = 0, elementBytes = 0;
  bool output = false;
  void* allocation = nullptr;
  aclTensor* handle = nullptr;
  ~Tensor() { if (handle) aclDestroyTensor(handle); if (allocation) aclrtFree(allocation); }
  void* Data() const { return static_cast<uint8_t*>(allocation) + kGuard; }
};
float Float(const uint8_t* data, const std::string& type) {
  if (type == "fp32") { float value; std::memcpy(&value, data, 4); return value; }
  uint16_t bits; std::memcpy(&bits, data, 2);
  uint32_t wide = uint32_t(bits) << 16;
  float value; std::memcpy(&value, &wide, 4); return value;
}
bool Compare(const Tensor& tensor, const std::vector<uint8_t>& actual, const fs::path& expectedPath) {
  const auto expected = Read(expectedPath, tensor.bytes);
  size_t bad = 0;
  double maxAbs = 0;
  const bool floating = tensor.dtype == "fp32" || tensor.dtype == "bfloat16";
  const double atol = tensor.name == "output" ? 0.01 : tensor.name == "topk_scores" ? 0.0001 : 0.00001;
  const double rtol = tensor.name == "output" ? 0.01 : 0.00001;
  for (size_t offset = 0; offset < tensor.bytes; offset += tensor.elementBytes) {
    if (floating) {
      const double a = Float(actual.data() + offset, tensor.dtype);
      const double b = Float(expected.data() + offset, tensor.dtype);
      const double delta = std::abs(a - b);
      maxAbs = std::max(maxAbs, delta);
      if (!std::isfinite(a) || !std::isfinite(b) || delta > atol + rtol * std::abs(b)) ++bad;
    } else if (std::memcmp(actual.data() + offset, expected.data() + offset, tensor.elementBytes)) ++bad;
  }
  std::cout << tensor.name << " elements=" << tensor.bytes / tensor.elementBytes
            << " mismatches=" << bad << " max_abs=" << maxAbs << '\n';
  return bad == 0;
}
}  // namespace
int main(int argc, char** argv) {
  if (argc != 6 && argc != 7) {
    std::cerr << "Usage: csa_aclnn_test INPUT_DIR OUTPUT_DIR EXPECTED_DIR_OR_DASH VARIANT DEVICE [WORKSPACE_DUMP]\n";
    return 2;
  }
  try {
    const fs::path inputDir(argv[1]), outputDir(argv[2]);
    const int64_t variant = std::stoll(argv[4]);
    const int device = std::stoi(argv[5]);
    fs::create_directories(outputDir);
    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    Check(aclrtSetOpExecuteTimeOut(30), "execution timeout");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "stream");
    // Fixed-size vector avoids moving owning Tensor objects.
    std::vector<Tensor> owned(57);
    std::vector<aclTensor*> tensors;
    std::ifstream manifest(inputDir / "tensors.tsv");
    if (!manifest) throw std::runtime_error("Missing tensors.tsv");
    std::string line;
    size_t index = 0;
    while (std::getline(manifest, line)) {
      if (line.empty() || line[0] == '#') continue;
      const auto fields = Split(line, '\t');
      if (fields.size() != 7 || index >= owned.size()) throw std::runtime_error("Invalid manifest");
      auto& tensor = owned[index++];
      tensor.name = fields[0]; tensor.dtype = fields[1];
      if (tensor.name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_") != std::string::npos)
        throw std::runtime_error("Invalid tensor name");
      tensor.shape = Shape(fields[3]); tensor.storageShape = Shape(fields[4]);
      tensor.output = fields[5] == "1";
      if (tensor.name != kTensorNames[index - 1] || tensor.output != kOutput[index - 1])
        throw std::runtime_error("Tensor ordering/direction mismatch");
      if (fields[2] != "NZ" && fields[2] != "ND") throw std::runtime_error("Invalid format");
      aclDataType dtype = ACL_DT_UNDEFINED;
      if (tensor.dtype == "fp32") { dtype = ACL_FLOAT; tensor.elementBytes = 4; }
      else if (tensor.dtype == "bfloat16") { dtype = ACL_BF16; tensor.elementBytes = 2; }
      else if (tensor.dtype == "int8") { dtype = ACL_INT8; tensor.elementBytes = 1; }
      else if (tensor.dtype == "int32") { dtype = ACL_INT32; tensor.elementBytes = 4; }
      else if (tensor.dtype == "int64") { dtype = ACL_INT64; tensor.elementBytes = 8; }
      else throw std::runtime_error("Unsupported dtype");
      tensor.bytes = tensor.elementBytes;
      for (auto dim : tensor.storageShape) {
        if (uint64_t(dim) > (uint64_t{1} << 40) / tensor.bytes) throw std::runtime_error("Size overflow");
        tensor.bytes *= dim;
      }
      if (std::stoull(fields[6]) != tensor.bytes) throw std::runtime_error("Invalid byte count");
      tensor.initial = Read(inputDir / (tensor.name + ".bin"), tensor.bytes);
      tensor.strides.assign(tensor.shape.size(), 1);
      for (int i = static_cast<int>(tensor.shape.size()) - 2; i >= 0; --i)
        tensor.strides[i] = tensor.strides[i + 1] * tensor.shape[i + 1];
      Check(aclrtMalloc(&tensor.allocation, tensor.bytes + 2 * kGuard, ACL_MEM_MALLOC_HUGE_FIRST), "tensor allocation");
      std::vector<uint8_t> guarded(tensor.bytes + 2 * kGuard, 0xa5);
      std::copy(tensor.initial.begin(), tensor.initial.end(), guarded.begin() + kGuard);
      Check(aclrtMemcpy(tensor.allocation, guarded.size(), guarded.data(), guarded.size(), ACL_MEMCPY_HOST_TO_DEVICE), "input copy");
      const auto format = fields[2] == "NZ" ? ACL_FORMAT_FRACTAL_NZ : ACL_FORMAT_ND;
      tensor.handle = aclCreateTensor(tensor.shape.data(), tensor.shape.size(), dtype,
          tensor.strides.data(), 0, format, tensor.storageShape.data(), tensor.storageShape.size(), tensor.Data());
      if (!tensor.handle) throw std::runtime_error("aclCreateTensor failed");
      tensors.push_back(tensor.handle);
    }
    if (index != 57) throw std::runtime_error("Expected 57 tensor records");
    uint64_t workspaceBytes = 0;
    aclOpExecutor* executor = nullptr;
    Check(Prepare(tensors, variant, &workspaceBytes, &executor), "GetWorkspaceSize");
    void* workspaceAllocation = nullptr;
    Check(aclrtMalloc(&workspaceAllocation, workspaceBytes + 2 * kGuard, ACL_MEM_MALLOC_HUGE_FIRST), "workspace");
    auto* workspace = static_cast<uint8_t*>(workspaceAllocation) + kGuard;
    Check(aclrtMemset(workspaceAllocation, workspaceBytes + 2 * kGuard, 0xa5, kGuard), "workspace pre-guard");
    Check(aclrtMemset(workspace + workspaceBytes, kGuard, 0xa5, kGuard), "workspace post-guard");
    Check(aclnnCsaSpmdTndPto(workspace, workspaceBytes, executor, stream), "aclnnCsaSpmdTndPto");
    Check(aclrtSynchronizeStream(stream), "synchronize");
    if (argc == 7) {
      std::vector<uint8_t> snapshot(workspaceBytes);
      Check(aclrtMemcpy(snapshot.data(), snapshot.size(), workspace, workspaceBytes,
                       ACL_MEMCPY_DEVICE_TO_HOST), "workspace snapshot");
      const fs::path dumpPath(argv[6]);
      if (fs::exists(dumpPath)) throw std::runtime_error("Workspace dump already exists");
      std::ofstream dump(dumpPath, std::ios::binary);
      dump.write(reinterpret_cast<const char*>(snapshot.data()), snapshot.size());
      if (!dump) throw std::runtime_error("Workspace snapshot write failed");
    }
    bool passed = true;
    for (const auto& tensor : owned) {
      std::vector<uint8_t> guarded(tensor.bytes + 2 * kGuard);
      Check(aclrtMemcpy(guarded.data(), guarded.size(), tensor.allocation, guarded.size(), ACL_MEMCPY_DEVICE_TO_HOST), "output copy");
      const auto unchanged = [](uint8_t value) { return value == 0xa5; };
      if (!std::all_of(guarded.begin(), guarded.begin() + kGuard, unchanged) ||
          !std::all_of(guarded.end() - kGuard, guarded.end(), unchanged)) {
        std::cerr << tensor.name << ": guard overwritten\n"; passed = false;
      }
      std::vector<uint8_t> actual(guarded.begin() + kGuard, guarded.end() - kGuard);
      if (!tensor.output) {
        if (actual != tensor.initial) { std::cerr << tensor.name << ": readonly input changed\n"; passed = false; }
        continue;
      }
      std::ofstream output(outputDir / (tensor.name + ".bin"), std::ios::binary);
      output.write(reinterpret_cast<const char*>(actual.data()), actual.size());
      if (!output) throw std::runtime_error("Output write failed");
      if (std::string(argv[3]) != "-") passed = Compare(tensor, actual, fs::path(argv[3]) / (tensor.name + ".bin")) && passed;
    }
    for (const auto* edge : {static_cast<uint8_t*>(workspaceAllocation), workspace + workspaceBytes}) {
      std::vector<uint8_t> guard(kGuard);
      Check(aclrtMemcpy(guard.data(), kGuard, edge, kGuard, ACL_MEMCPY_DEVICE_TO_HOST), "workspace guard copy");
      if (!std::all_of(guard.begin(), guard.end(), [](uint8_t value){return value == 0xa5;})) {
        std::cerr << "workspace guard overwritten\n"; passed = false;
      }
    }
    Check(aclrtFree(workspaceAllocation), "free workspace");
    Check(aclrtDestroyStream(stream), "destroy stream");
    std::cout << "workspace_bytes=" << workspaceBytes << " result=" << (passed ? "PASS" : "FAIL") << '\n';
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
