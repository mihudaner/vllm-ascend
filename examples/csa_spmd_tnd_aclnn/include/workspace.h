// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace csa {
constexpr uint64_t kAlignment=512;
constexpr uint64_t kMaxBytes=uint64_t{1}<<40;
inline uint64_t Align(uint64_t n){if(n>kMaxBytes)throw std::overflow_error("workspace too large");return (n+kAlignment-1)/kAlignment*kAlignment;}
struct Tensor {
 uint32_t shapes[5]{};
 int rank=0;
 int bytes=0;
 int source=-1;
 uint64_t offset=0;
 uint64_t Elements() const {uint64_t n=1;for(int i=0;i<rank;++i){if(shapes[i] && n>kMaxBytes/shapes[i])throw std::overflow_error("shape overflow");n*=shapes[i];}return n;}
 Tensor reshape(const uint32_t* dims,int ndim) const {
  if(ndim<1 || ndim>5)throw std::invalid_argument("rank");
  Tensor t=*this;t.rank=ndim;std::fill(t.shapes,t.shapes+5,0);
  for(int i=0;i<ndim;++i)t.shapes[i]=dims[i];
  if(t.Elements()!=Elements())throw std::invalid_argument("reshape changes element count");
  return t;
 }
 Tensor view(const uint32_t* dims,const uint32_t* start) const {
  if(rank!=1 || uint64_t(start[0])+dims[0]>shapes[0])throw std::invalid_argument("view outside storage");
  Tensor t=*this;t.shapes[0]=dims[0];t.offset+=uint64_t(start[0])*bytes;return t;
 }
};
inline Tensor Allocate(const uint32_t* shape,int rank,int bytes,uint64_t& cursor){
 Tensor t;t.rank=rank;t.bytes=bytes;for(int i=0;i<rank;++i)t.shapes[i]=shape[i];
 t.offset=Align(cursor);uint64_t n=t.Elements();if(n>kMaxBytes/bytes)throw std::overflow_error("tensor too large");
 cursor=t.offset+n*bytes;Align(cursor);return t;
}
struct Layout {uint64_t offsets[133]{};int32_t sources[133]{};int64_t scalars[28]{};uint64_t workspaceBytes=0;};
}
