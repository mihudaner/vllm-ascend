#include "pto/pto-inst.hpp"
using namespace pto;

template <typename Tensor>
static AICORE inline auto PTOAS__GLOBAL_TENSOR_DATA(Tensor &tensor)
    -> decltype(tensor.data()) {
  return tensor.data();
}


enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

template <typename Ptr>
static AICORE inline void PTOAS__DCCI_SINGLE_CACHE_LINE(Ptr ptr) {
  dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
}

AICORE void hc_large_aic(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ bfloat16_t* v12, __gm__ float* v13, int64_t v14, int32_t v15, int64_t v16, int64_t v17, int64_t v18, int64_t v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int64_t v28, int64_t v29, int64_t v30, int32_t v31, int32_t v32) {
  using T = float;

  #if defined(__DAV_CUBE__)
  const int64_t v33 = 384;
  const int32_t v34 = 0;
  // pto: %c0_i64
  const int64_t v35 = 0;
  // pto: %c262144_i64
  const int64_t v36 = 262144;
  // pto: %c327680_i64
  const int64_t v37 = 327680;
  // pto: %c32768_i64
  const int64_t v38 = 32768;
  // pto: %c16384_i64
  const int64_t v39 = 16384;
  // pto: %c49152_i64
  const int64_t v40 = 49152;
  // pto: %c32_index
  const int64_t v41 = 32;
  // pto: %c1_index
  const int64_t v42 = 1;
  // pto: %c24_index
  const int64_t v43 = 24;
  // pto: %c16384_index
  const int64_t v44 = 16384;
  // pto: %c8_index
  const int64_t v45 = 8;
  // pto: %c64_index
  const int64_t v46 = 64;
  // pto: %c2048_index
  const int64_t v47 = 2048;
  // pto: %c0_index
  const int64_t v48 = 0;
  // pto: %c4_index
  const int64_t v49 = 4;
  // pto: %c2_index
  const int64_t v50 = 2;
  // pto: %c512_index
  const int64_t v51 = 512;
  // pto: %c256_index
  const int64_t v52 = 256;
  // pto: %c128_index
  const int64_t v53 = 128;
  // pto: %cn128_index
  const int64_t v54 = -128;
  // pto: %63
  int64_t v55 = (int64_t) ((uint64_t) v29 * (uint64_t) v45);
  // pto: %mixes_partials_inline1746__ssa_v0_view
  const int64_t v56 = 1;
  // pto: %mixes_partials_inline1746__ssa_v0_view
  const int64_t v57 = 1;
  // pto: %mixes_partials_inline1746__ssa_v0_view
  const int64_t v58 = 1;
  // pto: %mixes_partials_inline1746__ssa_v0_view
  int64_t v59 = v55 * v41;
  // pto: %mixes_partials_inline1746__ssa_v0_view
  int64_t v60 = v58 * v59;
  // pto: %mixes_partials_inline1746__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v61 = pto::Shape<1, 1, 1, -1, -1>(v56, v57, v58, v55, v41);
  // pto: %mixes_partials_inline1746__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v62 = pto::Stride<-1, -1, -1, -1, -1>(v57 * v60, v60, v59, v41, v42);
  // pto: %mixes_partials_inline1746__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v63 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v61, v62);
  // pto: %hc_fn__ssa_v0_view
  const int64_t v64 = 1;
  // pto: %hc_fn__ssa_v0_view
  const int64_t v65 = 1;
  // pto: %hc_fn__ssa_v0_view
  const int64_t v66 = 1;
  // pto: %hc_fn__ssa_v0_view
  int64_t v67 = v43 * v44;
  // pto: %hc_fn__ssa_v0_view
  int64_t v68 = v66 * v67;
  // pto: %hc_fn__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v69 = pto::Shape<1, 1, 1, -1, -1>(v64, v65, v66, v43, v44);
  // pto: %hc_fn__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v70 = pto::Stride<-1, -1, -1, -1, -1>(v65 * v68, v68, v67, v44, v42);
  // pto: %hc_fn__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v71 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v69, v70);
  // pto: %mid_stream_product_inline1658__phi_v5
  Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v72 = Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v46, v43);
  // pto: %mid_stream_product_inline1658__phi_v5
  uint64_t v73 = (uint64_t) v35;
  TASSIGN(v72, v73);
  const int32_t v74 = 0;
  auto v75 = TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>(v13, v74, v34);
  // pto: %worker_inline1745__ssa_v0
  // pto: %linear_workers_inline1691__ssa_v0_idx
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
  for (int64_t v76 = (int64_t) v31; v76 < v14; v76 += (int64_t) v15) {
    // pto: %20
    ;
    int64_t v77 = v76 % v45;
    // pto: %21
    ;
    int64_t v78 = (int64_t) ((uint64_t) v77 * (uint64_t) v47);
    // pto: %22, %23, %18, %19
    ;
    int64_t v79 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v77 * (uint64_t) v16) + (uint64_t) ((int64_t) (uint64_t) (v76 / v45) * (uint64_t) v46));
    // pto: %mid_stream_product_inline1658__tile
    ;
    Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v80 = Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v46, v41);
    // pto: %mid_stream_product_inline1658__tile
    ;
    uint64_t v81 = (uint64_t) v35;
    TASSIGN(v80, v81);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int64_t v82 = v48; v82 < v49; v82 += v50) {
      // pto: %24
      ;
      int64_t v83 = (int64_t) ((uint64_t) v82 * (uint64_t) v51);
      // pto: %25
      ;
      int64_t v84 = (int64_t) ((uint64_t) v78 + (uint64_t) v83);
      // pto: %28, %27
      ;
      int64_t v85 = (int64_t) ((uint64_t) v78 + (uint64_t) ((int64_t) (uint64_t) v83 + (uint64_t) v51));
      // pto: %mid_weight_chunk_inline1711__tile
      ;
      Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v86 = Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v43, v51);
      // pto: %mid_weight_chunk_inline1711__tile
      ;
      uint64_t v87 = (uint64_t) v36;
      TASSIGN(v86, v87);
      // pto: %29
      ;
      int64_t v88 = v84 < v48 ? v48 : v84;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v89 = 0;
      // pto: %hc_fn__ssa_v0_pview
      ;
      __gm__ float* v90 = PTOAS__GLOBAL_TENSOR_DATA(v71);
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v91 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v92 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v93 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      int64_t v94 = v43 * v44;
      // pto: %hc_fn__ssa_v0_pview
      ;
      int64_t v95 = v93 * v94;
      // pto: %hc_fn__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 24, -1> v96 = pto::Shape<1, 1, 1, 24, -1>(v91, v92, v93, v43, v51);
      // pto: %hc_fn__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v97 = pto::Stride<-1, -1, -1, -1, -1>(v92 * v95, v95, v94, v44, v42);
      // pto: %hc_fn__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v98 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v90 + (v89 + v48 * v44 + v88 * v42), v96, v97);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v86, v98);
      // pto: %0
      ;
      Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v99 = Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v43, v51);
      // pto: %0
      ;
      uint64_t v100 = (uint64_t) v37;
      TASSIGN(v99, v100);
      // pto: %30
      ;
      int64_t v101 = v85 < v48 ? v48 : v85;
      // pto: %31
      ;
      const int64_t v102 = 0;
      // pto: %31
      ;
      __gm__ float* v103 = PTOAS__GLOBAL_TENSOR_DATA(v71);
      // pto: %31
      ;
      const int64_t v104 = 1;
      // pto: %31
      ;
      const int64_t v105 = 1;
      // pto: %31
      ;
      const int64_t v106 = 1;
      // pto: %31
      ;
      int64_t v107 = v43 * v44;
      // pto: %31
      ;
      int64_t v108 = v106 * v107;
      // pto: %31
      ;
      pto::Shape<1, 1, 1, 24, -1> v109 = pto::Shape<1, 1, 1, 24, -1>(v104, v105, v106, v43, v51);
      // pto: %31
      ;
      pto::Stride<-1, -1, -1, -1, -1> v110 = pto::Stride<-1, -1, -1, -1, -1>(v105 * v108, v108, v107, v44, v42);
      // pto: %31
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v111 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v103 + (v102 + v48 * v44 + v101 * v42), v109, v110);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TLOAD(v99, v111);
      // pto: %mid_gathered_chunk_inline1633__tile
      ;
      const int32_t v112 = 64;
      // pto: %mid_gathered_chunk_inline1633__tile
      ;
      const int32_t v113 = 512;
      // pto: %mid_gathered_chunk_inline1633__tile
      ;
      Tile<TileType::Mat, float, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v114 = Tile<TileType::Mat, float, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v112, v113);
      pipe_barrier(PIPE_MTE2);
      TPOP<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, Tile<TileType::Mat, float, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v75, v114);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %32
      ;
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      if (v82 == v48) {
        // pto: %mid_weight_chunk_inline1711__tile_t
        ;
        Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v115 = Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v51, v43);
        // pto: %mid_weight_chunk_inline1711__tile_t
        ;
        uint64_t v116 = (uint64_t) v36;
        TASSIGN(v115, v116);
        // pto: %mid_stream_product_inline1658__tile_l0_init_storage
        ;
        Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v117 = Tile<TileType::Acc, float, 64, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v46, v41);
        // pto: %mid_stream_product_inline1658__tile_l0_init_storage
        ;
        uint64_t v118 = (uint64_t) v35;
        TASSIGN(v117, v118);
        v117.SetValidShape(v46, v43);
        for (int64_t v119 = v48; v119 < v51; v119 += v52) {
          // pto: %mid_stream_product_inline1658__tile_l0_a
          ;
          Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v120 = Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v46, v53);
          // pto: %mid_stream_product_inline1658__tile_l0_a
          ;
          uint64_t v121 = (uint64_t) v35;
          TASSIGN(v120, v121);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          pipe_barrier(PIPE_MTE1);
          uint16_t v122 = static_cast<uint16_t>(v48);
          uint16_t v123 = static_cast<uint16_t>(v119);
          TEXTRACT(v120, v114, v122, v123);
          // pto: %33
          ;
          int64_t v124 = (int64_t) ((uint64_t) v51 - (uint64_t) v119);
          // pto: %34
          ;
          int64_t v125 = v124 < v53 ? v124 : v53;
          // pto: %mid_stream_product_inline1658__tile_l0_b
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v126 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v125, v43);
          // pto: %mid_stream_product_inline1658__tile_l0_b
          ;
          uint64_t v127 = (uint64_t) v35;
          TASSIGN(v126, v127);
          uint16_t v128 = static_cast<uint16_t>(v119);
          uint16_t v129 = static_cast<uint16_t>(v48);
          TEXTRACT(v126, v115, v128, v129);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          // pto: %1
          ;
          Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v130 = Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v46, v53);
          // pto: %1
          ;
          uint64_t v131 = (uint64_t) v38;
          TASSIGN(v130, v131);
          // pto: %35
          ;
          int64_t v132 = (int64_t) ((uint64_t) v119 + (uint64_t) v53);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          uint16_t v133 = static_cast<uint16_t>(v48);
          uint16_t v134 = static_cast<uint16_t>(v132);
          TEXTRACT(v130, v114, v133, v134);
          int64_t v135 = (int64_t) ((uint64_t) v33 - (uint64_t) v119);
          // pto: %38
          ;
          int64_t v136 = v135 < v48 ? v48 : v135;
          // pto: %39
          ;
          int64_t v137 = v136 < v53 ? v136 : v53;
          // pto: %2
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v138 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v137, v43);
          // pto: %2
          ;
          uint64_t v139 = (uint64_t) v39;
          TASSIGN(v138, v139);
          uint16_t v140 = static_cast<uint16_t>(v132);
          uint16_t v141 = static_cast<uint16_t>(v48);
          TEXTRACT(v138, v115, v140, v141);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          // pto: %41
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          if (v119 == v48) {
            pipe_barrier(PIPE_M);
            TMATMUL(v117, v120, v126);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v117, v117, v120, v126);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          // pto: %42
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          if (v119 == v54) {
            pipe_barrier(PIPE_M);
            TMATMUL(v117, v130, v138);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v117, v117, v130, v138);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        };
      } else {
        // pto: %4
        ;
        Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v142 = Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v51, v43);
        // pto: %4
        ;
        uint64_t v143 = (uint64_t) v36;
        TASSIGN(v142, v143);
        for (int64_t v144 = v48; v144 < v51; v144 += v52) {
          // pto: %5
          ;
          Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v145 = Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v46, v53);
          // pto: %5
          ;
          uint64_t v146 = (uint64_t) v35;
          TASSIGN(v145, v146);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
          uint16_t v147 = static_cast<uint16_t>(v48);
          uint16_t v148 = static_cast<uint16_t>(v144);
          TEXTRACT(v145, v114, v147, v148);
          // pto: %44
          ;
          int64_t v149 = (int64_t) ((uint64_t) v51 - (uint64_t) v144);
          // pto: %45
          ;
          int64_t v150 = v149 < v53 ? v149 : v53;
          // pto: %6
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v151 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v150, v43);
          // pto: %6
          ;
          uint64_t v152 = (uint64_t) v35;
          TASSIGN(v151, v152);
          uint16_t v153 = static_cast<uint16_t>(v144);
          uint16_t v154 = static_cast<uint16_t>(v48);
          TEXTRACT(v151, v142, v153, v154);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
          // pto: %7
          ;
          Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v155 = Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v46, v53);
          // pto: %7
          ;
          uint64_t v156 = (uint64_t) v38;
          TASSIGN(v155, v156);
          // pto: %46
          ;
          int64_t v157 = (int64_t) ((uint64_t) v144 + (uint64_t) v53);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
          uint16_t v158 = static_cast<uint16_t>(v48);
          uint16_t v159 = static_cast<uint16_t>(v157);
          TEXTRACT(v155, v114, v158, v159);
          int64_t v160 = (int64_t) ((uint64_t) v33 - (uint64_t) v144);
          // pto: %49
          ;
          int64_t v161 = v160 < v48 ? v48 : v160;
          // pto: %50
          ;
          int64_t v162 = v161 < v53 ? v161 : v53;
          // pto: %8
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v163 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v162, v43);
          // pto: %8
          ;
          uint64_t v164 = (uint64_t) v39;
          TASSIGN(v163, v164);
          uint16_t v165 = static_cast<uint16_t>(v157);
          uint16_t v166 = static_cast<uint16_t>(v48);
          TEXTRACT(v163, v142, v165, v166);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
          pipe_barrier(PIPE_M);
          TMATMUL_ACC(v80, v80, v145, v151);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
          pipe_barrier(PIPE_M);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
          TMATMUL_ACC(v80, v80, v155, v163);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        };
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TFREE<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v75);
      // pto: %52
      ;
      const int32_t v167 = 64;
      // pto: %52
      ;
      const int32_t v168 = 512;
      // pto: %52
      ;
      Tile<TileType::Mat, float, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v169 = Tile<TileType::Mat, float, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v167, v168);
      TPOP<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, Tile<TileType::Mat, float, 64, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v75, v169);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      // pto: %11
      ;
      Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v170 = Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v51, v43);
      // pto: %11
      ;
      uint64_t v171 = (uint64_t) v37;
      TASSIGN(v170, v171);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
      for (int64_t v172 = v48; v172 < v51; v172 += v52) {
        // pto: %12
        ;
        Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v173 = Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v46, v53);
        // pto: %12
        ;
        uint64_t v174 = (uint64_t) v35;
        TASSIGN(v173, v174);
        pipe_barrier(PIPE_MTE1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        uint16_t v175 = static_cast<uint16_t>(v48);
        uint16_t v176 = static_cast<uint16_t>(v172);
        TEXTRACT(v173, v169, v175, v176);
        // pto: %54
        ;
        int64_t v177 = (int64_t) ((uint64_t) v51 - (uint64_t) v172);
        // pto: %55
        ;
        int64_t v178 = v177 < v53 ? v177 : v53;
        // pto: %13
        ;
        Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v179 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v178, v43);
        // pto: %13
        ;
        uint64_t v180 = (uint64_t) v38;
        TASSIGN(v179, v180);
        uint16_t v181 = static_cast<uint16_t>(v172);
        uint16_t v182 = static_cast<uint16_t>(v48);
        TEXTRACT(v179, v170, v181, v182);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
        // pto: %14
        ;
        Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v183 = Tile<TileType::Left, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v46, v53);
        // pto: %14
        ;
        uint64_t v184 = (uint64_t) v38;
        TASSIGN(v183, v184);
        // pto: %56
        ;
        int64_t v185 = (int64_t) ((uint64_t) v172 + (uint64_t) v53);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
        uint16_t v186 = static_cast<uint16_t>(v48);
        uint16_t v187 = static_cast<uint16_t>(v185);
        TEXTRACT(v183, v169, v186, v187);
        int64_t v188 = (int64_t) ((uint64_t) v33 - (uint64_t) v172);
        // pto: %59
        ;
        int64_t v189 = v188 < v48 ? v48 : v188;
        // pto: %60
        ;
        int64_t v190 = v189 < v53 ? v189 : v53;
        // pto: %15
        ;
        Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v191 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v190, v43);
        // pto: %15
        ;
        uint64_t v192 = (uint64_t) v40;
        TASSIGN(v191, v192);
        uint16_t v193 = static_cast<uint16_t>(v185);
        uint16_t v194 = static_cast<uint16_t>(v48);
        TEXTRACT(v191, v170, v193, v194);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v72, v72, v173, v179);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
        TMATMUL_ACC(v72, v72, v183, v191);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TFREE<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v75);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    // pto: %62
    ;
    int64_t v195 = v79 < v48 ? v48 : v79;
    // pto: %mixes_partials_inline1746__iter_v1_pview
    ;
    __gm__ float* v196 = PTOAS__GLOBAL_TENSOR_DATA(v63);
    // pto: %mixes_partials_inline1746__iter_v1_pview
    ;
    const int64_t v197 = 0;
    // pto: %mixes_partials_inline1746__iter_v1_pview
    ;
    const int64_t v198 = 32;
    // pto: %mixes_partials_inline1746__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 64, 32> v199 = pto::Shape<1, 1, 1, 64, 32>();
    // pto: %mixes_partials_inline1746__iter_v1_pview
    ;
    pto::Stride<2048, 2048, 2048, 32, 1> v200 = pto::Stride<2048, 2048, 2048, 32, 1>();
    // pto: %mixes_partials_inline1746__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 64, 32>, pto::Stride<2048, 2048, 2048, 32, 1>, pto::Layout::ND> v201 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 32>, pto::Stride<2048, 2048, 2048, 32, 1>, pto::Layout::ND>(v196 + (v197 + v195 * v198), v199, v200);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v201, v80);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  pipe_barrier(PIPE_MTE2);
  pipe_barrier(PIPE_MTE3);
  pipe_barrier(PIPE_FIX);
  dsb(DSB_DDR);
  SYNCALL<SyncCoreType::Mix>();
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

AICORE void hc_large_aiv(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ bfloat16_t* v12, __gm__ float* v13, int64_t v14, int32_t v15, int64_t v16, int64_t v17, int64_t v18, int64_t v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int64_t v28, int64_t v29, int64_t v30, int32_t v31, int32_t v32, int32_t v33) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v34 = 0;
  // pto: %c98432_i64
  const int64_t v35 = 98432;
  // pto: %c128_i64
  const int64_t v36 = 128;
  // pto: %c32896_i64
  const int64_t v37 = 32896;
  // pto: %c163968_i64
  const int64_t v38 = 163968;
  // pto: %c33920_i64
  const int64_t v39 = 33920;
  // pto: %c34176_i64
  const int64_t v40 = 34176;
  // pto: %c33984_i64
  const int64_t v41 = 33984;
  const int64_t v42 = 32928;
  // pto: %c33408_i64
  const int64_t v43 = 33408;
  // pto: %c33664_i64
  const int64_t v44 = 33664;
  // pto: %c34688_i64
  const int64_t v45 = 34688;
  // pto: %c35200_i64
  const int64_t v46 = 35200;
  // pto: %c35712_i64
  const int64_t v47 = 35712;
  // pto: %c36224_i64
  const int64_t v48 = 36224;
  // pto: %c36736_i64
  const int64_t v49 = 36736;
  // pto: %c37248_i64
  const int64_t v50 = 37248;
  // pto: %c34720_i64
  const int64_t v51 = 34720;
  // pto: %c34752_i64
  const int64_t v52 = 34752;
  // pto: %c34784_i64
  const int64_t v53 = 34784;
  // pto: %c33952_i64
  const int64_t v54 = 33952;
  // pto: %c33152_i64
  const int64_t v55 = 33152;
  // pto: %c147584_i64
  const int64_t v56 = 147584;
  // pto: %c33024_i64
  const int64_t v57 = 33024;
  // pto: %c49280_i64
  const int64_t v58 = 49280;
  // pto: %c65664_i64
  const int64_t v59 = 65664;
  // pto: %c73856_i64
  const int64_t v60 = 73856;
  // pto: %c131200_i64
  const int64_t v61 = 131200;
  // pto: %c114816_i64
  const int64_t v62 = 114816;
  // pto: %c123008_i64
  const int64_t v63 = 123008;
  // pto: %c139392_i64
  const int64_t v64 = 139392;
  // pto: %c82048_i64
  const int64_t v65 = 82048;
  // pto: %c32_index
  const int64_t v66 = 32;
  // pto: %c1_index
  const int64_t v67 = 1;
  // pto: %c24_index
  const int64_t v68 = 24;
  // pto: %c16384_index
  const int64_t v69 = 16384;
  // pto: %c3_index
  const int64_t v70 = 3;
  // pto: %c8_index
  const int64_t v71 = 8;
  const int64_t v72 = 16;
  // pto: %c4096_index
  const int64_t v73 = 4096;
  // pto: %c0_i32
  const int32_t v74 = 0;
  // pto: %c64_index
  const int64_t v75 = 64;
  // pto: %c2048_index
  const int64_t v76 = 2048;
  // pto: %cst_43
  const float v77 = 0.0f;
  // pto: %c0_index
  const int64_t v78 = 0;
  // pto: %c4_index
  const int64_t v79 = 4;
  // pto: %c2_index
  const int64_t v80 = 2;
  // pto: %c512_index
  const int64_t v81 = 512;
  // pto: %c256_index
  const int64_t v82 = 256;
  // pto: %cst_49
  const float v83 = 6.10351563E-5f;
  // pto: %cst_50
  const float v84 = 9.99999997E-7f;
  // pto: %c128_index
  const int64_t v85 = 128;
  // pto: %c192_index
  const int64_t v86 = 192;
  // pto: %cst_53
  const float v87 = 1.0f;
  // pto: %cst_54
  const float v88 = 0.5f;
  // pto: %cst_55
  const float v89 = 2.0f;
  // pto: %cst_56
  const float v90 = 0.25f;
  // pto: %cst_57
  const float v91 = 4.0f;
  // pto: %cst_58
  const float v92 = 0.125f;
  // pto: %cst_59
  const float v93 = 8.0f;
  // pto: %cst_60
  const float v94 = 0.0625f;
  // pto: %cst_61
  const float v95 = 16.0f;
  // pto: %c19_index
  const int64_t v96 = 19;
  // pto: %234
  int64_t v97 = (int64_t) ((uint64_t) v29 * (uint64_t) v71);
  // pto: %rms_partials_inline1725__ssa_v0_view
  const int64_t v98 = 1;
  // pto: %rms_partials_inline1725__ssa_v0_view
  const int64_t v99 = 1;
  // pto: %rms_partials_inline1725__ssa_v0_view
  const int64_t v100 = 1;
  // pto: %rms_partials_inline1725__ssa_v0_view
  int64_t v101 = v97 * v67;
  // pto: %rms_partials_inline1725__ssa_v0_view
  int64_t v102 = v100 * v101;
  // pto: %rms_partials_inline1725__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v103 = pto::Shape<1, 1, 1, -1, -1>(v98, v99, v100, v97, v67);
  // pto: %rms_partials_inline1725__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v104 = pto::Stride<-1, -1, -1, -1, -1>(v99 * v102, v102, v101, v67, v97);
  // pto: %rms_partials_inline1725__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v105 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v103, v104);
  // pto: %x_flat_inline1727__ssa_v0_view
  const int64_t v106 = 1;
  // pto: %x_flat_inline1727__ssa_v0_view
  const int64_t v107 = 1;
  // pto: %x_flat_inline1727__ssa_v0_view
  const int64_t v108 = 1;
  // pto: %x_flat_inline1727__ssa_v0_view
  int64_t v109 = (int64_t) v30;
  // pto: %x_flat_inline1727__ssa_v0_view
  int64_t v110 = v109 * v69;
  // pto: %x_flat_inline1727__ssa_v0_view
  int64_t v111 = v108 * v110;
  // pto: %x_flat_inline1727__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v112 = pto::Shape<1, 1, 1, -1, -1>(v106, v107, v108, v109, v69);
  // pto: %x_flat_inline1727__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v113 = pto::Stride<-1, -1, -1, -1, -1>(v107 * v111, v111, v110, v69, v67);
  // pto: %x_flat_inline1727__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v114 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v112, v113);
  // pto: %235
  int64_t v115 = (int64_t) ((uint64_t) v29 * (uint64_t) v66);
  // pto: %parts_2d_inline1634__ssa_v0_view
  const int64_t v116 = 1;
  // pto: %parts_2d_inline1634__ssa_v0_view
  const int64_t v117 = 1;
  // pto: %parts_2d_inline1634__ssa_v0_view
  const int64_t v118 = 1;
  // pto: %parts_2d_inline1634__ssa_v0_view
  int64_t v119 = v71 * v115;
  // pto: %parts_2d_inline1634__ssa_v0_view
  int64_t v120 = v118 * v119;
  // pto: %parts_2d_inline1634__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v121 = pto::Shape<1, 1, 1, -1, -1>(v116, v117, v118, v71, v115);
  // pto: %parts_2d_inline1634__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v122 = pto::Stride<-1, -1, -1, -1, -1>(v117 * v120, v120, v119, v115, v67);
  // pto: %parts_2d_inline1634__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v123 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v121, v122);
  // pto: %rms_2d_inline1721__ssa_v0_view
  const int64_t v124 = 1;
  // pto: %rms_2d_inline1721__ssa_v0_view
  const int64_t v125 = 1;
  // pto: %rms_2d_inline1721__ssa_v0_view
  const int64_t v126 = 1;
  // pto: %rms_2d_inline1721__ssa_v0_view
  int64_t v127 = (int64_t) v29;
  // pto: %rms_2d_inline1721__ssa_v0_view
  int64_t v128 = v71 * v127;
  // pto: %rms_2d_inline1721__ssa_v0_view
  int64_t v129 = v126 * v128;
  // pto: %rms_2d_inline1721__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v130 = pto::Shape<1, 1, 1, -1, -1>(v124, v125, v126, v71, (int64_t) v29);
  // pto: %rms_2d_inline1721__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v131 = pto::Stride<-1, -1, -1, -1, -1>(v125 * v129, v129, v128, v127, v67);
  // pto: %rms_2d_inline1721__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v132 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v130, v131);
  // pto: %hc_base_2d_inline1856__ssa_v0_view
  const int64_t v133 = 1;
  // pto: %hc_base_2d_inline1856__ssa_v0_view
  const int64_t v134 = 1;
  // pto: %hc_base_2d_inline1856__ssa_v0_view
  const int64_t v135 = 1;
  // pto: %hc_base_2d_inline1856__ssa_v0_view
  int64_t v136 = v67 * v68;
  // pto: %hc_base_2d_inline1856__ssa_v0_view
  int64_t v137 = v135 * v136;
  // pto: %hc_base_2d_inline1856__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v138 = pto::Shape<1, 1, 1, -1, -1>(v133, v134, v135, v67, v68);
  // pto: %hc_base_2d_inline1856__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v139 = pto::Stride<-1, -1, -1, -1, -1>(v134 * v137, v137, v136, v68, v67);
  // pto: %hc_base_2d_inline1856__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v140 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v8, v138, v139);
  // pto: %comb_inline1693__ssa_v0_view
  const int64_t v141 = 1;
  // pto: %comb_inline1693__ssa_v0_view
  const int64_t v142 = 1;
  // pto: %comb_inline1693__ssa_v0_view
  const int64_t v143 = 1;
  // pto: %comb_inline1693__ssa_v0_view
  int64_t v144 = (int64_t) v30;
  // pto: %comb_inline1693__ssa_v0_view
  int64_t v145 = v144 * v72;
  // pto: %comb_inline1693__ssa_v0_view
  int64_t v146 = v143 * v145;
  // pto: %comb_inline1693__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v147 = pto::Shape<1, 1, 1, -1, -1>(v141, v142, v143, v144, v72);
  // pto: %comb_inline1693__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v148 = pto::Stride<-1, -1, -1, -1, -1>(v142 * v146, v146, v145, v72, v67);
  // pto: %comb_inline1693__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v149 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v9, v147, v148);
  // pto: %hc_scale_2d_inline1757__ssa_v0_view
  const int64_t v150 = 1;
  // pto: %hc_scale_2d_inline1757__ssa_v0_view
  const int64_t v151 = 1;
  // pto: %hc_scale_2d_inline1757__ssa_v0_view
  const int64_t v152 = 1;
  // pto: %hc_scale_2d_inline1757__ssa_v0_view
  int64_t v153 = v67 * v70;
  // pto: %hc_scale_2d_inline1757__ssa_v0_view
  int64_t v154 = v152 * v153;
  // pto: %hc_scale_2d_inline1757__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v155 = pto::Shape<1, 1, 1, -1, -1>(v150, v151, v152, v67, v70);
  // pto: %hc_scale_2d_inline1757__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v156 = pto::Stride<-1, -1, -1, -1, -1>(v151 * v154, v154, v153, v70, v67);
  // pto: %hc_scale_2d_inline1757__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v157 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v10, v155, v156);
  // pto: %236
  int64_t v158 = (int64_t) ((uint64_t) v30 * (uint64_t) v79);
  // pto: %post_flat_inline1720__ssa_v0_view
  const int64_t v159 = 1;
  // pto: %post_flat_inline1720__ssa_v0_view
  const int64_t v160 = 1;
  // pto: %post_flat_inline1720__ssa_v0_view
  const int64_t v161 = 1;
  // pto: %post_flat_inline1720__ssa_v0_view
  int64_t v162 = v67 * v158;
  // pto: %post_flat_inline1720__ssa_v0_view
  int64_t v163 = v161 * v162;
  // pto: %post_flat_inline1720__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v164 = pto::Shape<1, 1, 1, -1, -1>(v159, v160, v161, v67, v158);
  // pto: %post_flat_inline1720__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v165 = pto::Stride<-1, -1, -1, -1, -1>(v160 * v163, v163, v162, v158, v67);
  // pto: %post_flat_inline1720__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v166 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v11, v164, v165);
  // pto: %mixed_view_inline1624__ssa_v0_view
  const int64_t v167 = 1;
  // pto: %mixed_view_inline1624__ssa_v0_view
  const int64_t v168 = 1;
  // pto: %mixed_view_inline1624__ssa_v0_view
  const int64_t v169 = 1;
  // pto: %mixed_view_inline1624__ssa_v0_view
  int64_t v170 = (int64_t) v30;
  // pto: %mixed_view_inline1624__ssa_v0_view
  int64_t v171 = v170 * v73;
  // pto: %mixed_view_inline1624__ssa_v0_view
  int64_t v172 = v169 * v171;
  // pto: %mixed_view_inline1624__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v173 = pto::Shape<1, 1, 1, -1, -1>(v167, v168, v169, v170, v73);
  // pto: %mixed_view_inline1624__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v174 = pto::Stride<-1, -1, -1, -1, -1>(v168 * v172, v172, v171, v73, v67);
  // pto: %mixed_view_inline1624__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v175 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v12, v173, v174);
  const int32_t v176 = 0;
  auto v177 = TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>(v13, v176, v74);
  // pto: %worker_inline1745__ssa_v0
  int64_t v178 = (int64_t) v31;
  // pto: %linear_workers_inline1691__ssa_v0_idx
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  set_flag(PIPE_S, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
  for (int64_t v179 = v178; v179 < v14; v179 += (int64_t) v15) {
    // pto: %46, %47
    ;
    int64_t v180 = (int64_t) ((uint64_t) (v179 / v71) * (uint64_t) v75);
    // pto: %48
    ;
    int64_t v181 = v179 % v71;
    // pto: %49
    ;
    int64_t v182 = (int64_t) ((uint64_t) v181 * (uint64_t) v76);
    // pto: %mid_stream_rms_total_inline1607__tile
    ;
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v183 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v66);
    // pto: %mid_stream_rms_total_inline1607__tile
    ;
    uint64_t v184 = (uint64_t) v34;
    TASSIGN(v183, v184);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(v183, v77);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    for (int64_t v185 = v78; v185 < v79; v185 += v80) {
      // pto: %52
      ;
      int64_t v186 = (int64_t) ((uint64_t) v185 * (uint64_t) v81);
      // pto: %53
      ;
      int64_t v187 = (int64_t) ((uint64_t) v182 + (uint64_t) v186);
      // pto: %55, %lane_inline1748__ssa_v0, %54
      ;
      int64_t v188 = (int64_t) ((uint64_t) v180 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) * (uint64_t) v66));
      // pto: %56
      ;
      int64_t v189 = (int64_t) ((uint64_t) v17 - (uint64_t) v188);
      // pto: %57
      ;
      int64_t v190 = v189 < v66 ? v189 : v66;
      // pto: %58
      ;
      int64_t v191 = v190 < v78 ? v78 : v190;
      // pto: %61, %60
      ;
      int64_t v192 = (int64_t) ((uint64_t) v182 + (uint64_t) ((int64_t) (uint64_t) v186 + (uint64_t) v81));
      // pto: %mid_stream_source_inline1641__tile
      ;
      Tile<TileType::Vec, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v193 = Tile<TileType::Vec, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v191, v81);
      // pto: %mid_stream_source_inline1641__tile
      ;
      uint64_t v194 = (uint64_t) v35;
      TASSIGN(v193, v194);
      // pto: %68
      ;
      int64_t v195 = v188 < v78 ? v78 : v188;
      // pto: %69
      ;
      int64_t v196 = v187 < v78 ? v78 : v187;
      // pto: %x_flat_inline1727__ssa_v0_pview
      ;
      const int64_t v197 = 0;
      // pto: %x_flat_inline1727__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v198 = PTOAS__GLOBAL_TENSOR_DATA(v114);
      // pto: %x_flat_inline1727__ssa_v0_pview
      ;
      const int64_t v199 = 1;
      // pto: %x_flat_inline1727__ssa_v0_pview
      ;
      const int64_t v200 = 1;
      // pto: %x_flat_inline1727__ssa_v0_pview
      ;
      const int64_t v201 = 1;
      // pto: %x_flat_inline1727__ssa_v0_pview
      ;
      int64_t v202 = v191 * v69;
      // pto: %x_flat_inline1727__ssa_v0_pview
      ;
      int64_t v203 = v201 * v202;
      // pto: %x_flat_inline1727__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, -1> v204 = pto::Shape<1, 1, 1, -1, -1>(v199, v200, v201, v191, v81);
      // pto: %x_flat_inline1727__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v205 = pto::Stride<-1, -1, -1, -1, -1>(v200 * v203, v203, v202, v69, v67);
      // pto: %x_flat_inline1727__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v206 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v198 + (v197 + v195 * v69 + v196 * v67), v204, v205);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v193, v206);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %0
      ;
      Tile<TileType::Vec, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v207 = Tile<TileType::Vec, bfloat16_t, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v191, v81);
      // pto: %0
      ;
      uint64_t v208 = (uint64_t) v36;
      TASSIGN(v207, v208);
      // pto: %71
      ;
      int64_t v209 = v192 < v78 ? v78 : v192;
      // pto: %72
      ;
      const int64_t v210 = 0;
      // pto: %72
      ;
      __gm__ bfloat16_t* v211 = PTOAS__GLOBAL_TENSOR_DATA(v114);
      // pto: %72
      ;
      const int64_t v212 = 1;
      // pto: %72
      ;
      const int64_t v213 = 1;
      // pto: %72
      ;
      const int64_t v214 = 1;
      // pto: %72
      ;
      int64_t v215 = v191 * v69;
      // pto: %72
      ;
      int64_t v216 = v214 * v215;
      // pto: %72
      ;
      pto::Shape<1, 1, 1, -1, -1> v217 = pto::Shape<1, 1, 1, -1, -1>(v212, v213, v214, v191, v81);
      // pto: %72
      ;
      pto::Stride<-1, -1, -1, -1, -1> v218 = pto::Stride<-1, -1, -1, -1, -1>(v213 * v216, v216, v215, v69, v67);
      // pto: %72
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v219 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v211 + (v210 + v195 * v69 + v209 * v67), v217, v218);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      TLOAD(v207, v219);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %mid_stream_widened_inline1690__tile
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v220 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v191, v81);
      // pto: %mid_stream_widened_inline1690__tile
      ;
      uint64_t v221 = (uint64_t) v37;
      TASSIGN(v220, v221);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      RoundMode v222 = RoundMode::CAST_ROUND;
      SaturationMode v223 = SaturationMode::ON;
      TCVT(v220, v193, v222, v223);
      v220.SetValidShape(v191, v81);
      // pto: %mid_stream_clean_inline1640__tile
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v224 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v66, v81);
      // pto: %mid_stream_clean_inline1640__tile
      ;
      uint64_t v225 = (uint64_t) v37;
      TASSIGN(v224, v225);
      pipe_barrier(PIPE_V);
      TFILLPAD<pto::TFillPadMode::InPlace>(v224, v220);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      v224.SetValidShape(v66, v81);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TPUSH<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v177, v224);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      // pto: %1
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v226 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v66, v81);
      // pto: %1
      ;
      uint64_t v227 = (uint64_t) v37;
      TASSIGN(v226, v227);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      TMUL(v226, v224, v224);
      // pto: %tmp_tile
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v228 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v81);
      // pto: %tmp_tile
      ;
      uint64_t v229 = (uint64_t) v35;
      TASSIGN(v228, v229);
      // pto: %2
      ;
      Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v230 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v67);
      // pto: %2
      ;
      uint64_t v231 = (uint64_t) v38;
      TASSIGN(v230, v231);
      pipe_barrier(PIPE_V);
      TROWSUM(v230, v226, v228);
      // pto: %mid_stream_sq_inline1734__tile
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v232 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v66);
      // pto: %mid_stream_sq_inline1734__tile
      ;
      uint64_t v233 = (uint64_t) v38;
      TASSIGN(v232, v233);
      // pto: %3
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v234 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v66);
      // pto: %3
      ;
      uint64_t v235 = (uint64_t) v38;
      TASSIGN(v234, v235);
      pipe_barrier(PIPE_V);
      TADD(v234, v183, v232);
      // pto: %4
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v236 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v191, v81);
      // pto: %4
      ;
      uint64_t v237 = (uint64_t) v37;
      TASSIGN(v236, v237);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      RoundMode v238 = RoundMode::CAST_ROUND;
      SaturationMode v239 = SaturationMode::ON;
      TCVT(v236, v207, v238, v239);
      v236.SetValidShape(v191, v81);
      // pto: %6
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v240 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v66, v81);
      // pto: %6
      ;
      uint64_t v241 = (uint64_t) v37;
      TASSIGN(v240, v241);
      pipe_barrier(PIPE_V);
      TFILLPAD<pto::TFillPadMode::InPlace>(v240, v236);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      v240.SetValidShape(v66, v81);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      TPUSH<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v177, v240);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      // pto: %8
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v242 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v66, v81);
      // pto: %8
      ;
      uint64_t v243 = (uint64_t) v37;
      TASSIGN(v242, v243);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      TMUL(v242, v240, v240);
      // pto: %9
      ;
      Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v244 = Tile<TileType::Vec, float, 32, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v81);
      // pto: %9
      ;
      uint64_t v245 = (uint64_t) v35;
      TASSIGN(v244, v245);
      // pto: %10
      ;
      Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v246 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v67);
      // pto: %10
      ;
      uint64_t v247 = (uint64_t) v36;
      TASSIGN(v246, v247);
      pipe_barrier(PIPE_V);
      TROWSUM(v246, v242, v244);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      // pto: %11
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v248 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v66);
      // pto: %11
      ;
      uint64_t v249 = (uint64_t) v36;
      TASSIGN(v248, v249);
      // pto: %12
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v250 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v66);
      // pto: %12
      ;
      uint64_t v251 = (uint64_t) v34;
      TASSIGN(v250, v251);
      pipe_barrier(PIPE_V);
      TADD(v250, v234, v248);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    // pto: %50, %51, %74, %rms_lane_inline1635__ssa_v0, %73
    ;
    int64_t v252 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v181 * (uint64_t) v16) + (uint64_t) v180) + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) * (uint64_t) v66));
    // pto: %13
    ;
    Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v253 = Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v67);
    // pto: %13
    ;
    uint64_t v254 = (uint64_t) v34;
    TASSIGN(v253, v254);
    // pto: %75
    ;
    int64_t v255 = v252 < v78 ? v78 : v252;
    // pto: %rms_partials_inline1725__iter_v1_pview
    ;
    const int64_t v256 = 0;
    // pto: %rms_partials_inline1725__iter_v1_pview
    ;
    __gm__ float* v257 = PTOAS__GLOBAL_TENSOR_DATA(v105);
    // pto: %rms_partials_inline1725__iter_v1_pview
    ;
    const int64_t v258 = 1;
    // pto: %rms_partials_inline1725__iter_v1_pview
    ;
    const int64_t v259 = 1;
    // pto: %rms_partials_inline1725__iter_v1_pview
    ;
    const int64_t v260 = 1;
    // pto: %rms_partials_inline1725__iter_v1_pview
    ;
    int64_t v261 = v66 * v67;
    // pto: %rms_partials_inline1725__iter_v1_pview
    ;
    int64_t v262 = v260 * v261;
    // pto: %rms_partials_inline1725__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 32, 1> v263 = pto::Shape<1, 1, 1, 32, 1>(v258, v259, v260, v66, v67);
    // pto: %rms_partials_inline1725__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v264 = pto::Stride<-1, -1, -1, -1, -1>(v259 * v262, v262, v261, v67, v97);
    // pto: %rms_partials_inline1725__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v265 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v257 + (v256 + v255 * v67 + v78 * v97), v263, v264);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    TSTORE(v265, v253);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  pipe_barrier(PIPE_ALL);
  dsb(DSB_DDR);
  SYNCALL<SyncCoreType::Mix>();
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  // pto: %76, %77, %tail_lane_inline1704__ssa_v0
  int64_t v266 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v178 * (uint64_t) v80) + (uint64_t) ((int64_t) v33));
  // pto: %78
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  if (v266 < v18) {
    // pto: %80
    ;
    int64_t v267 = v266 < v20 ? v266 : v20;
    // pto: %79, %81
    ;
    int64_t v268 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v266 * (uint64_t) v19) + (uint64_t) v267);
    // pto: %83
    ;
    int64_t v269 = (int64_t) ((uint64_t) v20 - (uint64_t) v266);
    // pto: %84
    ;
    int64_t v270 = v269 < v78 ? v78 : v269;
    // pto: %85
    ;
    int64_t v271 = v270 < v67 ? v270 : v67;
    // pto: %82, %86
    ;
    int64_t v272 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v268 + (uint64_t) v19) + (uint64_t) v271);
    for (int64_t v273 = v268; v273 < v272; v273 += v71) {
      // pto: %scale2_inline1609__tile
      ;
      float v274 = (v5)[v80];
      // pto: %87
      ;
      int64_t v275 = (int64_t) ((uint64_t) v272 - (uint64_t) v273);
      // pto: %88
      ;
      int64_t v276 = v275 < v71 ? v275 : v71;
      // pto: %89
      ;
      int64_t v277 = (int64_t) ((uint64_t) v276 * (uint64_t) v66);
      // pto: %c_parts_load_inline1646__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v278 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v277);
      // pto: %c_parts_load_inline1646__ssa_v0
      ;
      uint64_t v279 = (uint64_t) v37;
      TASSIGN(v278, v279);
      // pto: %90
      ;
      int64_t v280 = (int64_t) ((uint64_t) v273 * (uint64_t) v66);
      // pto: %91
      ;
      int64_t v281 = v280 < v78 ? v78 : v280;
      // pto: %parts_2d_inline1634__ssa_v0_pview
      ;
      const int64_t v282 = 0;
      // pto: %parts_2d_inline1634__ssa_v0_pview
      ;
      __gm__ float* v283 = PTOAS__GLOBAL_TENSOR_DATA(v123);
      // pto: %parts_2d_inline1634__ssa_v0_pview
      ;
      const int64_t v284 = 1;
      // pto: %parts_2d_inline1634__ssa_v0_pview
      ;
      const int64_t v285 = 1;
      // pto: %parts_2d_inline1634__ssa_v0_pview
      ;
      const int64_t v286 = 1;
      // pto: %parts_2d_inline1634__ssa_v0_pview
      ;
      int64_t v287 = v71 * v115;
      // pto: %parts_2d_inline1634__ssa_v0_pview
      ;
      int64_t v288 = v286 * v287;
      // pto: %parts_2d_inline1634__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, -1> v289 = pto::Shape<1, 1, 1, 8, -1>(v284, v285, v286, v71, v277);
      // pto: %parts_2d_inline1634__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v290 = pto::Stride<-1, -1, -1, -1, -1>(v285 * v288, v288, v287, v115, v67);
      // pto: %parts_2d_inline1634__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v291 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v283 + (v282 + v78 * v115 + v281 * v67), v289, v290);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(v278, v291);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %t__tmp_v451
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v292 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v71, v82);
      // pto: %t__tmp_v451
      ;
      uint64_t v293 = (uint64_t) v37;
      TASSIGN(v292, v293);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      TFILLPAD<pto::TFillPadMode::InPlace>(v292, v278);
      v292.SetValidShape(v71, v82);
      // pto: %t__tmp_v452
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v294 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v82);
      // pto: %t__tmp_v452
      ;
      uint64_t v295 = (uint64_t) v37;
      TASSIGN(v294, v295);
      pipe_barrier(PIPE_V);
      TCOLSUM(v294, v292);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      // pto: %c_total_inline1758__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v296 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v66);
      // pto: %c_total_inline1758__ssa_v0
      ;
      uint64_t v297 = (uint64_t) v37;
      TASSIGN(v296, v297);
      // pto: %c_sq_load_inline1606__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v298 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v276);
      // pto: %c_sq_load_inline1606__ssa_v0
      ;
      uint64_t v299 = (uint64_t) v39;
      TASSIGN(v298, v299);
      // pto: %93
      ;
      int64_t v300 = v273 < v78 ? v78 : v273;
      // pto: %rms_2d_inline1721__ssa_v0_pview
      ;
      int64_t v301 = (int64_t) v29;
      // pto: %rms_2d_inline1721__ssa_v0_pview
      ;
      const int64_t v302 = 0;
      // pto: %rms_2d_inline1721__ssa_v0_pview
      ;
      __gm__ float* v303 = PTOAS__GLOBAL_TENSOR_DATA(v132);
      // pto: %rms_2d_inline1721__ssa_v0_pview
      ;
      const int64_t v304 = 1;
      // pto: %rms_2d_inline1721__ssa_v0_pview
      ;
      const int64_t v305 = 1;
      // pto: %rms_2d_inline1721__ssa_v0_pview
      ;
      const int64_t v306 = 1;
      // pto: %rms_2d_inline1721__ssa_v0_pview
      ;
      int64_t v307 = v71 * v301;
      // pto: %rms_2d_inline1721__ssa_v0_pview
      ;
      int64_t v308 = v306 * v307;
      // pto: %rms_2d_inline1721__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, -1> v309 = pto::Shape<1, 1, 1, 8, -1>(v304, v305, v306, v71, v276);
      // pto: %rms_2d_inline1721__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v310 = pto::Stride<-1, -1, -1, -1, -1>(v305 * v308, v308, v307, v301, v67);
      // pto: %rms_2d_inline1721__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v311 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v303 + (v302 + v78 * v301 + v300 * v67), v309, v310);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v298, v311);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %t__tmp_v453
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v312 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v71, v71);
      // pto: %t__tmp_v453
      ;
      uint64_t v313 = (uint64_t) v39;
      TASSIGN(v312, v313);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      TFILLPAD<pto::TFillPadMode::InPlace>(v312, v298);
      v312.SetValidShape(v71, v71);
      // pto: %c_rms_tmp_inline1715__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v314 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v71);
      // pto: %c_rms_tmp_inline1715__ssa_v0
      ;
      uint64_t v315 = (uint64_t) v40;
      TASSIGN(v314, v315);
      // pto: %t__tmp_v454
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v316 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v71);
      // pto: %t__tmp_v454
      ;
      uint64_t v317 = (uint64_t) v39;
      TASSIGN(v316, v317);
      pipe_barrier(PIPE_V);
      TCOLSUM(v316, v312);
      // pto: %t__tmp_v455
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v318 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v71);
      // pto: %t__tmp_v455
      ;
      uint64_t v319 = (uint64_t) v39;
      TASSIGN(v318, v319);
      pipe_barrier(PIPE_V);
      TMULS(v318, v316, v83);
      // pto: %t__tmp_v456
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v320 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v71);
      // pto: %t__tmp_v456
      ;
      uint64_t v321 = (uint64_t) v39;
      TASSIGN(v320, v321);
      pipe_barrier(PIPE_V);
      TADDS(v320, v318, v84);
      // pto: %t__tmp_v457
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v322 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v71);
      // pto: %t__tmp_v457
      ;
      uint64_t v323 = (uint64_t) v41;
      TASSIGN(v322, v323);
      pipe_barrier(PIPE_V);
      TRSQRT(v322, v320, v314);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      // pto: %c_inv_inline1604__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v324 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v67);
      // pto: %c_inv_inline1604__ssa_v0
      ;
      uint64_t v325 = (uint64_t) v41;
      TASSIGN(v324, v325);
      // pto: %comb_inv_inline1736__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v326 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v67);
      // pto: %comb_inv_inline1736__ssa_v0
      ;
      uint64_t v327 = (uint64_t) v41;
      TASSIGN(v326, v327);
      // pto: %t__tmp_v458
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v328 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v72);
      // pto: %t__tmp_v458
      ;
      uint64_t v329 = (uint64_t) v42;
      TASSIGN(v328, v329);
      // pto: %slice_view
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v330;
      // pto: %slice_view
      ;
      uint64_t v331 = (uint64_t) v42;
      TASSIGN(v330, v331);
      // pto: %raw_inline1714__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v332 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v72);
      // pto: %raw_inline1714__ssa_v0
      ;
      uint64_t v333 = (uint64_t) v37;
      TASSIGN(v332, v333);
      TADDS(v332, v330, v77);
      // pto: %base_inline1778__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v334 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v72);
      // pto: %base_inline1778__ssa_v0
      ;
      uint64_t v335 = (uint64_t) v39;
      TASSIGN(v334, v335);
      // pto: %hc_base_2d_inline1856__ssa_v0_pview
      ;
      __gm__ float* v336 = PTOAS__GLOBAL_TENSOR_DATA(v140);
      // pto: %hc_base_2d_inline1856__ssa_v0_pview
      ;
      unsigned v337 = 8;
      // pto: %hc_base_2d_inline1856__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 16> v338 = pto::Shape<1, 1, 1, 1, 16>();
      // pto: %hc_base_2d_inline1856__ssa_v0_pview
      ;
      pto::Stride<24, 24, 24, 24, 1> v339 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %hc_base_2d_inline1856__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v340 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v336 + v337, v338, v339);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      TLOAD(v334, v340);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      // pto: %t__tmp_v459
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v341 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v72);
      // pto: %t__tmp_v459
      ;
      uint64_t v342 = (uint64_t) v43;
      TASSIGN(v341, v342);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v341, v332, v326);
      // pto: %t__tmp_v460
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v343 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v72);
      // pto: %t__tmp_v460
      ;
      uint64_t v344 = (uint64_t) v43;
      TASSIGN(v343, v344);
      pipe_barrier(PIPE_V);
      TMULS(v343, v341, v274);
      // pto: %t__tmp_v461
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v345 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v72);
      // pto: %t__tmp_v461
      ;
      uint64_t v346 = (uint64_t) v37;
      TASSIGN(v345, v346);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      TCOLEXPAND(v345, v334);
      // pto: %logits_inline1779__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v347 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v72);
      // pto: %logits_inline1779__ssa_v0
      ;
      uint64_t v348 = (uint64_t) v44;
      TASSIGN(v347, v348);
      pipe_barrier(PIPE_V);
      TADD(v347, v343, v345);
      // pto: %t__tmp_v462
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v349 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v72);
      // pto: %t__tmp_v462
      ;
      uint64_t v350 = (uint64_t) v44;
      TASSIGN(v349, v350);
      // pto: %regular_logits_inline1774__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v351 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %regular_logits_inline1774__ssa_v0
      ;
      uint64_t v352 = (uint64_t) v44;
      TASSIGN(v351, v352);
      // pto: %ids_inline1783__ci_tmp_v0
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v353 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v86);
      // pto: %ids_inline1783__ci_tmp_v0
      ;
      uint64_t v354 = (uint64_t) v37;
      TASSIGN(v353, v354);
      // pto: %ids_inline1783__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v355 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %ids_inline1783__ssa_v1
      ;
      uint64_t v356 = (uint64_t) v40;
      TASSIGN(v355, v356);
      // pto: %ci_tmp_view
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v357;
      TRESHAPE(v357, v353);
      // pto: %ci_dst_view
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v358;
      TRESHAPE(v358, v355);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v358, v74, v357);
      // pto: %tmp_inline1785__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v359 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %tmp_inline1785__ssa_v1
      ;
      uint64_t v360 = (uint64_t) v37;
      TASSIGN(v359, v360);
      // pto: %idx_float_inline1833__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v361 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %idx_float_inline1833__ssa_v0
      ;
      uint64_t v362 = (uint64_t) v40;
      TASSIGN(v361, v362);
      pipe_barrier(PIPE_V);
      RoundMode v363 = RoundMode::CAST_ROUND;
      SaturationMode v364 = SaturationMode::ON;
      TCVT(v361, v355, v363, v364);
      // pto: %t__tmp_v463
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v365 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v463
      ;
      uint64_t v366 = (uint64_t) v45;
      TASSIGN(v365, v366);
      pipe_barrier(PIPE_V);
      TMULS(v365, v361, v87);
      // pto: %t__tmp_v464
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v367 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v464
      ;
      uint64_t v368 = (uint64_t) v45;
      TASSIGN(v367, v368);
      pipe_barrier(PIPE_V);
      RoundMode v369 = RoundMode::CAST_FLOOR;
      SaturationMode v370 = SaturationMode::ON;
      TCVT(v367, v365, v369, v370);
      // pto: %ri1_low_inline1835__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v371 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %ri1_low_inline1835__ssa_v0
      ;
      uint64_t v372 = (uint64_t) v45;
      TASSIGN(v371, v372);
      pipe_barrier(PIPE_V);
      RoundMode v373 = RoundMode::CAST_ROUND;
      SaturationMode v374 = SaturationMode::ON;
      TCVT(v371, v367, v373, v374);
      // pto: %t__tmp_v465
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v375 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v465
      ;
      uint64_t v376 = (uint64_t) v46;
      TASSIGN(v375, v376);
      TMULS(v375, v361, v88);
      // pto: %t__tmp_v466
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v377 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v466
      ;
      uint64_t v378 = (uint64_t) v46;
      TASSIGN(v377, v378);
      pipe_barrier(PIPE_V);
      RoundMode v379 = RoundMode::CAST_FLOOR;
      SaturationMode v380 = SaturationMode::ON;
      TCVT(v377, v375, v379, v380);
      // pto: %ri1_high_inline1838__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v381 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %ri1_high_inline1838__ssa_v0
      ;
      uint64_t v382 = (uint64_t) v46;
      TASSIGN(v381, v382);
      pipe_barrier(PIPE_V);
      RoundMode v383 = RoundMode::CAST_ROUND;
      SaturationMode v384 = SaturationMode::ON;
      TCVT(v381, v377, v383, v384);
      // pto: %t__tmp_v467
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v385 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v467
      ;
      uint64_t v386 = (uint64_t) v46;
      TASSIGN(v385, v386);
      pipe_barrier(PIPE_V);
      TMULS(v385, v381, v89);
      // pto: %ri1_bit_inline1840__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v387 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %ri1_bit_inline1840__ssa_v0
      ;
      uint64_t v388 = (uint64_t) v45;
      TASSIGN(v387, v388);
      pipe_barrier(PIPE_V);
      TSUB(v387, v371, v385);
      // pto: %t__tmp_v468
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v389 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v468
      ;
      uint64_t v390 = (uint64_t) v45;
      TASSIGN(v389, v390);
      pipe_barrier(PIPE_V);
      TMULS(v389, v387, v89);
      // pto: %t__tmp_v469
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v391 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v469
      ;
      uint64_t v392 = (uint64_t) v45;
      TASSIGN(v391, v392);
      pipe_barrier(PIPE_V);
      TSUB(v391, v361, v389);
      // pto: %t__tmp_v470
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v393 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v470
      ;
      uint64_t v394 = (uint64_t) v45;
      TASSIGN(v393, v394);
      pipe_barrier(PIPE_V);
      TADDS(v393, v391, v87);
      // pto: %ri1_inline1803__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v395 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %ri1_inline1803__ssa_v1
      ;
      uint64_t v396 = (uint64_t) v45;
      TASSIGN(v395, v396);
      pipe_barrier(PIPE_V);
      RoundMode v397 = RoundMode::CAST_ROUND;
      SaturationMode v398 = SaturationMode::ON;
      TCVT(v395, v393, v397, v398);
      // pto: %t__tmp_v471
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v399 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v471
      ;
      uint64_t v400 = (uint64_t) v46;
      TASSIGN(v399, v400);
      TMULS(v399, v361, v88);
      // pto: %t__tmp_v472
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v401 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v472
      ;
      uint64_t v402 = (uint64_t) v46;
      TASSIGN(v401, v402);
      pipe_barrier(PIPE_V);
      RoundMode v403 = RoundMode::CAST_FLOOR;
      SaturationMode v404 = SaturationMode::ON;
      TCVT(v401, v399, v403, v404);
      // pto: %ri2_low_inline1738__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v405 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %ri2_low_inline1738__ssa_v0
      ;
      uint64_t v406 = (uint64_t) v46;
      TASSIGN(v405, v406);
      pipe_barrier(PIPE_V);
      RoundMode v407 = RoundMode::CAST_ROUND;
      SaturationMode v408 = SaturationMode::ON;
      TCVT(v405, v401, v407, v408);
      // pto: %t__tmp_v473
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v409 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v473
      ;
      uint64_t v410 = (uint64_t) v47;
      TASSIGN(v409, v410);
      TMULS(v409, v361, v90);
      // pto: %t__tmp_v474
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v411 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v474
      ;
      uint64_t v412 = (uint64_t) v47;
      TASSIGN(v411, v412);
      pipe_barrier(PIPE_V);
      RoundMode v413 = RoundMode::CAST_FLOOR;
      SaturationMode v414 = SaturationMode::ON;
      TCVT(v411, v409, v413, v414);
      // pto: %ri2_high_inline1730__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v415 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %ri2_high_inline1730__ssa_v0
      ;
      uint64_t v416 = (uint64_t) v47;
      TASSIGN(v415, v416);
      pipe_barrier(PIPE_V);
      RoundMode v417 = RoundMode::CAST_ROUND;
      SaturationMode v418 = SaturationMode::ON;
      TCVT(v415, v411, v417, v418);
      // pto: %t__tmp_v475
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v419 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v475
      ;
      uint64_t v420 = (uint64_t) v47;
      TASSIGN(v419, v420);
      pipe_barrier(PIPE_V);
      TMULS(v419, v415, v89);
      // pto: %ri2_bit_inline1843__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v421 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %ri2_bit_inline1843__ssa_v0
      ;
      uint64_t v422 = (uint64_t) v46;
      TASSIGN(v421, v422);
      pipe_barrier(PIPE_V);
      TSUB(v421, v405, v419);
      // pto: %t__tmp_v476
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v423 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v476
      ;
      uint64_t v424 = (uint64_t) v46;
      TASSIGN(v423, v424);
      pipe_barrier(PIPE_V);
      TMULS(v423, v421, v91);
      // pto: %t__tmp_v477
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v425 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v477
      ;
      uint64_t v426 = (uint64_t) v46;
      TASSIGN(v425, v426);
      pipe_barrier(PIPE_V);
      TSUB(v425, v361, v423);
      // pto: %t__tmp_v478
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v427 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v478
      ;
      uint64_t v428 = (uint64_t) v46;
      TASSIGN(v427, v428);
      pipe_barrier(PIPE_V);
      TADDS(v427, v425, v89);
      // pto: %ri2_inline1844__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v429 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %ri2_inline1844__ssa_v0
      ;
      uint64_t v430 = (uint64_t) v46;
      TASSIGN(v429, v430);
      pipe_barrier(PIPE_V);
      RoundMode v431 = RoundMode::CAST_ROUND;
      SaturationMode v432 = SaturationMode::ON;
      TCVT(v429, v427, v431, v432);
      // pto: %t__tmp_v479
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v433 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v479
      ;
      uint64_t v434 = (uint64_t) v47;
      TASSIGN(v433, v434);
      TMULS(v433, v361, v90);
      // pto: %t__tmp_v480
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v435 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v480
      ;
      uint64_t v436 = (uint64_t) v47;
      TASSIGN(v435, v436);
      pipe_barrier(PIPE_V);
      RoundMode v437 = RoundMode::CAST_FLOOR;
      SaturationMode v438 = SaturationMode::ON;
      TCVT(v435, v433, v437, v438);
      // pto: %ci1_low_inline1845__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v439 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %ci1_low_inline1845__ssa_v0
      ;
      uint64_t v440 = (uint64_t) v47;
      TASSIGN(v439, v440);
      pipe_barrier(PIPE_V);
      RoundMode v441 = RoundMode::CAST_ROUND;
      SaturationMode v442 = SaturationMode::ON;
      TCVT(v439, v435, v441, v442);
      // pto: %t__tmp_v481
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v443 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v481
      ;
      uint64_t v444 = (uint64_t) v48;
      TASSIGN(v443, v444);
      TMULS(v443, v361, v92);
      // pto: %t__tmp_v482
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v445 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v482
      ;
      uint64_t v446 = (uint64_t) v48;
      TASSIGN(v445, v446);
      pipe_barrier(PIPE_V);
      RoundMode v447 = RoundMode::CAST_FLOOR;
      SaturationMode v448 = SaturationMode::ON;
      TCVT(v445, v443, v447, v448);
      // pto: %ci1_high_inline1847__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v449 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %ci1_high_inline1847__ssa_v0
      ;
      uint64_t v450 = (uint64_t) v48;
      TASSIGN(v449, v450);
      pipe_barrier(PIPE_V);
      RoundMode v451 = RoundMode::CAST_ROUND;
      SaturationMode v452 = SaturationMode::ON;
      TCVT(v449, v445, v451, v452);
      // pto: %t__tmp_v483
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v453 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v483
      ;
      uint64_t v454 = (uint64_t) v48;
      TASSIGN(v453, v454);
      pipe_barrier(PIPE_V);
      TMULS(v453, v449, v89);
      // pto: %ci1_bit_inline1848__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v455 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %ci1_bit_inline1848__ssa_v0
      ;
      uint64_t v456 = (uint64_t) v47;
      TASSIGN(v455, v456);
      pipe_barrier(PIPE_V);
      TSUB(v455, v439, v453);
      // pto: %t__tmp_v484
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v457 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v484
      ;
      uint64_t v458 = (uint64_t) v47;
      TASSIGN(v457, v458);
      pipe_barrier(PIPE_V);
      TMULS(v457, v455, v93);
      // pto: %t__tmp_v485
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v459 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v485
      ;
      uint64_t v460 = (uint64_t) v47;
      TASSIGN(v459, v460);
      pipe_barrier(PIPE_V);
      TSUB(v459, v361, v457);
      // pto: %t__tmp_v486
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v461 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v486
      ;
      uint64_t v462 = (uint64_t) v47;
      TASSIGN(v461, v462);
      pipe_barrier(PIPE_V);
      TADDS(v461, v459, v91);
      // pto: %ci1_inline1858__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v463 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %ci1_inline1858__ssa_v0
      ;
      uint64_t v464 = (uint64_t) v47;
      TASSIGN(v463, v464);
      pipe_barrier(PIPE_V);
      RoundMode v465 = RoundMode::CAST_ROUND;
      SaturationMode v466 = SaturationMode::ON;
      TCVT(v463, v461, v465, v466);
      // pto: %t__tmp_v487
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v467 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v487
      ;
      uint64_t v468 = (uint64_t) v48;
      TASSIGN(v467, v468);
      TMULS(v467, v361, v92);
      // pto: %t__tmp_v488
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v469 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v488
      ;
      uint64_t v470 = (uint64_t) v48;
      TASSIGN(v469, v470);
      pipe_barrier(PIPE_V);
      RoundMode v471 = RoundMode::CAST_FLOOR;
      SaturationMode v472 = SaturationMode::ON;
      TCVT(v469, v467, v471, v472);
      // pto: %ci2_low_inline1849__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v473 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %ci2_low_inline1849__ssa_v0
      ;
      uint64_t v474 = (uint64_t) v48;
      TASSIGN(v473, v474);
      pipe_barrier(PIPE_V);
      RoundMode v475 = RoundMode::CAST_ROUND;
      SaturationMode v476 = SaturationMode::ON;
      TCVT(v473, v469, v475, v476);
      // pto: %t__tmp_v489
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v477 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v489
      ;
      uint64_t v478 = (uint64_t) v49;
      TASSIGN(v477, v478);
      TMULS(v477, v361, v94);
      // pto: %t__tmp_v490
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v479 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v490
      ;
      uint64_t v480 = (uint64_t) v49;
      TASSIGN(v479, v480);
      pipe_barrier(PIPE_V);
      RoundMode v481 = RoundMode::CAST_FLOOR;
      SaturationMode v482 = SaturationMode::ON;
      TCVT(v479, v477, v481, v482);
      // pto: %ci2_high_inline1851__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v483 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %ci2_high_inline1851__ssa_v0
      ;
      uint64_t v484 = (uint64_t) v49;
      TASSIGN(v483, v484);
      pipe_barrier(PIPE_V);
      RoundMode v485 = RoundMode::CAST_ROUND;
      SaturationMode v486 = SaturationMode::ON;
      TCVT(v483, v479, v485, v486);
      // pto: %t__tmp_v491
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v487 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v491
      ;
      uint64_t v488 = (uint64_t) v49;
      TASSIGN(v487, v488);
      pipe_barrier(PIPE_V);
      TMULS(v487, v483, v89);
      // pto: %ci2_bit_inline1798__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v489 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %ci2_bit_inline1798__ssa_v0
      ;
      uint64_t v490 = (uint64_t) v48;
      TASSIGN(v489, v490);
      pipe_barrier(PIPE_V);
      TSUB(v489, v473, v487);
      // pto: %t__tmp_v492
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v491 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v492
      ;
      uint64_t v492 = (uint64_t) v48;
      TASSIGN(v491, v492);
      pipe_barrier(PIPE_V);
      TMULS(v491, v489, v95);
      // pto: %t__tmp_v493
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v493 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v493
      ;
      uint64_t v494 = (uint64_t) v40;
      TASSIGN(v493, v494);
      pipe_barrier(PIPE_V);
      TSUB(v493, v361, v491);
      // pto: %t__tmp_v494
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v495 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v494
      ;
      uint64_t v496 = (uint64_t) v40;
      TASSIGN(v495, v496);
      pipe_barrier(PIPE_V);
      TADDS(v495, v493, v93);
      // pto: %ci2_inline1731__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v497 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %ci2_inline1731__ssa_v0
      ;
      uint64_t v498 = (uint64_t) v40;
      TASSIGN(v497, v498);
      pipe_barrier(PIPE_V);
      RoundMode v499 = RoundMode::CAST_ROUND;
      SaturationMode v500 = SaturationMode::ON;
      TCVT(v497, v495, v499, v500);
      // pto: %t__tmp_v495
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v501 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v495
      ;
      uint64_t v502 = (uint64_t) v48;
      TASSIGN(v501, v502);
      TGATHER(v501, v351, v395, v359);
      // pto: %max2_inline1753__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v503 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %max2_inline1753__ssa_v1
      ;
      uint64_t v504 = (uint64_t) v48;
      TASSIGN(v503, v504);
      pipe_barrier(PIPE_V);
      TMAX(v503, v351, v501);
      // pto: %t__tmp_v496
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v505 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v496
      ;
      uint64_t v506 = (uint64_t) v49;
      TASSIGN(v505, v506);
      pipe_barrier(PIPE_V);
      TGATHER(v505, v503, v429, v359);
      // pto: %max4_inline1811__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v507 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %max4_inline1811__ssa_v1
      ;
      uint64_t v508 = (uint64_t) v48;
      TASSIGN(v507, v508);
      pipe_barrier(PIPE_V);
      TMAX(v507, v503, v505);
      // pto: %t__tmp_v497
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v509 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v497
      ;
      uint64_t v510 = (uint64_t) v43;
      TASSIGN(v509, v510);
      pipe_barrier(PIPE_V);
      TSUB(v509, v351, v507);
      // pto: %exps_inline1824__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v511 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %exps_inline1824__ssa_v1
      ;
      uint64_t v512 = (uint64_t) v43;
      TASSIGN(v511, v512);
      pipe_barrier(PIPE_V);
      TEXP(v511, v509);
      // pto: %t__tmp_v498
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v513 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v498
      ;
      uint64_t v514 = (uint64_t) v48;
      TASSIGN(v513, v514);
      pipe_barrier(PIPE_V);
      TGATHER(v513, v511, v395, v359);
      // pto: %sum2_inline1618__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v515 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %sum2_inline1618__ssa_v1
      ;
      uint64_t v516 = (uint64_t) v48;
      TASSIGN(v515, v516);
      pipe_barrier(PIPE_V);
      TADD(v515, v511, v513);
      // pto: %t__tmp_v499
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v517 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v499
      ;
      uint64_t v518 = (uint64_t) v49;
      TASSIGN(v517, v518);
      pipe_barrier(PIPE_V);
      TGATHER(v517, v515, v429, v359);
      // pto: %sum4_inline1814__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v519 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %sum4_inline1814__ssa_v1
      ;
      uint64_t v520 = (uint64_t) v48;
      TASSIGN(v519, v520);
      pipe_barrier(PIPE_V);
      TADD(v519, v515, v517);
      // pto: %t__tmp_v500
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v521 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v500
      ;
      uint64_t v522 = (uint64_t) v43;
      TASSIGN(v521, v522);
      pipe_barrier(PIPE_V);
      TDIV(v521, v511, v519);
      // pto: %probs_inline1816__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v523 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %probs_inline1816__ssa_v1
      ;
      uint64_t v524 = (uint64_t) v43;
      TASSIGN(v523, v524);
      pipe_barrier(PIPE_V);
      TADDS(v523, v521, v84);
      // pto: %t__tmp_v501
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v525 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v501
      ;
      uint64_t v526 = (uint64_t) v48;
      TASSIGN(v525, v526);
      pipe_barrier(PIPE_V);
      TGATHER(v525, v523, v463, v359);
      // pto: %col2_inline1820__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v527 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %col2_inline1820__ssa_v4
      ;
      uint64_t v528 = (uint64_t) v48;
      TASSIGN(v527, v528);
      pipe_barrier(PIPE_V);
      TADD(v527, v523, v525);
      // pto: %t__tmp_v502
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v529 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v502
      ;
      uint64_t v530 = (uint64_t) v49;
      TASSIGN(v529, v530);
      pipe_barrier(PIPE_V);
      TGATHER(v529, v527, v497, v359);
      // pto: %t__tmp_v503
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v531 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %t__tmp_v503
      ;
      uint64_t v532 = (uint64_t) v48;
      TASSIGN(v531, v532);
      pipe_barrier(PIPE_V);
      TADD(v531, v527, v529);
      // pto: %col4_inline1822__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v533 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %col4_inline1822__ssa_v4
      ;
      uint64_t v534 = (uint64_t) v48;
      TASSIGN(v533, v534);
      pipe_barrier(PIPE_V);
      TADDS(v533, v531, v84);
      // pto: %cur_inline1825__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v535 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
      // pto: %cur_inline1825__ssa_v4
      ;
      uint64_t v536 = (uint64_t) v43;
      TASSIGN(v535, v536);
      pipe_barrier(PIPE_V);
      TDIV(v535, v523, v533);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      for (int64_t v537 = v78; v537 < v96; v537 += v67) {
        // pto: %t__tmp_v504
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v538 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
        // pto: %t__tmp_v504
        ;
        uint64_t v539 = (uint64_t) v48;
        TASSIGN(v538, v539);
        pipe_barrier(PIPE_V);
        TGATHER(v538, v535, v395, v359);
        // pto: %row2_inline1837__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v540 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
        // pto: %row2_inline1837__ssa_v1
        ;
        uint64_t v541 = (uint64_t) v48;
        TASSIGN(v540, v541);
        pipe_barrier(PIPE_V);
        TADD(v540, v535, v538);
        // pto: %t__tmp_v505
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v542 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
        // pto: %t__tmp_v505
        ;
        uint64_t v543 = (uint64_t) v49;
        TASSIGN(v542, v543);
        pipe_barrier(PIPE_V);
        TGATHER(v542, v540, v429, v359);
        // pto: %t__tmp_v506
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v544 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
        // pto: %t__tmp_v506
        ;
        uint64_t v545 = (uint64_t) v48;
        TASSIGN(v544, v545);
        pipe_barrier(PIPE_V);
        TADD(v544, v540, v542);
        // pto: %row4_inline1829__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v546 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
        // pto: %row4_inline1829__ssa_v1
        ;
        uint64_t v547 = (uint64_t) v48;
        TASSIGN(v546, v547);
        pipe_barrier(PIPE_V);
        TADDS(v546, v544, v84);
        // pto: %norm_inline1742__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v548 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
        // pto: %norm_inline1742__ssa_v1
        ;
        uint64_t v549 = (uint64_t) v48;
        TASSIGN(v548, v549);
        pipe_barrier(PIPE_V);
        TDIV(v548, v535, v546);
        // pto: %t__tmp_v507
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v550 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
        // pto: %t__tmp_v507
        ;
        uint64_t v551 = (uint64_t) v49;
        TASSIGN(v550, v551);
        pipe_barrier(PIPE_V);
        TGATHER(v550, v548, v463, v359);
        // pto: %col2_inline1820__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v552 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
        // pto: %col2_inline1820__ssa_v7
        ;
        uint64_t v553 = (uint64_t) v49;
        TASSIGN(v552, v553);
        pipe_barrier(PIPE_V);
        TADD(v552, v548, v550);
        // pto: %t__tmp_v508
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v554 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
        // pto: %t__tmp_v508
        ;
        uint64_t v555 = (uint64_t) v50;
        TASSIGN(v554, v555);
        pipe_barrier(PIPE_V);
        TGATHER(v554, v552, v497, v359);
        // pto: %t__tmp_v509
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v556 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
        // pto: %t__tmp_v509
        ;
        uint64_t v557 = (uint64_t) v49;
        TASSIGN(v556, v557);
        pipe_barrier(PIPE_V);
        TADD(v556, v552, v554);
        // pto: %col4_inline1822__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v558 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
        // pto: %col4_inline1822__ssa_v7
        ;
        uint64_t v559 = (uint64_t) v49;
        TASSIGN(v558, v559);
        pipe_barrier(PIPE_V);
        TADDS(v558, v556, v84);
        // pto: %cur_inline1825__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v560 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v85);
        // pto: %cur_inline1825__ssa_v7
        ;
        uint64_t v561 = (uint64_t) v43;
        TASSIGN(v560, v561);
        pipe_barrier(PIPE_V);
        TDIV(v560, v548, v558);
      };
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      // pto: %t__tmp_v510
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v562 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v72);
      // pto: %t__tmp_v510
      ;
      uint64_t v563 = (uint64_t) v43;
      TASSIGN(v562, v563);
      v562.SetValidShape(v276, v72);
      // pto: %comb_inline1693__ssa_v0_pview
      ;
      const int64_t v564 = 0;
      // pto: %comb_inline1693__ssa_v0_pview
      ;
      __gm__ float* v565 = PTOAS__GLOBAL_TENSOR_DATA(v149);
      // pto: %comb_inline1693__ssa_v0_pview
      ;
      const int64_t v566 = 1;
      // pto: %comb_inline1693__ssa_v0_pview
      ;
      const int64_t v567 = 1;
      // pto: %comb_inline1693__ssa_v0_pview
      ;
      const int64_t v568 = 1;
      // pto: %comb_inline1693__ssa_v0_pview
      ;
      int64_t v569 = v276 * v72;
      // pto: %comb_inline1693__ssa_v0_pview
      ;
      int64_t v570 = v568 * v569;
      // pto: %comb_inline1693__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, 16> v571 = pto::Shape<1, 1, 1, -1, 16>(v566, v567, v568, v276, v72);
      // pto: %comb_inline1693__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v572 = pto::Stride<-1, -1, -1, -1, -1>(v567 * v570, v570, v569, v72, v67);
      // pto: %comb_inline1693__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v573 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v565 + (v564 + v300 * v72 + v78 * v67), v571, v572);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      TSTORE(v573, v562);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    };
  } else {
    // pto: %96
    ;
    int64_t v574 = (int64_t) ((uint64_t) v266 - (uint64_t) v18);
    // pto: %98
    ;
    int64_t v575 = v574 < v22 ? v574 : v22;
    // pto: %97, %99, %100
    ;
    int64_t v576 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v574 * (uint64_t) v21) + (uint64_t) v575) * (uint64_t) v80);
    // pto: %101
    ;
    int64_t v577 = (int64_t) ((uint64_t) v22 - (uint64_t) v574);
    // pto: %102
    ;
    int64_t v578 = v577 < v78 ? v78 : v577;
    // pto: %103
    ;
    int64_t v579 = v578 < v67 ? v578 : v67;
    // pto: %106, %104, %105
    ;
    int64_t v580 = (int64_t) ((uint64_t) v576 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v21 + (uint64_t) v579) * (uint64_t) v80));
    // pto: %107
    ;
    // pto: %108
    ;
    // pto: %d_begin_inline1860__phi_v2
    ;
    int64_t v581;
    // pto: %work_begin_inline1678__phi_v3
    ;
    int64_t v582;
    // pto: %work_end_inline1850__phi_v3
    ;
    int64_t v583;
    if (v24 < v25) {
      // pto: %109
      ;
      // pto: %d_worker_inline1625__phi_v2
      ;
      int64_t v584;
      // pto: %d_workers_inline1622__phi_v2
      ;
      int64_t v585;
      // pto: %pair_id_inline1870__phi_v2
      ;
      int64_t v586;
      if (v574 < v26) {
        // pto: %110
        ;
        int64_t v587 = (int64_t) ((uint64_t) v27 + (uint64_t) v67);
        // pto: %111
        ;
        // pto: %113
        ;
        v584 = v574 % v587;
        v585 = v587;
        v586 = v574 / v587;
      } else {
        // pto: %115
        ;
        int64_t v588 = (int64_t) ((uint64_t) v574 - (uint64_t) v26);
        // pto: %117, %116
        ;
        // pto: %118
        ;
        v584 = v588 % v27;
        v585 = v27;
        v586 = (int64_t) ((uint64_t) v28 + (uint64_t) (v588 / v27));
      };
      // pto: %119
      ;
      uint64_t v589 = (uint64_t) v586;
      // pto: %119
      ;
      int64_t v590 = (int64_t) (v589 * (uint64_t) v80);
      // pto: %120
      ;
      int64_t v591 = v23 < v590 ? v23 : v590;
      // pto: %121
      ;
      int64_t v592 = (int64_t) ((uint64_t) v591 + (uint64_t) v80);
      // pto: %122
      ;
      // pto: %123
      ;
      int64_t v593 = v71 / v585;
      // pto: %124
      ;
      int64_t v594 = v71 % v585;
      // pto: %125
      ;
      uint64_t v595 = (uint64_t) v584;
      // pto: %126
      ;
      bool v596 = v584 < v594;
      // pto: %126
      ;
      int64_t v597 = v596 ? v584 : v594;
      // pto: %125, %127
      ;
      v581 = (int64_t) ((uint64_t) ((int64_t) v595 * (uint64_t) v593) + (uint64_t) v597);
      v582 = v591;
      v583 = v23 < v592 ? v23 : v592;
    } else {
      v581 = v78;
      v582 = v576;
      v583 = v23 < v580 ? v23 : v580;
    };
    // pto: %128
    ;
    bool v598 = v25 <= v574;
    // pto: %work_begin_inline1678__phi_v5
    ;
    int64_t v599 = v598 ? v78 : v582;
    // pto: %work_end_inline1850__phi_v5
    ;
    int64_t v600 = v598 ? v78 : v583;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    for (int64_t v601 = v599; v601 < v600; v601 += v71) {
      // pto: %129
      ;
      int64_t v602 = (int64_t) ((uint64_t) v600 - (uint64_t) v601);
      // pto: %130
      ;
      int64_t v603 = v602 < v71 ? v602 : v71;
      // pto: %131
      ;
      int64_t v604 = (int64_t) ((uint64_t) v603 * (uint64_t) v66);
      // pto: %m_parts_load_inline1710__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v605 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v604);
      // pto: %m_parts_load_inline1710__ssa_v0
      ;
      uint64_t v606 = (uint64_t) v37;
      TASSIGN(v605, v606);
      // pto: %132
      ;
      int64_t v607 = (int64_t) ((uint64_t) v601 * (uint64_t) v66);
      // pto: %133
      ;
      int64_t v608 = v607 < v78 ? v78 : v607;
      // pto: %135
      ;
      const int64_t v609 = 0;
      // pto: %135
      ;
      __gm__ float* v610 = PTOAS__GLOBAL_TENSOR_DATA(v123);
      // pto: %135
      ;
      const int64_t v611 = 1;
      // pto: %135
      ;
      const int64_t v612 = 1;
      // pto: %135
      ;
      const int64_t v613 = 1;
      // pto: %135
      ;
      int64_t v614 = v71 * v115;
      // pto: %135
      ;
      int64_t v615 = v613 * v614;
      // pto: %135
      ;
      pto::Shape<1, 1, 1, 8, -1> v616 = pto::Shape<1, 1, 1, 8, -1>(v611, v612, v613, v71, v604);
      // pto: %135
      ;
      pto::Stride<-1, -1, -1, -1, -1> v617 = pto::Stride<-1, -1, -1, -1, -1>(v612 * v615, v615, v614, v115, v67);
      // pto: %135
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v618 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v610 + (v609 + v78 * v115 + v608 * v67), v616, v617);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      TLOAD(v605, v618);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      // pto: %t__tmp_v511
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v619 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v71, v82);
      // pto: %t__tmp_v511
      ;
      uint64_t v620 = (uint64_t) v37;
      TASSIGN(v619, v620);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      TFILLPAD<pto::TFillPadMode::InPlace>(v619, v605);
      v619.SetValidShape(v71, v82);
      // pto: %t__tmp_v512
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v621 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v82);
      // pto: %t__tmp_v512
      ;
      uint64_t v622 = (uint64_t) v37;
      TASSIGN(v621, v622);
      pipe_barrier(PIPE_V);
      TCOLSUM(v621, v619);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      // pto: %m_total_inline1878__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v623 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v66);
      // pto: %m_total_inline1878__ssa_v0
      ;
      uint64_t v624 = (uint64_t) v37;
      TASSIGN(v623, v624);
      // pto: %m_sq_load_inline1880__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v625 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v603);
      // pto: %m_sq_load_inline1880__ssa_v0
      ;
      uint64_t v626 = (uint64_t) v39;
      TASSIGN(v625, v626);
      // pto: %136
      ;
      int64_t v627 = v601 < v78 ? v78 : v601;
      // pto: %137
      ;
      int64_t v628 = (int64_t) v29;
      // pto: %137
      ;
      const int64_t v629 = 0;
      // pto: %137
      ;
      __gm__ float* v630 = PTOAS__GLOBAL_TENSOR_DATA(v132);
      // pto: %137
      ;
      const int64_t v631 = 1;
      // pto: %137
      ;
      const int64_t v632 = 1;
      // pto: %137
      ;
      const int64_t v633 = 1;
      // pto: %137
      ;
      int64_t v634 = v71 * v628;
      // pto: %137
      ;
      int64_t v635 = v633 * v634;
      // pto: %137
      ;
      pto::Shape<1, 1, 1, 8, -1> v636 = pto::Shape<1, 1, 1, 8, -1>(v631, v632, v633, v71, v603);
      // pto: %137
      ;
      pto::Stride<-1, -1, -1, -1, -1> v637 = pto::Stride<-1, -1, -1, -1, -1>(v632 * v635, v635, v634, v628, v67);
      // pto: %137
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v638 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v630 + (v629 + v78 * v628 + v627 * v67), v636, v637);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      TLOAD(v625, v638);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      // pto: %t__tmp_v513
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v639 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v71, v71);
      // pto: %t__tmp_v513
      ;
      uint64_t v640 = (uint64_t) v39;
      TASSIGN(v639, v640);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      TFILLPAD<pto::TFillPadMode::InPlace>(v639, v625);
      v639.SetValidShape(v71, v71);
      // pto: %m_rms_tmp_inline1881__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v641 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v71);
      // pto: %m_rms_tmp_inline1881__ssa_v0
      ;
      uint64_t v642 = (uint64_t) v40;
      TASSIGN(v641, v642);
      // pto: %t__tmp_v514
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v643 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v71);
      // pto: %t__tmp_v514
      ;
      uint64_t v644 = (uint64_t) v39;
      TASSIGN(v643, v644);
      pipe_barrier(PIPE_V);
      TCOLSUM(v643, v639);
      // pto: %t__tmp_v515
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v645 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v71);
      // pto: %t__tmp_v515
      ;
      uint64_t v646 = (uint64_t) v39;
      TASSIGN(v645, v646);
      pipe_barrier(PIPE_V);
      TMULS(v645, v643, v83);
      // pto: %t__tmp_v516
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v647 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v71);
      // pto: %t__tmp_v516
      ;
      uint64_t v648 = (uint64_t) v39;
      TASSIGN(v647, v648);
      pipe_barrier(PIPE_V);
      TADDS(v647, v645, v84);
      // pto: %t__tmp_v517
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v649 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v71);
      // pto: %t__tmp_v517
      ;
      uint64_t v650 = (uint64_t) v45;
      TASSIGN(v649, v650);
      pipe_barrier(PIPE_V);
      TRSQRT(v649, v647, v641);
      // pto: %m_inv_inline1883__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v651 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v67);
      // pto: %m_inv_inline1883__ssa_v0
      ;
      uint64_t v652 = (uint64_t) v45;
      TASSIGN(v651, v652);
      // pto: %gate_base_inline1812__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v653 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v71);
      // pto: %gate_base_inline1812__ssa_v0
      ;
      uint64_t v654 = (uint64_t) v51;
      TASSIGN(v653, v654);
      // pto: %138
      ;
      __gm__ float* v655 = PTOAS__GLOBAL_TENSOR_DATA(v140);
      // pto: %138
      ;
      pto::Shape<1, 1, 1, 1, 8> v656 = pto::Shape<1, 1, 1, 1, 8>();
      // pto: %138
      ;
      pto::Stride<24, 24, 24, 24, 1> v657 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %138
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v658 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v655, v656, v657);
      TLOAD(v653, v658);
      // pto: %gate_scales_inline1760__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v659 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v70);
      // pto: %gate_scales_inline1760__ssa_v0
      ;
      uint64_t v660 = (uint64_t) v52;
      TASSIGN(v659, v660);
      // pto: %hc_scale_2d_inline1757__ssa_v0_pview
      ;
      __gm__ float* v661 = PTOAS__GLOBAL_TENSOR_DATA(v157);
      // pto: %hc_scale_2d_inline1757__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 3> v662 = pto::Shape<1, 1, 1, 1, 3>();
      // pto: %hc_scale_2d_inline1757__ssa_v0_pview
      ;
      pto::Stride<3, 3, 3, 3, 1> v663 = pto::Stride<3, 3, 3, 3, 1>();
      // pto: %hc_scale_2d_inline1757__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND> v664 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND>(v661, v662, v663);
      TLOAD(v659, v664);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      // pto: %t__ci_tmp_v1
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v665 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v86);
      // pto: %t__ci_tmp_v1
      ;
      uint64_t v666 = (uint64_t) v39;
      TASSIGN(v665, v666);
      // pto: %t__tmp_v518
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v667 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v71);
      // pto: %t__tmp_v518
      ;
      uint64_t v668 = (uint64_t) v53;
      TASSIGN(v667, v668);
      // pto: %139
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v669;
      TRESHAPE(v669, v665);
      // pto: %140
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v670;
      TRESHAPE(v670, v667);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v670, v74, v669);
      // pto: %gate_ids_inline1626__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v671 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v71);
      // pto: %gate_ids_inline1626__ssa_v0
      ;
      uint64_t v672 = (uint64_t) v39;
      TASSIGN(v671, v672);
      pipe_barrier(PIPE_V);
      RoundMode v673 = RoundMode::CAST_ROUND;
      SaturationMode v674 = SaturationMode::ON;
      TCVT(v671, v667, v673, v674);
      // pto: %t__tmp_v519
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v675 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v71);
      // pto: %t__tmp_v519
      ;
      uint64_t v676 = (uint64_t) v39;
      TASSIGN(v675, v676);
      pipe_barrier(PIPE_V);
      TMULS(v675, v671, v90);
      // pto: %gate_scale_ids_inline1872__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v677 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v71);
      // pto: %gate_scale_ids_inline1872__ssa_v0
      ;
      uint64_t v678 = (uint64_t) v39;
      TASSIGN(v677, v678);
      pipe_barrier(PIPE_V);
      RoundMode v679 = RoundMode::CAST_FLOOR;
      SaturationMode v680 = SaturationMode::ON;
      TCVT(v677, v675, v679, v680);
      // pto: %gate_tmp_inline1649__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v681 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v71);
      // pto: %gate_tmp_inline1649__ssa_v0
      ;
      uint64_t v682 = (uint64_t) v54;
      TASSIGN(v681, v682);
      // pto: %gate_scale_vec_inline1756__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v683 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v71);
      // pto: %gate_scale_vec_inline1756__ssa_v0
      ;
      uint64_t v684 = (uint64_t) v41;
      TASSIGN(v683, v684);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      TGATHER(v683, v659, v677, v681);
      // pto: %t__tmp_v520
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v685 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v71);
      // pto: %t__tmp_v520
      ;
      uint64_t v686 = (uint64_t) v37;
      TASSIGN(v685, v686);
      // pto: %141
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v687;
      // pto: %141
      ;
      uint64_t v688 = (uint64_t) v37;
      TASSIGN(v687, v688);
      // pto: %gate_raw_inline1873__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v689 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v71);
      // pto: %gate_raw_inline1873__ssa_v0
      ;
      uint64_t v690 = (uint64_t) v37;
      TASSIGN(v689, v690);
      TADDS(v689, v687, v77);
      // pto: %gate_norm_inline1884__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v691 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v71);
      // pto: %gate_norm_inline1884__ssa_v0
      ;
      uint64_t v692 = (uint64_t) v37;
      TASSIGN(v691, v692);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v691, v689, v651);
      // pto: %gate_scaled_inline1885__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v693 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v71);
      // pto: %gate_scaled_inline1885__ssa_v0
      ;
      uint64_t v694 = (uint64_t) v37;
      TASSIGN(v693, v694);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v693, v691, v683);
      // pto: %t__tmp_v521
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v695 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v71);
      // pto: %t__tmp_v521
      ;
      uint64_t v696 = (uint64_t) v55;
      TASSIGN(v695, v696);
      TCOLEXPAND(v695, v653);
      // pto: %gate_logits_inline1703__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v697 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v71);
      // pto: %gate_logits_inline1703__ssa_v0
      ;
      uint64_t v698 = (uint64_t) v37;
      TASSIGN(v697, v698);
      pipe_barrier(PIPE_V);
      TADD(v697, v693, v695);
      // pto: %t__tmp_v522
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v699 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v71);
      // pto: %t__tmp_v522
      ;
      uint64_t v700 = (uint64_t) v37;
      TASSIGN(v699, v700);
      pipe_barrier(PIPE_V);
      TNEG(v699, v697);
      // pto: %t__tmp_v523
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v701 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v71);
      // pto: %t__tmp_v523
      ;
      uint64_t v702 = (uint64_t) v37;
      TASSIGN(v701, v702);
      pipe_barrier(PIPE_V);
      TEXP(v701, v699);
      // pto: %t__tmp_v524
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v703 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v71);
      // pto: %t__tmp_v524
      ;
      uint64_t v704 = (uint64_t) v37;
      TASSIGN(v703, v704);
      pipe_barrier(PIPE_V);
      TADDS(v703, v701, v87);
      // pto: %gate_sigmoid_inline1886__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v705 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v71, v71);
      // pto: %gate_sigmoid_inline1886__ssa_v0
      ;
      uint64_t v706 = (uint64_t) v56;
      TASSIGN(v705, v706);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
      TRECIP(v705, v703);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      // pto: %t__ci_tmp_v2
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v707 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v86);
      // pto: %t__ci_tmp_v2
      ;
      uint64_t v708 = (uint64_t) v37;
      TASSIGN(v707, v708);
      // pto: %t__tmp_v525
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v709 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v66);
      // pto: %t__tmp_v525
      ;
      uint64_t v710 = (uint64_t) v44;
      TASSIGN(v709, v710);
      // pto: %142
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v711;
      TRESHAPE(v711, v707);
      // pto: %143
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v712;
      TRESHAPE(v712, v709);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v712, v74, v711);
      // pto: %post_ids_float_inline1763__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v713 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v66);
      // pto: %post_ids_float_inline1763__ssa_v0
      ;
      uint64_t v714 = (uint64_t) v37;
      TASSIGN(v713, v714);
      pipe_barrier(PIPE_V);
      RoundMode v715 = RoundMode::CAST_ROUND;
      SaturationMode v716 = SaturationMode::ON;
      TCVT(v713, v709, v715, v716);
      // pto: %t__tmp_v526
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v717 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v66);
      // pto: %t__tmp_v526
      ;
      uint64_t v718 = (uint64_t) v57;
      TASSIGN(v717, v718);
      pipe_barrier(PIPE_V);
      TMULS(v717, v713, v90);
      // pto: %t__tmp_v527
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v719 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v66);
      // pto: %t__tmp_v527
      ;
      uint64_t v720 = (uint64_t) v57;
      TASSIGN(v719, v720);
      pipe_barrier(PIPE_V);
      RoundMode v721 = RoundMode::CAST_FLOOR;
      SaturationMode v722 = SaturationMode::ON;
      TCVT(v719, v717, v721, v722);
      // pto: %post_rows_inline1781__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v723 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v66);
      // pto: %post_rows_inline1781__ssa_v0
      ;
      uint64_t v724 = (uint64_t) v57;
      TASSIGN(v723, v724);
      pipe_barrier(PIPE_V);
      RoundMode v725 = RoundMode::CAST_ROUND;
      SaturationMode v726 = SaturationMode::ON;
      TCVT(v723, v719, v725, v726);
      // pto: %t__tmp_v528
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v727 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v66);
      // pto: %t__tmp_v528
      ;
      uint64_t v728 = (uint64_t) v57;
      TASSIGN(v727, v728);
      pipe_barrier(PIPE_V);
      TMULS(v727, v723, v91);
      // pto: %t__tmp_v529
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v729 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v66);
      // pto: %t__tmp_v529
      ;
      uint64_t v730 = (uint64_t) v37;
      TASSIGN(v729, v730);
      pipe_barrier(PIPE_V);
      TADD(v729, v713, v727);
      // pto: %t__tmp_v530
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v731 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v66);
      // pto: %t__tmp_v530
      ;
      uint64_t v732 = (uint64_t) v37;
      TASSIGN(v731, v732);
      pipe_barrier(PIPE_V);
      TADDS(v731, v729, v91);
      // pto: %post_ids_inline1876__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v733 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v66);
      // pto: %post_ids_inline1876__ssa_v0
      ;
      uint64_t v734 = (uint64_t) v37;
      TASSIGN(v733, v734);
      pipe_barrier(PIPE_V);
      RoundMode v735 = RoundMode::CAST_ROUND;
      SaturationMode v736 = SaturationMode::ON;
      TCVT(v733, v731, v735, v736);
      // pto: %post_tmp_inline1888__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v737 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v66);
      // pto: %post_tmp_inline1888__ssa_v0
      ;
      uint64_t v738 = (uint64_t) v57;
      TASSIGN(v737, v738);
      // pto: %t__tmp_v531
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v739 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v75);
      // pto: %t__tmp_v531
      ;
      uint64_t v740 = (uint64_t) v56;
      TASSIGN(v739, v740);
      // pto: %post_values_inline1889__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v741 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v66);
      // pto: %post_values_inline1889__ssa_v0
      ;
      uint64_t v742 = (uint64_t) v55;
      TASSIGN(v741, v742);
      pipe_barrier(PIPE_V);
      TGATHER(v741, v739, v733, v737);
      // pto: %t__tmp_v532
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v743 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v66);
      // pto: %t__tmp_v532
      ;
      uint64_t v744 = (uint64_t) v37;
      TASSIGN(v743, v744);
      pipe_barrier(PIPE_V);
      TMULS(v743, v741, v89);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      // pto: %144
      ;
      int64_t v745 = (int64_t) ((uint64_t) v603 * (uint64_t) v79);
      v743.SetValidShape(v67, v745);
      // pto: %145
      ;
      bool v746 = v581 == v78;
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
      if (v746) {
        // pto: %147
        ;
        int64_t v747 = (int64_t) ((uint64_t) v601 * (uint64_t) v79);
        // pto: %148
        ;
        int64_t v748 = v747 < v78 ? v78 : v747;
        // pto: %post_flat_inline1720__ssa_v0_pview
        ;
        const int64_t v749 = 0;
        // pto: %post_flat_inline1720__ssa_v0_pview
        ;
        __gm__ float* v750 = PTOAS__GLOBAL_TENSOR_DATA(v166);
        // pto: %post_flat_inline1720__ssa_v0_pview
        ;
        const int64_t v751 = 1;
        // pto: %post_flat_inline1720__ssa_v0_pview
        ;
        const int64_t v752 = 1;
        // pto: %post_flat_inline1720__ssa_v0_pview
        ;
        const int64_t v753 = 1;
        // pto: %post_flat_inline1720__ssa_v0_pview
        ;
        int64_t v754 = v67 * v158;
        // pto: %post_flat_inline1720__ssa_v0_pview
        ;
        int64_t v755 = v753 * v754;
        // pto: %post_flat_inline1720__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, -1> v756 = pto::Shape<1, 1, 1, 1, -1>(v751, v752, v753, v67, v745);
        // pto: %post_flat_inline1720__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v757 = pto::Stride<-1, -1, -1, -1, -1>(v752 * v755, v755, v754, v158, v67);
        // pto: %post_flat_inline1720__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v758 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v750 + (v749 + v78 * v158 + v748 * v67), v756, v757);
        TSTORE(v758, v743);
      };
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      // pto: %149
      ;
      int64_t v759 = v603 / v80;
      // pto: %150
      ;
      int64_t v760 = (int64_t) ((uint64_t) v759 * (uint64_t) v80);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      for (int64_t v761 = v78; v761 < v760; v761 += v80) {
        // pto: %151
        ;
        int64_t v762 = (int64_t) ((uint64_t) v601 + (uint64_t) v761);
        // pto: %152
        ;
        int64_t v763 = (int64_t) ((uint64_t) v761 + (uint64_t) v67);
        // pto: %153
        ;
        int64_t v764 = (int64_t) ((uint64_t) v601 + (uint64_t) v763);
        // pto: %flat_offset_mul
        ;
        int64_t v765 = (int64_t) ((uint64_t) v761 * (uint64_t) v71);
        // pto: %t__tmp_v533
        ;
        float v766 = v705.GetValue(v765);
        // pto: %154
        ;
        float v767 = v766 + v84;
        // pto: %156, %t__tmp_v534
        ;
        float v768 = v705.GetValue((int64_t) ((uint64_t) v765 + (uint64_t) v67));
        // pto: %157
        ;
        float v769 = v768 + v84;
        // pto: %159, %t__tmp_v535
        ;
        float v770 = v705.GetValue((int64_t) ((uint64_t) v765 + (uint64_t) v80));
        // pto: %160
        ;
        float v771 = v770 + v84;
        // pto: %162, %t__tmp_v536
        ;
        float v772 = v705.GetValue((int64_t) ((uint64_t) v765 + (uint64_t) v70));
        // pto: %163
        ;
        float v773 = v772 + v84;
        // pto: %t__tmp_v537
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v774 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %t__tmp_v537
        ;
        uint64_t v775 = (uint64_t) v58;
        TASSIGN(v774, v775);
        // pto: %164
        ;
        int64_t v776 = v762 < v78 ? v78 : v762;
        // pto: %165
        ;
        __gm__ bfloat16_t* v777 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %165
        ;
        const int64_t v778 = 0;
        // pto: %165
        ;
        const int64_t v779 = 16384;
        // pto: %165
        ;
        pto::Shape<1, 1, 1, 1, 4096> v780 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %165
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v781 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %165
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v782 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v777 + (v778 + v776 * v779), v780, v781);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v774, v782);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %t__tmp_v538
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v783 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %t__tmp_v538
        ;
        uint64_t v784 = (uint64_t) v59;
        TASSIGN(v783, v784);
        // pto: %167
        ;
        __gm__ bfloat16_t* v785 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %167
        ;
        const int64_t v786 = 4096;
        // pto: %167
        ;
        const int64_t v787 = 16384;
        // pto: %167
        ;
        pto::Shape<1, 1, 1, 1, 4096> v788 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %167
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v789 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %167
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v790 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v785 + (v786 + v776 * v787), v788, v789);
        TLOAD(v783, v790);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %t__tmp_v539
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v791 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %t__tmp_v539
        ;
        uint64_t v792 = (uint64_t) v60;
        TASSIGN(v791, v792);
        // pto: %169
        ;
        __gm__ bfloat16_t* v793 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %169
        ;
        const int64_t v794 = 8192;
        // pto: %169
        ;
        const int64_t v795 = 16384;
        // pto: %169
        ;
        pto::Shape<1, 1, 1, 1, 4096> v796 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %169
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v797 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %169
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v798 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v793 + (v794 + v776 * v795), v796, v797);
        TLOAD(v791, v798);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %t__tmp_v540
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v799 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %t__tmp_v540
        ;
        uint64_t v800 = (uint64_t) v61;
        TASSIGN(v799, v800);
        // pto: %171
        ;
        __gm__ bfloat16_t* v801 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %171
        ;
        const int64_t v802 = 12288;
        // pto: %171
        ;
        const int64_t v803 = 16384;
        // pto: %171
        ;
        pto::Shape<1, 1, 1, 1, 4096> v804 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %171
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v805 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %171
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v806 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v801 + (v802 + v776 * v803), v804, v805);
        TLOAD(v799, v806);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %174
        ;
        int64_t v807 = (int64_t) ((uint64_t) v763 * (uint64_t) v71);
        // pto: %172
        ;
        float v808 = v705.GetValue(v807);
        // pto: %176
        ;
        float v809 = v808 + v84;
        // pto: %180, %177
        ;
        float v810 = v705.GetValue((int64_t) ((uint64_t) v807 + (uint64_t) v67));
        // pto: %181
        ;
        float v811 = v810 + v84;
        // pto: %185, %182
        ;
        float v812 = v705.GetValue((int64_t) ((uint64_t) v807 + (uint64_t) v80));
        // pto: %186
        ;
        float v813 = v812 + v84;
        // pto: %190, %187
        ;
        float v814 = v705.GetValue((int64_t) ((uint64_t) v807 + (uint64_t) v70));
        // pto: %191
        ;
        float v815 = v814 + v84;
        // pto: %14
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v816 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %14
        ;
        uint64_t v817 = (uint64_t) v35;
        TASSIGN(v816, v817);
        // pto: %192
        ;
        int64_t v818 = v764 < v78 ? v78 : v764;
        // pto: %193
        ;
        __gm__ bfloat16_t* v819 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %193
        ;
        const int64_t v820 = 0;
        // pto: %193
        ;
        const int64_t v821 = 16384;
        // pto: %193
        ;
        pto::Shape<1, 1, 1, 1, 4096> v822 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %193
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v823 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %193
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v824 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v819 + (v820 + v818 * v821), v822, v823);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TLOAD(v816, v824);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %15
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v825 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %15
        ;
        uint64_t v826 = (uint64_t) v62;
        TASSIGN(v825, v826);
        // pto: %195
        ;
        __gm__ bfloat16_t* v827 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %195
        ;
        const int64_t v828 = 4096;
        // pto: %195
        ;
        const int64_t v829 = 16384;
        // pto: %195
        ;
        pto::Shape<1, 1, 1, 1, 4096> v830 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %195
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v831 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %195
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v832 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v827 + (v828 + v818 * v829), v830, v831);
        TLOAD(v825, v832);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %16
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v833 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %16
        ;
        uint64_t v834 = (uint64_t) v63;
        TASSIGN(v833, v834);
        // pto: %197
        ;
        __gm__ bfloat16_t* v835 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %197
        ;
        const int64_t v836 = 8192;
        // pto: %197
        ;
        const int64_t v837 = 16384;
        // pto: %197
        ;
        pto::Shape<1, 1, 1, 1, 4096> v838 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %197
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v839 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %197
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v840 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v835 + (v836 + v818 * v837), v838, v839);
        TLOAD(v833, v840);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        // pto: %17
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v841 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %17
        ;
        uint64_t v842 = (uint64_t) v64;
        TASSIGN(v841, v842);
        // pto: %199
        ;
        __gm__ bfloat16_t* v843 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %199
        ;
        const int64_t v844 = 12288;
        // pto: %199
        ;
        const int64_t v845 = 16384;
        // pto: %199
        ;
        pto::Shape<1, 1, 1, 1, 4096> v846 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %199
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v847 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %199
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v848 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v843 + (v844 + v818 * v845), v846, v847);
        TLOAD(v841, v848);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        // pto: %scalar_x0_inline1864__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v849 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %scalar_x0_inline1864__ssa_v0
        ;
        uint64_t v850 = (uint64_t) v37;
        TASSIGN(v849, v850);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        RoundMode v851 = RoundMode::CAST_ROUND;
        SaturationMode v852 = SaturationMode::ON;
        TCVT(v849, v774, v851, v852);
        // pto: %scalar_y0_inline1587__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v853 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %scalar_y0_inline1587__ssa_v0
        ;
        uint64_t v854 = (uint64_t) v37;
        TASSIGN(v853, v854);
        pipe_barrier(PIPE_V);
        TMULS(v853, v849, v767);
        // pto: %scalar_x1_inline1586__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v855 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %scalar_x1_inline1586__ssa_v0
        ;
        uint64_t v856 = (uint64_t) v58;
        TASSIGN(v855, v856);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        RoundMode v857 = RoundMode::CAST_ROUND;
        SaturationMode v858 = SaturationMode::ON;
        TCVT(v855, v783, v857, v858);
        // pto: %scalar_y1_inline1767__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v859 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %scalar_y1_inline1767__ssa_v0
        ;
        uint64_t v860 = (uint64_t) v58;
        TASSIGN(v859, v860);
        pipe_barrier(PIPE_V);
        TMULS(v859, v855, v769);
        // pto: %scalar_sum01_inline1600__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v861 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %scalar_sum01_inline1600__ssa_v0
        ;
        uint64_t v862 = (uint64_t) v37;
        TASSIGN(v861, v862);
        pipe_barrier(PIPE_V);
        TADD(v861, v853, v859);
        // pto: %scalar_x2_inline1585__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v863 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %scalar_x2_inline1585__ssa_v0
        ;
        uint64_t v864 = (uint64_t) v58;
        TASSIGN(v863, v864);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        RoundMode v865 = RoundMode::CAST_ROUND;
        SaturationMode v866 = SaturationMode::ON;
        TCVT(v863, v791, v865, v866);
        // pto: %scalar_y2_inline1584__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v867 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %scalar_y2_inline1584__ssa_v0
        ;
        uint64_t v868 = (uint64_t) v58;
        TASSIGN(v867, v868);
        pipe_barrier(PIPE_V);
        TMULS(v867, v863, v771);
        // pto: %scalar_x3_inline1583__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v869 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %scalar_x3_inline1583__ssa_v0
        ;
        uint64_t v870 = (uint64_t) v59;
        TASSIGN(v869, v870);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        RoundMode v871 = RoundMode::CAST_ROUND;
        SaturationMode v872 = SaturationMode::ON;
        TCVT(v869, v799, v871, v872);
        // pto: %scalar_y3_inline1582__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v873 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %scalar_y3_inline1582__ssa_v0
        ;
        uint64_t v874 = (uint64_t) v59;
        TASSIGN(v873, v874);
        pipe_barrier(PIPE_V);
        TMULS(v873, v869, v773);
        // pto: %t__tmp_v541
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v875 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %t__tmp_v541
        ;
        uint64_t v876 = (uint64_t) v58;
        TASSIGN(v875, v876);
        pipe_barrier(PIPE_V);
        TADD(v875, v867, v873);
        // pto: %scalar_result_inline1836__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v877 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %scalar_result_inline1836__ssa_v0
        ;
        uint64_t v878 = (uint64_t) v37;
        TASSIGN(v877, v878);
        pipe_barrier(PIPE_V);
        TADD(v877, v861, v875);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %scalar_bf16_inline1580__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v879 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %scalar_bf16_inline1580__ssa_v0
        ;
        uint64_t v880 = (uint64_t) v37;
        TASSIGN(v879, v880);
        pipe_barrier(PIPE_V);
        RoundMode v881 = RoundMode::CAST_RINT;
        SaturationMode v882 = SaturationMode::ON;
        TCVT(v879, v877, v881, v882);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        // pto: %mixed_view_inline1624__ssa_v0_pview
        ;
        __gm__ bfloat16_t* v883 = PTOAS__GLOBAL_TENSOR_DATA(v175);
        // pto: %mixed_view_inline1624__ssa_v0_pview
        ;
        const int64_t v884 = 0;
        // pto: %mixed_view_inline1624__ssa_v0_pview
        ;
        const int64_t v885 = 4096;
        // pto: %mixed_view_inline1624__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, 4096> v886 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %mixed_view_inline1624__ssa_v0_pview
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v887 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %mixed_view_inline1624__ssa_v0_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v888 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v883 + (v884 + v776 * v885), v886, v887);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v888, v879);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        // pto: %18
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v889 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %18
        ;
        uint64_t v890 = (uint64_t) v65;
        TASSIGN(v889, v890);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        RoundMode v891 = RoundMode::CAST_ROUND;
        SaturationMode v892 = SaturationMode::ON;
        TCVT(v889, v816, v891, v892);
        // pto: %19
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v893 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %19
        ;
        uint64_t v894 = (uint64_t) v65;
        TASSIGN(v893, v894);
        pipe_barrier(PIPE_V);
        TMULS(v893, v889, v809);
        // pto: %20
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v895 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %20
        ;
        uint64_t v896 = (uint64_t) v35;
        TASSIGN(v895, v896);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        RoundMode v897 = RoundMode::CAST_ROUND;
        SaturationMode v898 = SaturationMode::ON;
        TCVT(v895, v825, v897, v898);
        // pto: %21
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v899 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %21
        ;
        uint64_t v900 = (uint64_t) v35;
        TASSIGN(v899, v900);
        pipe_barrier(PIPE_V);
        TMULS(v899, v895, v811);
        // pto: %22
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v901 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %22
        ;
        uint64_t v902 = (uint64_t) v65;
        TASSIGN(v901, v902);
        pipe_barrier(PIPE_V);
        TADD(v901, v893, v899);
        // pto: %23
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v903 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %23
        ;
        uint64_t v904 = (uint64_t) v35;
        TASSIGN(v903, v904);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        RoundMode v905 = RoundMode::CAST_ROUND;
        SaturationMode v906 = SaturationMode::ON;
        TCVT(v903, v833, v905, v906);
        // pto: %24
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v907 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %24
        ;
        uint64_t v908 = (uint64_t) v35;
        TASSIGN(v907, v908);
        pipe_barrier(PIPE_V);
        TMULS(v907, v903, v813);
        // pto: %25
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v909 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %25
        ;
        uint64_t v910 = (uint64_t) v62;
        TASSIGN(v909, v910);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        RoundMode v911 = RoundMode::CAST_ROUND;
        SaturationMode v912 = SaturationMode::ON;
        TCVT(v909, v841, v911, v912);
        // pto: %26
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v913 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %26
        ;
        uint64_t v914 = (uint64_t) v62;
        TASSIGN(v913, v914);
        pipe_barrier(PIPE_V);
        TMULS(v913, v909, v815);
        // pto: %27
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v915 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %27
        ;
        uint64_t v916 = (uint64_t) v35;
        TASSIGN(v915, v916);
        pipe_barrier(PIPE_V);
        TADD(v915, v907, v913);
        // pto: %28
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v917 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %28
        ;
        uint64_t v918 = (uint64_t) v65;
        TASSIGN(v917, v918);
        pipe_barrier(PIPE_V);
        TADD(v917, v901, v915);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        // pto: %29
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v919 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %29
        ;
        uint64_t v920 = (uint64_t) v65;
        TASSIGN(v919, v920);
        pipe_barrier(PIPE_V);
        RoundMode v921 = RoundMode::CAST_RINT;
        SaturationMode v922 = SaturationMode::ON;
        TCVT(v919, v917, v921, v922);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %203
        ;
        __gm__ bfloat16_t* v923 = PTOAS__GLOBAL_TENSOR_DATA(v175);
        // pto: %203
        ;
        const int64_t v924 = 0;
        // pto: %203
        ;
        const int64_t v925 = 4096;
        // pto: %203
        ;
        pto::Shape<1, 1, 1, 1, 4096> v926 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %203
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v927 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %203
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v928 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v923 + (v924 + v818 * v925), v926, v927);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v928, v919);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
      };
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_S, PIPE_V, EVENT_ID1);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      // pto: %204, %205
      ;
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      if ((int64_t) ((uint64_t) v603 - (uint64_t) v760) == v67) {
        // pto: %206
        ;
        int64_t v929 = (int64_t) ((uint64_t) v601 + (uint64_t) v760);
        int64_t v930 = (int64_t) ((uint64_t) v759 * (uint64_t) v72);
        // pto: %207
        ;
        float v931 = v705.GetValue(v930);
        // pto: %210
        ;
        float v932 = v931 + v84;
        // pto: %213, %211
        ;
        float v933 = v705.GetValue((int64_t) ((uint64_t) v930 + (uint64_t) v67));
        // pto: %214
        ;
        float v934 = v933 + v84;
        // pto: %217, %215
        ;
        float v935 = v705.GetValue((int64_t) ((uint64_t) v930 + (uint64_t) v80));
        // pto: %218
        ;
        float v936 = v935 + v84;
        // pto: %221, %219
        ;
        float v937 = v705.GetValue((int64_t) ((uint64_t) v930 + (uint64_t) v70));
        // pto: %222
        ;
        float v938 = v937 + v84;
        // pto: %30
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v939 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %30
        ;
        uint64_t v940 = (uint64_t) v58;
        TASSIGN(v939, v940);
        // pto: %223
        ;
        int64_t v941 = v929 < v78 ? v78 : v929;
        // pto: %224
        ;
        __gm__ bfloat16_t* v942 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %224
        ;
        const int64_t v943 = 0;
        // pto: %224
        ;
        const int64_t v944 = 16384;
        // pto: %224
        ;
        pto::Shape<1, 1, 1, 1, 4096> v945 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %224
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v946 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %224
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v947 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v942 + (v943 + v941 * v944), v945, v946);
        TLOAD(v939, v947);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %31
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v948 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %31
        ;
        uint64_t v949 = (uint64_t) v37;
        TASSIGN(v948, v949);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v950 = RoundMode::CAST_ROUND;
        SaturationMode v951 = SaturationMode::ON;
        TCVT(v948, v939, v950, v951);
        // pto: %32
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v952 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %32
        ;
        uint64_t v953 = (uint64_t) v37;
        TASSIGN(v952, v953);
        pipe_barrier(PIPE_V);
        TMULS(v952, v948, v932);
        // pto: %33
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v954 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %33
        ;
        uint64_t v955 = (uint64_t) v59;
        TASSIGN(v954, v955);
        // pto: %226
        ;
        __gm__ bfloat16_t* v956 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %226
        ;
        const int64_t v957 = 4096;
        // pto: %226
        ;
        const int64_t v958 = 16384;
        // pto: %226
        ;
        pto::Shape<1, 1, 1, 1, 4096> v959 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %226
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v960 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %226
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v961 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v956 + (v957 + v941 * v958), v959, v960);
        TLOAD(v954, v961);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %34
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v962 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %34
        ;
        uint64_t v963 = (uint64_t) v58;
        TASSIGN(v962, v963);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v964 = RoundMode::CAST_ROUND;
        SaturationMode v965 = SaturationMode::ON;
        TCVT(v962, v954, v964, v965);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %35
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v966 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %35
        ;
        uint64_t v967 = (uint64_t) v58;
        TASSIGN(v966, v967);
        pipe_barrier(PIPE_V);
        TMULS(v966, v962, v934);
        // pto: %36
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v968 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %36
        ;
        uint64_t v969 = (uint64_t) v37;
        TASSIGN(v968, v969);
        pipe_barrier(PIPE_V);
        TADD(v968, v952, v966);
        // pto: %37
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v970 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %37
        ;
        uint64_t v971 = (uint64_t) v59;
        TASSIGN(v970, v971);
        // pto: %228
        ;
        __gm__ bfloat16_t* v972 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %228
        ;
        const int64_t v973 = 8192;
        // pto: %228
        ;
        const int64_t v974 = 16384;
        // pto: %228
        ;
        pto::Shape<1, 1, 1, 1, 4096> v975 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %228
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v976 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %228
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v977 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v972 + (v973 + v941 * v974), v975, v976);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v970, v977);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %38
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v978 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %38
        ;
        uint64_t v979 = (uint64_t) v58;
        TASSIGN(v978, v979);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v980 = RoundMode::CAST_ROUND;
        SaturationMode v981 = SaturationMode::ON;
        TCVT(v978, v970, v980, v981);
        // pto: %39
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v982 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %39
        ;
        uint64_t v983 = (uint64_t) v58;
        TASSIGN(v982, v983);
        pipe_barrier(PIPE_V);
        TMULS(v982, v978, v936);
        // pto: %40
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v984 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %40
        ;
        uint64_t v985 = (uint64_t) v65;
        TASSIGN(v984, v985);
        // pto: %230
        ;
        __gm__ bfloat16_t* v986 = PTOAS__GLOBAL_TENSOR_DATA(v114);
        // pto: %230
        ;
        const int64_t v987 = 12288;
        // pto: %230
        ;
        const int64_t v988 = 16384;
        // pto: %230
        ;
        pto::Shape<1, 1, 1, 1, 4096> v989 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %230
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v990 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %230
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v991 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v986 + (v987 + v941 * v988), v989, v990);
        TLOAD(v984, v991);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %41
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v992 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %41
        ;
        uint64_t v993 = (uint64_t) v59;
        TASSIGN(v992, v993);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v994 = RoundMode::CAST_ROUND;
        SaturationMode v995 = SaturationMode::ON;
        TCVT(v992, v984, v994, v995);
        // pto: %42
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v996 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %42
        ;
        uint64_t v997 = (uint64_t) v59;
        TASSIGN(v996, v997);
        pipe_barrier(PIPE_V);
        TMULS(v996, v992, v938);
        // pto: %43
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v998 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %43
        ;
        uint64_t v999 = (uint64_t) v58;
        TASSIGN(v998, v999);
        pipe_barrier(PIPE_V);
        TADD(v998, v982, v996);
        // pto: %44
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1000 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %44
        ;
        uint64_t v1001 = (uint64_t) v37;
        TASSIGN(v1000, v1001);
        pipe_barrier(PIPE_V);
        TADD(v1000, v968, v998);
        // pto: %45
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1002 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v73);
        // pto: %45
        ;
        uint64_t v1003 = (uint64_t) v37;
        TASSIGN(v1002, v1003);
        pipe_barrier(PIPE_V);
        RoundMode v1004 = RoundMode::CAST_RINT;
        SaturationMode v1005 = SaturationMode::ON;
        TCVT(v1002, v1000, v1004, v1005);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %233
        ;
        __gm__ bfloat16_t* v1006 = PTOAS__GLOBAL_TENSOR_DATA(v175);
        // pto: %233
        ;
        const int64_t v1007 = 0;
        // pto: %233
        ;
        const int64_t v1008 = 4096;
        // pto: %233
        ;
        pto::Shape<1, 1, 1, 1, 4096> v1009 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %233
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v1010 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %233
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v1011 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v1006 + (v1007 + v941 * v1008), v1009, v1010);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v1011, v1002);
      };
      set_flag(PIPE_S, PIPE_V, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}