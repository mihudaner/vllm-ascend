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

AICORE void hc_large144_aic(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ bfloat16_t* v12, __gm__ float* v13, int64_t v14, int32_t v15, int64_t v16, int64_t v17, int64_t v18, int64_t v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int64_t v28, int64_t v29, int64_t v30, int32_t v31, int32_t v32) {
  using T = float;

  #if defined(__DAV_CUBE__)
  const int64_t v33 = 384;
  const int32_t v34 = 0;
  // pto: %c0_i64
  const int64_t v35 = 0;
  // pto: %c196608_i64
  const int64_t v36 = 196608;
  // pto: %c262144_i64
  const int64_t v37 = 262144;
  // pto: %c24576_i64
  const int64_t v38 = 24576;
  // pto: %c16384_i64
  const int64_t v39 = 16384;
  // pto: %c32768_i64
  const int64_t v40 = 32768;
  // pto: %c49152_i64
  const int64_t v41 = 49152;
  // pto: %c32_index
  const int64_t v42 = 32;
  // pto: %c1_index
  const int64_t v43 = 1;
  // pto: %c24_index
  const int64_t v44 = 24;
  // pto: %c16384_index
  const int64_t v45 = 16384;
  // pto: %c8_index
  const int64_t v46 = 8;
  // pto: %c48_index
  const int64_t v47 = 48;
  // pto: %c2048_index
  const int64_t v48 = 2048;
  // pto: %c0_index
  const int64_t v49 = 0;
  // pto: %c4_index
  const int64_t v50 = 4;
  // pto: %c2_index
  const int64_t v51 = 2;
  // pto: %c512_index
  const int64_t v52 = 512;
  // pto: %c256_index
  const int64_t v53 = 256;
  // pto: %c128_index
  const int64_t v54 = 128;
  // pto: %cn128_index
  const int64_t v55 = -128;
  // pto: %63
  int64_t v56 = (int64_t) ((uint64_t) v29 * (uint64_t) v46);
  // pto: %mixes_partials_inline1431__ssa_v0_view
  const int64_t v57 = 1;
  // pto: %mixes_partials_inline1431__ssa_v0_view
  const int64_t v58 = 1;
  // pto: %mixes_partials_inline1431__ssa_v0_view
  const int64_t v59 = 1;
  // pto: %mixes_partials_inline1431__ssa_v0_view
  int64_t v60 = v56 * v42;
  // pto: %mixes_partials_inline1431__ssa_v0_view
  int64_t v61 = v59 * v60;
  // pto: %mixes_partials_inline1431__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v62 = pto::Shape<1, 1, 1, -1, -1>(v57, v58, v59, v56, v42);
  // pto: %mixes_partials_inline1431__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v63 = pto::Stride<-1, -1, -1, -1, -1>(v58 * v61, v61, v60, v42, v43);
  // pto: %mixes_partials_inline1431__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v64 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v62, v63);
  // pto: %hc_fn__ssa_v0_view
  const int64_t v65 = 1;
  // pto: %hc_fn__ssa_v0_view
  const int64_t v66 = 1;
  // pto: %hc_fn__ssa_v0_view
  const int64_t v67 = 1;
  // pto: %hc_fn__ssa_v0_view
  int64_t v68 = v44 * v45;
  // pto: %hc_fn__ssa_v0_view
  int64_t v69 = v67 * v68;
  // pto: %hc_fn__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v70 = pto::Shape<1, 1, 1, -1, -1>(v65, v66, v67, v44, v45);
  // pto: %hc_fn__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v71 = pto::Stride<-1, -1, -1, -1, -1>(v66 * v69, v69, v68, v45, v43);
  // pto: %hc_fn__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v72 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v70, v71);
  // pto: %mid_stream_product_inline1402__phi_v5
  Tile<TileType::Acc, float, 48, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v73 = Tile<TileType::Acc, float, 48, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v47, v44);
  // pto: %mid_stream_product_inline1402__phi_v5
  uint64_t v74 = (uint64_t) v35;
  TASSIGN(v73, v74);
  const int32_t v75 = 0;
  auto v76 = TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>(v13, v75, v34);
  // pto: %worker_inline1352__ssa_v0
  // pto: %linear_workers_inline1471__ssa_v0_idx
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
  for (int64_t v77 = (int64_t) v31; v77 < v14; v77 += (int64_t) v15) {
    // pto: %20
    ;
    int64_t v78 = v77 % v46;
    // pto: %21
    ;
    int64_t v79 = (int64_t) ((uint64_t) v78 * (uint64_t) v48);
    // pto: %22, %23, %18, %19
    ;
    int64_t v80 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v78 * (uint64_t) v16) + (uint64_t) ((int64_t) (uint64_t) (v77 / v46) * (uint64_t) v47));
    // pto: %mid_stream_product_inline1402__tile
    ;
    Tile<TileType::Acc, float, 48, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v81 = Tile<TileType::Acc, float, 48, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v47, v42);
    // pto: %mid_stream_product_inline1402__tile
    ;
    uint64_t v82 = (uint64_t) v35;
    TASSIGN(v81, v82);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int64_t v83 = v49; v83 < v50; v83 += v51) {
      // pto: %24
      ;
      int64_t v84 = (int64_t) ((uint64_t) v83 * (uint64_t) v52);
      // pto: %25
      ;
      int64_t v85 = (int64_t) ((uint64_t) v79 + (uint64_t) v84);
      // pto: %28, %27
      ;
      int64_t v86 = (int64_t) ((uint64_t) v79 + (uint64_t) ((int64_t) (uint64_t) v84 + (uint64_t) v52));
      // pto: %mid_weight_chunk_inline1332__tile
      ;
      Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v87 = Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v44, v52);
      // pto: %mid_weight_chunk_inline1332__tile
      ;
      uint64_t v88 = (uint64_t) v36;
      TASSIGN(v87, v88);
      // pto: %29
      ;
      int64_t v89 = v85 < v49 ? v49 : v85;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v90 = 0;
      // pto: %hc_fn__ssa_v0_pview
      ;
      __gm__ float* v91 = PTOAS__GLOBAL_TENSOR_DATA(v72);
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v92 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v93 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v94 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      int64_t v95 = v44 * v45;
      // pto: %hc_fn__ssa_v0_pview
      ;
      int64_t v96 = v94 * v95;
      // pto: %hc_fn__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 24, -1> v97 = pto::Shape<1, 1, 1, 24, -1>(v92, v93, v94, v44, v52);
      // pto: %hc_fn__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v98 = pto::Stride<-1, -1, -1, -1, -1>(v93 * v96, v96, v95, v45, v43);
      // pto: %hc_fn__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v99 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v91 + (v90 + v49 * v45 + v89 * v43), v97, v98);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v87, v99);
      // pto: %0
      ;
      Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v100 = Tile<TileType::Mat, float, 32, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v44, v52);
      // pto: %0
      ;
      uint64_t v101 = (uint64_t) v37;
      TASSIGN(v100, v101);
      // pto: %30
      ;
      int64_t v102 = v86 < v49 ? v49 : v86;
      // pto: %31
      ;
      const int64_t v103 = 0;
      // pto: %31
      ;
      __gm__ float* v104 = PTOAS__GLOBAL_TENSOR_DATA(v72);
      // pto: %31
      ;
      const int64_t v105 = 1;
      // pto: %31
      ;
      const int64_t v106 = 1;
      // pto: %31
      ;
      const int64_t v107 = 1;
      // pto: %31
      ;
      int64_t v108 = v44 * v45;
      // pto: %31
      ;
      int64_t v109 = v107 * v108;
      // pto: %31
      ;
      pto::Shape<1, 1, 1, 24, -1> v110 = pto::Shape<1, 1, 1, 24, -1>(v105, v106, v107, v44, v52);
      // pto: %31
      ;
      pto::Stride<-1, -1, -1, -1, -1> v111 = pto::Stride<-1, -1, -1, -1, -1>(v106 * v109, v109, v108, v45, v43);
      // pto: %31
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v112 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v104 + (v103 + v49 * v45 + v102 * v43), v110, v111);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TLOAD(v100, v112);
      // pto: %mid_gathered_chunk_inline1319__tile
      ;
      const int32_t v113 = 48;
      // pto: %mid_gathered_chunk_inline1319__tile
      ;
      const int32_t v114 = 512;
      // pto: %mid_gathered_chunk_inline1319__tile
      ;
      Tile<TileType::Mat, float, 48, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v115 = Tile<TileType::Mat, float, 48, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v113, v114);
      pipe_barrier(PIPE_MTE2);
      TPOP<TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>, Tile<TileType::Mat, float, 48, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v76, v115);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %32
      ;
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      if (v83 == v49) {
        // pto: %mid_weight_chunk_inline1332__tile_t
        ;
        Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v116 = Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v52, v44);
        // pto: %mid_weight_chunk_inline1332__tile_t
        ;
        uint64_t v117 = (uint64_t) v36;
        TASSIGN(v116, v117);
        // pto: %mid_stream_product_inline1402__tile_l0_init_storage
        ;
        Tile<TileType::Acc, float, 48, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v118 = Tile<TileType::Acc, float, 48, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v47, v42);
        // pto: %mid_stream_product_inline1402__tile_l0_init_storage
        ;
        uint64_t v119 = (uint64_t) v35;
        TASSIGN(v118, v119);
        v118.SetValidShape(v47, v44);
        for (int64_t v120 = v49; v120 < v52; v120 += v53) {
          // pto: %mid_stream_product_inline1402__tile_l0_a
          ;
          Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v121 = Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v47, v54);
          // pto: %mid_stream_product_inline1402__tile_l0_a
          ;
          uint64_t v122 = (uint64_t) v35;
          TASSIGN(v121, v122);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          pipe_barrier(PIPE_MTE1);
          uint16_t v123 = static_cast<uint16_t>(v49);
          uint16_t v124 = static_cast<uint16_t>(v120);
          TEXTRACT(v121, v115, v123, v124);
          // pto: %33
          ;
          int64_t v125 = (int64_t) ((uint64_t) v52 - (uint64_t) v120);
          // pto: %34
          ;
          int64_t v126 = v125 < v54 ? v125 : v54;
          // pto: %mid_stream_product_inline1402__tile_l0_b
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v127 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v126, v44);
          // pto: %mid_stream_product_inline1402__tile_l0_b
          ;
          uint64_t v128 = (uint64_t) v35;
          TASSIGN(v127, v128);
          uint16_t v129 = static_cast<uint16_t>(v120);
          uint16_t v130 = static_cast<uint16_t>(v49);
          TEXTRACT(v127, v116, v129, v130);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          // pto: %1
          ;
          Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v131 = Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v47, v54);
          // pto: %1
          ;
          uint64_t v132 = (uint64_t) v38;
          TASSIGN(v131, v132);
          // pto: %35
          ;
          int64_t v133 = (int64_t) ((uint64_t) v120 + (uint64_t) v54);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          uint16_t v134 = static_cast<uint16_t>(v49);
          uint16_t v135 = static_cast<uint16_t>(v133);
          TEXTRACT(v131, v115, v134, v135);
          int64_t v136 = (int64_t) ((uint64_t) v33 - (uint64_t) v120);
          // pto: %38
          ;
          int64_t v137 = v136 < v49 ? v49 : v136;
          // pto: %39
          ;
          int64_t v138 = v137 < v54 ? v137 : v54;
          // pto: %2
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v139 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v138, v44);
          // pto: %2
          ;
          uint64_t v140 = (uint64_t) v39;
          TASSIGN(v139, v140);
          uint16_t v141 = static_cast<uint16_t>(v133);
          uint16_t v142 = static_cast<uint16_t>(v49);
          TEXTRACT(v139, v116, v141, v142);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          // pto: %41
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          if (v120 == v49) {
            pipe_barrier(PIPE_M);
            TMATMUL(v118, v121, v127);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v118, v118, v121, v127);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          // pto: %42
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          if (v120 == v55) {
            pipe_barrier(PIPE_M);
            TMATMUL(v118, v131, v139);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v118, v118, v131, v139);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        };
      } else {
        // pto: %4
        ;
        Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v143 = Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v52, v44);
        // pto: %4
        ;
        uint64_t v144 = (uint64_t) v36;
        TASSIGN(v143, v144);
        for (int64_t v145 = v49; v145 < v52; v145 += v53) {
          // pto: %5
          ;
          Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v146 = Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v47, v54);
          // pto: %5
          ;
          uint64_t v147 = (uint64_t) v35;
          TASSIGN(v146, v147);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
          uint16_t v148 = static_cast<uint16_t>(v49);
          uint16_t v149 = static_cast<uint16_t>(v145);
          TEXTRACT(v146, v115, v148, v149);
          // pto: %44
          ;
          int64_t v150 = (int64_t) ((uint64_t) v52 - (uint64_t) v145);
          // pto: %45
          ;
          int64_t v151 = v150 < v54 ? v150 : v54;
          // pto: %6
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v152 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v151, v44);
          // pto: %6
          ;
          uint64_t v153 = (uint64_t) v35;
          TASSIGN(v152, v153);
          uint16_t v154 = static_cast<uint16_t>(v145);
          uint16_t v155 = static_cast<uint16_t>(v49);
          TEXTRACT(v152, v143, v154, v155);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
          // pto: %7
          ;
          Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v156 = Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v47, v54);
          // pto: %7
          ;
          uint64_t v157 = (uint64_t) v38;
          TASSIGN(v156, v157);
          // pto: %46
          ;
          int64_t v158 = (int64_t) ((uint64_t) v145 + (uint64_t) v54);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
          uint16_t v159 = static_cast<uint16_t>(v49);
          uint16_t v160 = static_cast<uint16_t>(v158);
          TEXTRACT(v156, v115, v159, v160);
          int64_t v161 = (int64_t) ((uint64_t) v33 - (uint64_t) v145);
          // pto: %49
          ;
          int64_t v162 = v161 < v49 ? v49 : v161;
          // pto: %50
          ;
          int64_t v163 = v162 < v54 ? v162 : v54;
          // pto: %8
          ;
          Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v164 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v163, v44);
          // pto: %8
          ;
          uint64_t v165 = (uint64_t) v39;
          TASSIGN(v164, v165);
          uint16_t v166 = static_cast<uint16_t>(v158);
          uint16_t v167 = static_cast<uint16_t>(v49);
          TEXTRACT(v164, v143, v166, v167);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
          pipe_barrier(PIPE_M);
          TMATMUL_ACC(v81, v81, v146, v152);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
          pipe_barrier(PIPE_M);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
          TMATMUL_ACC(v81, v81, v156, v164);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        };
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TFREE<TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v76);
      // pto: %52
      ;
      const int32_t v168 = 48;
      // pto: %52
      ;
      const int32_t v169 = 512;
      // pto: %52
      ;
      Tile<TileType::Mat, float, 48, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v170 = Tile<TileType::Mat, float, 48, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v168, v169);
      TPOP<TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>, Tile<TileType::Mat, float, 48, 512, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v76, v170);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      // pto: %11
      ;
      Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v171 = Tile<TileType::Mat, float, 512, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v52, v44);
      // pto: %11
      ;
      uint64_t v172 = (uint64_t) v37;
      TASSIGN(v171, v172);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
      for (int64_t v173 = v49; v173 < v52; v173 += v53) {
        // pto: %12
        ;
        Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v174 = Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v47, v54);
        // pto: %12
        ;
        uint64_t v175 = (uint64_t) v35;
        TASSIGN(v174, v175);
        pipe_barrier(PIPE_MTE1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        uint16_t v176 = static_cast<uint16_t>(v49);
        uint16_t v177 = static_cast<uint16_t>(v173);
        TEXTRACT(v174, v170, v176, v177);
        // pto: %54
        ;
        int64_t v178 = (int64_t) ((uint64_t) v52 - (uint64_t) v173);
        // pto: %55
        ;
        int64_t v179 = v178 < v54 ? v178 : v54;
        // pto: %13
        ;
        Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v180 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v179, v44);
        // pto: %13
        ;
        uint64_t v181 = (uint64_t) v40;
        TASSIGN(v180, v181);
        uint16_t v182 = static_cast<uint16_t>(v173);
        uint16_t v183 = static_cast<uint16_t>(v49);
        TEXTRACT(v180, v171, v182, v183);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
        // pto: %14
        ;
        Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v184 = Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v47, v54);
        // pto: %14
        ;
        uint64_t v185 = (uint64_t) v38;
        TASSIGN(v184, v185);
        // pto: %56
        ;
        int64_t v186 = (int64_t) ((uint64_t) v173 + (uint64_t) v54);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
        uint16_t v187 = static_cast<uint16_t>(v49);
        uint16_t v188 = static_cast<uint16_t>(v186);
        TEXTRACT(v184, v170, v187, v188);
        int64_t v189 = (int64_t) ((uint64_t) v33 - (uint64_t) v173);
        // pto: %59
        ;
        int64_t v190 = v189 < v49 ? v49 : v189;
        // pto: %60
        ;
        int64_t v191 = v190 < v54 ? v190 : v54;
        // pto: %15
        ;
        Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v192 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v191, v44);
        // pto: %15
        ;
        uint64_t v193 = (uint64_t) v41;
        TASSIGN(v192, v193);
        uint16_t v194 = static_cast<uint16_t>(v186);
        uint16_t v195 = static_cast<uint16_t>(v49);
        TEXTRACT(v192, v171, v194, v195);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v73, v73, v174, v180);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
        pipe_barrier(PIPE_M);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
        TMATMUL_ACC(v73, v73, v184, v192);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      TFREE<TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v76);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    // pto: %62
    ;
    int64_t v196 = v80 < v49 ? v49 : v80;
    // pto: %mixes_partials_inline1431__iter_v1_pview
    ;
    __gm__ float* v197 = PTOAS__GLOBAL_TENSOR_DATA(v64);
    // pto: %mixes_partials_inline1431__iter_v1_pview
    ;
    const int64_t v198 = 0;
    // pto: %mixes_partials_inline1431__iter_v1_pview
    ;
    const int64_t v199 = 32;
    // pto: %mixes_partials_inline1431__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 48, 32> v200 = pto::Shape<1, 1, 1, 48, 32>();
    // pto: %mixes_partials_inline1431__iter_v1_pview
    ;
    pto::Stride<1536, 1536, 1536, 32, 1> v201 = pto::Stride<1536, 1536, 1536, 32, 1>();
    // pto: %mixes_partials_inline1431__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 48, 32>, pto::Stride<1536, 1536, 1536, 32, 1>, pto::Layout::ND> v202 = GlobalTensor<float, pto::Shape<1, 1, 1, 48, 32>, pto::Stride<1536, 1536, 1536, 32, 1>, pto::Layout::ND>(v197 + (v198 + v196 * v199), v200, v201);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v202, v81);
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

AICORE void hc_large144_aiv(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ bfloat16_t* v12, __gm__ float* v13, int64_t v14, int32_t v15, int64_t v16, int64_t v17, int64_t v18, int64_t v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int64_t v28, int64_t v29, int64_t v30, int32_t v31, int32_t v32, int32_t v33) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int64_t v34 = 12800;
  const int64_t v35 = 8704;
  const int64_t v36 = 4608;
  // pto: %c131200_i64
  const int64_t v37 = 131200;
  // pto: %c82048_i64
  const int64_t v38 = 82048;
  // pto: %c0_i64
  const int64_t v39 = 0;
  // pto: %c24576_i64
  const int64_t v40 = 24576;
  // pto: %c131296_i64
  const int64_t v41 = 131296;
  // pto: %c25600_i64
  const int64_t v42 = 25600;
  // pto: %c25856_i64
  const int64_t v43 = 25856;
  const int64_t v44 = 25664;
  const int64_t v45 = 24608;
  // pto: %c25088_i64
  const int64_t v46 = 25088;
  // pto: %c25344_i64
  const int64_t v47 = 25344;
  // pto: %c26368_i64
  const int64_t v48 = 26368;
  // pto: %c26880_i64
  const int64_t v49 = 26880;
  // pto: %c27392_i64
  const int64_t v50 = 27392;
  // pto: %c27904_i64
  const int64_t v51 = 27904;
  // pto: %c28416_i64
  const int64_t v52 = 28416;
  // pto: %c28928_i64
  const int64_t v53 = 28928;
  // pto: %c26400_i64
  const int64_t v54 = 26400;
  // pto: %c26432_i64
  const int64_t v55 = 26432;
  // pto: %c26464_i64
  const int64_t v56 = 26464;
  const int64_t v57 = 25632;
  // pto: %c24832_i64
  const int64_t v58 = 24832;
  // pto: %c24704_i64
  const int64_t v59 = 24704;
  // pto: %c73728_i64
  const int64_t v60 = 73728;
  // pto: %c73760_i64
  const int64_t v61 = 73760;
  // pto: %c73792_i64
  const int64_t v62 = 73792;
  const int64_t v63 = 25696;
  // pto: %c73824_i64
  const int64_t v64 = 73824;
  // pto: %c40960_i64
  const int64_t v65 = 40960;
  // pto: %c57344_i64
  const int64_t v66 = 57344;
  // pto: %c16384_i64
  const int64_t v67 = 16384;
  // pto: %c114816_i64
  const int64_t v68 = 114816;
  // pto: %c73856_i64
  const int64_t v69 = 73856;
  // pto: %c98432_i64
  const int64_t v70 = 98432;
  // pto: %c32_index
  const int64_t v71 = 32;
  // pto: %c1_index
  const int64_t v72 = 1;
  // pto: %c24_index
  const int64_t v73 = 24;
  // pto: %c16384_index
  const int64_t v74 = 16384;
  // pto: %c3_index
  const int64_t v75 = 3;
  // pto: %c8_index
  const int64_t v76 = 8;
  // pto: %c16_index
  const int64_t v77 = 16;
  // pto: %c4096_index
  const int64_t v78 = 4096;
  // pto: %c0_i32
  const int32_t v79 = 0;
  // pto: %c48_index
  const int64_t v80 = 48;
  // pto: %c2048_index
  const int64_t v81 = 2048;
  // pto: %cst_45
  const float v82 = 0.0f;
  // pto: %c0_index
  const int64_t v83 = 0;
  // pto: %c4_index
  const int64_t v84 = 4;
  // pto: %c2_index
  const int64_t v85 = 2;
  // pto: %c512_index
  const int64_t v86 = 512;
  // pto: %c256_index
  const int64_t v87 = 256;
  // pto: %cst_51
  const float v88 = 6.10351563E-5f;
  // pto: %cst_52
  const float v89 = 9.99999997E-7f;
  // pto: %c128_index
  const int64_t v90 = 128;
  // pto: %c192_index
  const int64_t v91 = 192;
  // pto: %cst_55
  const float v92 = 1.0f;
  // pto: %cst_56
  const float v93 = 0.5f;
  // pto: %cst_57
  const float v94 = 2.0f;
  // pto: %cst_58
  const float v95 = 0.25f;
  // pto: %cst_59
  const float v96 = 4.0f;
  // pto: %cst_60
  const float v97 = 0.125f;
  // pto: %cst_61
  const float v98 = 8.0f;
  // pto: %cst_62
  const float v99 = 0.0625f;
  // pto: %cst_63
  const float v100 = 16.0f;
  // pto: %c19_index
  const int64_t v101 = 19;
  // pto: %c64_index
  const int64_t v102 = 64;
  // pto: %c8192_index
  const int64_t v103 = 8192;
  // pto: %c12288_index
  const int64_t v104 = 12288;
  // pto: %228
  int64_t v105 = (int64_t) ((uint64_t) v29 * (uint64_t) v76);
  // pto: %rms_partials_inline1399__ssa_v0_view
  const int64_t v106 = 1;
  // pto: %rms_partials_inline1399__ssa_v0_view
  const int64_t v107 = 1;
  // pto: %rms_partials_inline1399__ssa_v0_view
  const int64_t v108 = 1;
  // pto: %rms_partials_inline1399__ssa_v0_view
  int64_t v109 = v105 * v72;
  // pto: %rms_partials_inline1399__ssa_v0_view
  int64_t v110 = v108 * v109;
  // pto: %rms_partials_inline1399__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v111 = pto::Shape<1, 1, 1, -1, -1>(v106, v107, v108, v105, v72);
  // pto: %rms_partials_inline1399__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v112 = pto::Stride<-1, -1, -1, -1, -1>(v107 * v110, v110, v109, v72, v105);
  // pto: %rms_partials_inline1399__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v113 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v111, v112);
  // pto: %x_flat_inline1398__ssa_v0_view
  const int64_t v114 = 1;
  // pto: %x_flat_inline1398__ssa_v0_view
  const int64_t v115 = 1;
  // pto: %x_flat_inline1398__ssa_v0_view
  const int64_t v116 = 1;
  // pto: %x_flat_inline1398__ssa_v0_view
  int64_t v117 = (int64_t) v30;
  // pto: %x_flat_inline1398__ssa_v0_view
  int64_t v118 = v117 * v74;
  // pto: %x_flat_inline1398__ssa_v0_view
  int64_t v119 = v116 * v118;
  // pto: %x_flat_inline1398__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v120 = pto::Shape<1, 1, 1, -1, -1>(v114, v115, v116, v117, v74);
  // pto: %x_flat_inline1398__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v121 = pto::Stride<-1, -1, -1, -1, -1>(v115 * v119, v119, v118, v74, v72);
  // pto: %x_flat_inline1398__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v122 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v120, v121);
  // pto: %229
  int64_t v123 = (int64_t) ((uint64_t) v29 * (uint64_t) v71);
  // pto: %parts_2d_inline1364__ssa_v0_view
  const int64_t v124 = 1;
  // pto: %parts_2d_inline1364__ssa_v0_view
  const int64_t v125 = 1;
  // pto: %parts_2d_inline1364__ssa_v0_view
  const int64_t v126 = 1;
  // pto: %parts_2d_inline1364__ssa_v0_view
  int64_t v127 = v76 * v123;
  // pto: %parts_2d_inline1364__ssa_v0_view
  int64_t v128 = v126 * v127;
  // pto: %parts_2d_inline1364__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v129 = pto::Shape<1, 1, 1, -1, -1>(v124, v125, v126, v76, v123);
  // pto: %parts_2d_inline1364__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v130 = pto::Stride<-1, -1, -1, -1, -1>(v125 * v128, v128, v127, v123, v72);
  // pto: %parts_2d_inline1364__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v131 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v129, v130);
  // pto: %rms_2d_inline1338__ssa_v0_view
  const int64_t v132 = 1;
  // pto: %rms_2d_inline1338__ssa_v0_view
  const int64_t v133 = 1;
  // pto: %rms_2d_inline1338__ssa_v0_view
  const int64_t v134 = 1;
  // pto: %rms_2d_inline1338__ssa_v0_view
  int64_t v135 = (int64_t) v29;
  // pto: %rms_2d_inline1338__ssa_v0_view
  int64_t v136 = v76 * v135;
  // pto: %rms_2d_inline1338__ssa_v0_view
  int64_t v137 = v134 * v136;
  // pto: %rms_2d_inline1338__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v138 = pto::Shape<1, 1, 1, -1, -1>(v132, v133, v134, v76, (int64_t) v29);
  // pto: %rms_2d_inline1338__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v139 = pto::Stride<-1, -1, -1, -1, -1>(v133 * v137, v137, v136, v135, v72);
  // pto: %rms_2d_inline1338__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v140 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v138, v139);
  // pto: %hc_base_2d_inline1433__ssa_v0_view
  const int64_t v141 = 1;
  // pto: %hc_base_2d_inline1433__ssa_v0_view
  const int64_t v142 = 1;
  // pto: %hc_base_2d_inline1433__ssa_v0_view
  const int64_t v143 = 1;
  // pto: %hc_base_2d_inline1433__ssa_v0_view
  int64_t v144 = v72 * v73;
  // pto: %hc_base_2d_inline1433__ssa_v0_view
  int64_t v145 = v143 * v144;
  // pto: %hc_base_2d_inline1433__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v146 = pto::Shape<1, 1, 1, -1, -1>(v141, v142, v143, v72, v73);
  // pto: %hc_base_2d_inline1433__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v147 = pto::Stride<-1, -1, -1, -1, -1>(v142 * v145, v145, v144, v73, v72);
  // pto: %hc_base_2d_inline1433__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v148 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v8, v146, v147);
  // pto: %comb_inline1359__ssa_v0_view
  const int64_t v149 = 1;
  // pto: %comb_inline1359__ssa_v0_view
  const int64_t v150 = 1;
  // pto: %comb_inline1359__ssa_v0_view
  const int64_t v151 = 1;
  // pto: %comb_inline1359__ssa_v0_view
  int64_t v152 = (int64_t) v30;
  // pto: %comb_inline1359__ssa_v0_view
  int64_t v153 = v152 * v77;
  // pto: %comb_inline1359__ssa_v0_view
  int64_t v154 = v151 * v153;
  // pto: %comb_inline1359__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v155 = pto::Shape<1, 1, 1, -1, -1>(v149, v150, v151, v152, v77);
  // pto: %comb_inline1359__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v156 = pto::Stride<-1, -1, -1, -1, -1>(v150 * v154, v154, v153, v77, v72);
  // pto: %comb_inline1359__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v157 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v9, v155, v156);
  // pto: %hc_scale_2d_inline1412__ssa_v0_view
  const int64_t v158 = 1;
  // pto: %hc_scale_2d_inline1412__ssa_v0_view
  const int64_t v159 = 1;
  // pto: %hc_scale_2d_inline1412__ssa_v0_view
  const int64_t v160 = 1;
  // pto: %hc_scale_2d_inline1412__ssa_v0_view
  int64_t v161 = v72 * v75;
  // pto: %hc_scale_2d_inline1412__ssa_v0_view
  int64_t v162 = v160 * v161;
  // pto: %hc_scale_2d_inline1412__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v163 = pto::Shape<1, 1, 1, -1, -1>(v158, v159, v160, v72, v75);
  // pto: %hc_scale_2d_inline1412__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v164 = pto::Stride<-1, -1, -1, -1, -1>(v159 * v162, v162, v161, v75, v72);
  // pto: %hc_scale_2d_inline1412__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v165 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v10, v163, v164);
  // pto: %230
  int64_t v166 = (int64_t) ((uint64_t) v30 * (uint64_t) v84);
  // pto: %post_flat_inline1380__ssa_v0_view
  const int64_t v167 = 1;
  // pto: %post_flat_inline1380__ssa_v0_view
  const int64_t v168 = 1;
  // pto: %post_flat_inline1380__ssa_v0_view
  const int64_t v169 = 1;
  // pto: %post_flat_inline1380__ssa_v0_view
  int64_t v170 = v72 * v166;
  // pto: %post_flat_inline1380__ssa_v0_view
  int64_t v171 = v169 * v170;
  // pto: %post_flat_inline1380__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v172 = pto::Shape<1, 1, 1, -1, -1>(v167, v168, v169, v72, v166);
  // pto: %post_flat_inline1380__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v173 = pto::Stride<-1, -1, -1, -1, -1>(v168 * v171, v171, v170, v166, v72);
  // pto: %post_flat_inline1380__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v174 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v11, v172, v173);
  // pto: %mixed_view_inline1406__ssa_v0_view
  const int64_t v175 = 1;
  // pto: %mixed_view_inline1406__ssa_v0_view
  const int64_t v176 = 1;
  // pto: %mixed_view_inline1406__ssa_v0_view
  const int64_t v177 = 1;
  // pto: %mixed_view_inline1406__ssa_v0_view
  int64_t v178 = (int64_t) v30;
  // pto: %mixed_view_inline1406__ssa_v0_view
  int64_t v179 = v178 * v78;
  // pto: %mixed_view_inline1406__ssa_v0_view
  int64_t v180 = v177 * v179;
  // pto: %mixed_view_inline1406__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v181 = pto::Shape<1, 1, 1, -1, -1>(v175, v176, v177, v178, v78);
  // pto: %mixed_view_inline1406__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v182 = pto::Stride<-1, -1, -1, -1, -1>(v176 * v180, v180, v179, v78, v72);
  // pto: %mixed_view_inline1406__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v183 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v12, v181, v182);
  const int32_t v184 = 0;
  auto v185 = TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>(v13, v184, v79);
  // pto: %worker_inline1352__ssa_v0
  int64_t v186 = (int64_t) v31;
  // pto: %linear_workers_inline1471__ssa_v0_idx
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
  for (int64_t v187 = v186; v187 < v14; v187 += (int64_t) v15) {
    // pto: %48, %49
    ;
    int64_t v188 = (int64_t) ((uint64_t) (v187 / v76) * (uint64_t) v80);
    // pto: %50
    ;
    int64_t v189 = v187 % v76;
    // pto: %51
    ;
    int64_t v190 = (int64_t) ((uint64_t) v189 * (uint64_t) v81);
    // pto: %mid_stream_rms_total_inline1385__tile
    ;
    Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v191 = Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v73);
    // pto: %mid_stream_rms_total_inline1385__tile
    ;
    uint64_t v192 = (uint64_t) v37;
    TASSIGN(v191, v192);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(v191, v82);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    for (int64_t v193 = v83; v193 < v84; v193 += v85) {
      // pto: %54
      ;
      int64_t v194 = (int64_t) ((uint64_t) v193 * (uint64_t) v86);
      // pto: %55
      ;
      int64_t v195 = (int64_t) ((uint64_t) v190 + (uint64_t) v194);
      // pto: %57, %lane_inline1330__ssa_v0, %56
      ;
      int64_t v196 = (int64_t) ((uint64_t) v188 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) * (uint64_t) v73));
      // pto: %58
      ;
      int64_t v197 = (int64_t) ((uint64_t) v17 - (uint64_t) v196);
      // pto: %59
      ;
      int64_t v198 = v197 < v73 ? v197 : v73;
      // pto: %60
      ;
      int64_t v199 = v198 < v83 ? v83 : v198;
      // pto: %63, %62
      ;
      int64_t v200 = (int64_t) ((uint64_t) v190 + (uint64_t) ((int64_t) (uint64_t) v194 + (uint64_t) v86));
      // pto: %mid_stream_source_inline1438__tile
      ;
      Tile<TileType::Vec, bfloat16_t, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v201 = Tile<TileType::Vec, bfloat16_t, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v199, v86);
      // pto: %mid_stream_source_inline1438__tile
      ;
      uint64_t v202 = (uint64_t) v38;
      TASSIGN(v201, v202);
      // pto: %70
      ;
      int64_t v203 = v196 < v83 ? v83 : v196;
      // pto: %71
      ;
      int64_t v204 = v195 < v83 ? v83 : v195;
      // pto: %x_flat_inline1398__ssa_v0_pview
      ;
      const int64_t v205 = 0;
      // pto: %x_flat_inline1398__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v206 = PTOAS__GLOBAL_TENSOR_DATA(v122);
      // pto: %x_flat_inline1398__ssa_v0_pview
      ;
      const int64_t v207 = 1;
      // pto: %x_flat_inline1398__ssa_v0_pview
      ;
      const int64_t v208 = 1;
      // pto: %x_flat_inline1398__ssa_v0_pview
      ;
      const int64_t v209 = 1;
      // pto: %x_flat_inline1398__ssa_v0_pview
      ;
      int64_t v210 = v199 * v74;
      // pto: %x_flat_inline1398__ssa_v0_pview
      ;
      int64_t v211 = v209 * v210;
      // pto: %x_flat_inline1398__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, -1> v212 = pto::Shape<1, 1, 1, -1, -1>(v207, v208, v209, v199, v86);
      // pto: %x_flat_inline1398__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v213 = pto::Stride<-1, -1, -1, -1, -1>(v208 * v211, v211, v210, v74, v72);
      // pto: %x_flat_inline1398__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v214 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v206 + (v205 + v203 * v74 + v204 * v72), v212, v213);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v201, v214);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %0
      ;
      Tile<TileType::Vec, bfloat16_t, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v215 = Tile<TileType::Vec, bfloat16_t, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v199, v86);
      // pto: %0
      ;
      uint64_t v216 = (uint64_t) v39;
      TASSIGN(v215, v216);
      // pto: %73
      ;
      int64_t v217 = v200 < v83 ? v83 : v200;
      // pto: %74
      ;
      const int64_t v218 = 0;
      // pto: %74
      ;
      __gm__ bfloat16_t* v219 = PTOAS__GLOBAL_TENSOR_DATA(v122);
      // pto: %74
      ;
      const int64_t v220 = 1;
      // pto: %74
      ;
      const int64_t v221 = 1;
      // pto: %74
      ;
      const int64_t v222 = 1;
      // pto: %74
      ;
      int64_t v223 = v199 * v74;
      // pto: %74
      ;
      int64_t v224 = v222 * v223;
      // pto: %74
      ;
      pto::Shape<1, 1, 1, -1, -1> v225 = pto::Shape<1, 1, 1, -1, -1>(v220, v221, v222, v199, v86);
      // pto: %74
      ;
      pto::Stride<-1, -1, -1, -1, -1> v226 = pto::Stride<-1, -1, -1, -1, -1>(v221 * v224, v224, v223, v74, v72);
      // pto: %74
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v227 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v219 + (v218 + v203 * v74 + v217 * v72), v225, v226);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      TLOAD(v215, v227);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %mid_stream_widened_inline1339__tile
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v228 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v199, v86);
      // pto: %mid_stream_widened_inline1339__tile
      ;
      uint64_t v229 = (uint64_t) v40;
      TASSIGN(v228, v229);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      RoundMode v230 = RoundMode::CAST_ROUND;
      SaturationMode v231 = SaturationMode::ON;
      TCVT(v228, v201, v230, v231);
      v228.SetValidShape(v199, v86);
      // pto: %mid_stream_clean_inline1405__tile
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v232 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v73, v86);
      // pto: %mid_stream_clean_inline1405__tile
      ;
      uint64_t v233 = (uint64_t) v40;
      TASSIGN(v232, v233);
      pipe_barrier(PIPE_V);
      TFILLPAD<pto::TFillPadMode::InPlace>(v232, v228);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      v232.SetValidShape(v73, v86);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TPUSH<TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>, Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v185, v232);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      // pto: %1
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v234 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v73, v86);
      // pto: %1
      ;
      uint64_t v235 = (uint64_t) v40;
      TASSIGN(v234, v235);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      TMUL(v234, v232, v232);
      // pto: %tmp_tile
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v236 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v86);
      // pto: %tmp_tile
      ;
      uint64_t v237 = (uint64_t) v38;
      TASSIGN(v236, v237);
      // pto: %2
      ;
      Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v238 = Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %2
      ;
      uint64_t v239 = (uint64_t) v41;
      TASSIGN(v238, v239);
      pipe_barrier(PIPE_V);
      TROWSUM(v238, v234, v236);
      // pto: %mid_stream_sq_inline1321__tile
      ;
      Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v240 = Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v73);
      // pto: %mid_stream_sq_inline1321__tile
      ;
      uint64_t v241 = (uint64_t) v41;
      TASSIGN(v240, v241);
      // pto: %3
      ;
      Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v242 = Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v73);
      // pto: %3
      ;
      uint64_t v243 = (uint64_t) v41;
      TASSIGN(v242, v243);
      pipe_barrier(PIPE_V);
      TADD(v242, v191, v240);
      // pto: %4
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v244 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v199, v86);
      // pto: %4
      ;
      uint64_t v245 = (uint64_t) v40;
      TASSIGN(v244, v245);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      RoundMode v246 = RoundMode::CAST_ROUND;
      SaturationMode v247 = SaturationMode::ON;
      TCVT(v244, v215, v246, v247);
      v244.SetValidShape(v199, v86);
      // pto: %6
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v248 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v73, v86);
      // pto: %6
      ;
      uint64_t v249 = (uint64_t) v40;
      TASSIGN(v248, v249);
      pipe_barrier(PIPE_V);
      TFILLPAD<pto::TFillPadMode::InPlace>(v248, v244);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      v248.SetValidShape(v73, v86);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      TPUSH<TPipe<0, Direction::DIR_V2C, 98304, 2, 2, false>, Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v185, v248);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      // pto: %8
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v250 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v73, v86);
      // pto: %8
      ;
      uint64_t v251 = (uint64_t) v40;
      TASSIGN(v250, v251);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      TMUL(v250, v248, v248);
      // pto: %9
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v252 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v86);
      // pto: %9
      ;
      uint64_t v253 = (uint64_t) v38;
      TASSIGN(v252, v253);
      // pto: %10
      ;
      Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v254 = Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
      // pto: %10
      ;
      uint64_t v255 = (uint64_t) v39;
      TASSIGN(v254, v255);
      pipe_barrier(PIPE_V);
      TROWSUM(v254, v250, v252);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      // pto: %11
      ;
      Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v256 = Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v73);
      // pto: %11
      ;
      uint64_t v257 = (uint64_t) v39;
      TASSIGN(v256, v257);
      // pto: %12
      ;
      Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v258 = Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v73);
      // pto: %12
      ;
      uint64_t v259 = (uint64_t) v37;
      TASSIGN(v258, v259);
      pipe_barrier(PIPE_V);
      TADD(v258, v242, v256);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    // pto: %52, %53, %76, %rms_lane_inline1454__ssa_v0, %75
    ;
    int64_t v260 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v189 * (uint64_t) v16) + (uint64_t) v188) + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) * (uint64_t) v73));
    // pto: %13
    ;
    Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v261 = Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v73, v72);
    // pto: %13
    ;
    uint64_t v262 = (uint64_t) v37;
    TASSIGN(v261, v262);
    // pto: %77
    ;
    int64_t v263 = v260 < v83 ? v83 : v260;
    // pto: %rms_partials_inline1399__iter_v1_pview
    ;
    const int64_t v264 = 0;
    // pto: %rms_partials_inline1399__iter_v1_pview
    ;
    __gm__ float* v265 = PTOAS__GLOBAL_TENSOR_DATA(v113);
    // pto: %rms_partials_inline1399__iter_v1_pview
    ;
    const int64_t v266 = 1;
    // pto: %rms_partials_inline1399__iter_v1_pview
    ;
    const int64_t v267 = 1;
    // pto: %rms_partials_inline1399__iter_v1_pview
    ;
    const int64_t v268 = 1;
    // pto: %rms_partials_inline1399__iter_v1_pview
    ;
    int64_t v269 = v73 * v72;
    // pto: %rms_partials_inline1399__iter_v1_pview
    ;
    int64_t v270 = v268 * v269;
    // pto: %rms_partials_inline1399__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 24, 1> v271 = pto::Shape<1, 1, 1, 24, 1>(v266, v267, v268, v73, v72);
    // pto: %rms_partials_inline1399__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v272 = pto::Stride<-1, -1, -1, -1, -1>(v267 * v270, v270, v269, v72, v105);
    // pto: %rms_partials_inline1399__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 24, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v273 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v265 + (v264 + v263 * v72 + v83 * v105), v271, v272);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
    TSTORE(v273, v261);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  pipe_barrier(PIPE_ALL);
  dsb(DSB_DDR);
  SYNCALL<SyncCoreType::Mix>();
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  // pto: %78, %79, %tail_lane_inline1366__ssa_v0
  int64_t v274 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v186 * (uint64_t) v85) + (uint64_t) ((int64_t) v33));
  // pto: %80
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  if (v274 < v18) {
    // pto: %82
    ;
    int64_t v275 = v274 < v20 ? v274 : v20;
    // pto: %81, %83
    ;
    int64_t v276 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v274 * (uint64_t) v19) + (uint64_t) v275);
    // pto: %85
    ;
    int64_t v277 = (int64_t) ((uint64_t) v20 - (uint64_t) v274);
    // pto: %86
    ;
    int64_t v278 = v277 < v83 ? v83 : v277;
    // pto: %87
    ;
    int64_t v279 = v278 < v72 ? v278 : v72;
    // pto: %84, %88
    ;
    int64_t v280 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v276 + (uint64_t) v19) + (uint64_t) v279);
    for (int64_t v281 = v276; v281 < v280; v281 += v76) {
      // pto: %scale2_inline1444__tile
      ;
      float v282 = (v5)[v85];
      // pto: %89
      ;
      int64_t v283 = (int64_t) ((uint64_t) v280 - (uint64_t) v281);
      // pto: %90
      ;
      int64_t v284 = v283 < v76 ? v283 : v76;
      // pto: %91
      ;
      int64_t v285 = (int64_t) ((uint64_t) v284 * (uint64_t) v71);
      // pto: %c_parts_load_inline1387__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v286 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v285);
      // pto: %c_parts_load_inline1387__ssa_v0
      ;
      uint64_t v287 = (uint64_t) v40;
      TASSIGN(v286, v287);
      // pto: %92
      ;
      int64_t v288 = (int64_t) ((uint64_t) v281 * (uint64_t) v71);
      // pto: %93
      ;
      int64_t v289 = v288 < v83 ? v83 : v288;
      // pto: %parts_2d_inline1364__ssa_v0_pview
      ;
      const int64_t v290 = 0;
      // pto: %parts_2d_inline1364__ssa_v0_pview
      ;
      __gm__ float* v291 = PTOAS__GLOBAL_TENSOR_DATA(v131);
      // pto: %parts_2d_inline1364__ssa_v0_pview
      ;
      const int64_t v292 = 1;
      // pto: %parts_2d_inline1364__ssa_v0_pview
      ;
      const int64_t v293 = 1;
      // pto: %parts_2d_inline1364__ssa_v0_pview
      ;
      const int64_t v294 = 1;
      // pto: %parts_2d_inline1364__ssa_v0_pview
      ;
      int64_t v295 = v76 * v123;
      // pto: %parts_2d_inline1364__ssa_v0_pview
      ;
      int64_t v296 = v294 * v295;
      // pto: %parts_2d_inline1364__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, -1> v297 = pto::Shape<1, 1, 1, 8, -1>(v292, v293, v294, v76, v285);
      // pto: %parts_2d_inline1364__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v298 = pto::Stride<-1, -1, -1, -1, -1>(v293 * v296, v296, v295, v123, v72);
      // pto: %parts_2d_inline1364__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v299 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v291 + (v290 + v83 * v123 + v289 * v72), v297, v298);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      TLOAD(v286, v299);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %t__tmp_v355
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v300 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v76, v87);
      // pto: %t__tmp_v355
      ;
      uint64_t v301 = (uint64_t) v40;
      TASSIGN(v300, v301);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      TFILLPAD<pto::TFillPadMode::InPlace>(v300, v286);
      v300.SetValidShape(v76, v87);
      // pto: %t__tmp_v356
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v302 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v87);
      // pto: %t__tmp_v356
      ;
      uint64_t v303 = (uint64_t) v40;
      TASSIGN(v302, v303);
      pipe_barrier(PIPE_V);
      TCOLSUM(v302, v300);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      // pto: %c_total_inline1490__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v304 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v71);
      // pto: %c_total_inline1490__ssa_v0
      ;
      uint64_t v305 = (uint64_t) v40;
      TASSIGN(v304, v305);
      // pto: %c_sq_load_inline1298__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v306 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v284);
      // pto: %c_sq_load_inline1298__ssa_v0
      ;
      uint64_t v307 = (uint64_t) v42;
      TASSIGN(v306, v307);
      // pto: %95
      ;
      int64_t v308 = v281 < v83 ? v83 : v281;
      // pto: %rms_2d_inline1338__ssa_v0_pview
      ;
      int64_t v309 = (int64_t) v29;
      // pto: %rms_2d_inline1338__ssa_v0_pview
      ;
      const int64_t v310 = 0;
      // pto: %rms_2d_inline1338__ssa_v0_pview
      ;
      __gm__ float* v311 = PTOAS__GLOBAL_TENSOR_DATA(v140);
      // pto: %rms_2d_inline1338__ssa_v0_pview
      ;
      const int64_t v312 = 1;
      // pto: %rms_2d_inline1338__ssa_v0_pview
      ;
      const int64_t v313 = 1;
      // pto: %rms_2d_inline1338__ssa_v0_pview
      ;
      const int64_t v314 = 1;
      // pto: %rms_2d_inline1338__ssa_v0_pview
      ;
      int64_t v315 = v76 * v309;
      // pto: %rms_2d_inline1338__ssa_v0_pview
      ;
      int64_t v316 = v314 * v315;
      // pto: %rms_2d_inline1338__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, -1> v317 = pto::Shape<1, 1, 1, 8, -1>(v312, v313, v314, v76, v284);
      // pto: %rms_2d_inline1338__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v318 = pto::Stride<-1, -1, -1, -1, -1>(v313 * v316, v316, v315, v309, v72);
      // pto: %rms_2d_inline1338__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v319 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v311 + (v310 + v83 * v309 + v308 * v72), v317, v318);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v306, v319);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %t__tmp_v357
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v320 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v76, v76);
      // pto: %t__tmp_v357
      ;
      uint64_t v321 = (uint64_t) v42;
      TASSIGN(v320, v321);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      TFILLPAD<pto::TFillPadMode::InPlace>(v320, v306);
      v320.SetValidShape(v76, v76);
      // pto: %c_rms_tmp_inline1297__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v322 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %c_rms_tmp_inline1297__ssa_v0
      ;
      uint64_t v323 = (uint64_t) v43;
      TASSIGN(v322, v323);
      // pto: %t__tmp_v358
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v324 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %t__tmp_v358
      ;
      uint64_t v325 = (uint64_t) v42;
      TASSIGN(v324, v325);
      pipe_barrier(PIPE_V);
      TCOLSUM(v324, v320);
      // pto: %t__tmp_v359
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v326 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %t__tmp_v359
      ;
      uint64_t v327 = (uint64_t) v42;
      TASSIGN(v326, v327);
      pipe_barrier(PIPE_V);
      TMULS(v326, v324, v88);
      // pto: %t__tmp_v360
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v328 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %t__tmp_v360
      ;
      uint64_t v329 = (uint64_t) v42;
      TASSIGN(v328, v329);
      pipe_barrier(PIPE_V);
      TADDS(v328, v326, v89);
      // pto: %t__tmp_v361
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v330 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %t__tmp_v361
      ;
      uint64_t v331 = (uint64_t) v44;
      TASSIGN(v330, v331);
      pipe_barrier(PIPE_V);
      TRSQRT(v330, v328, v322);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      // pto: %c_inv_inline1301__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v332 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v72);
      // pto: %c_inv_inline1301__ssa_v0
      ;
      uint64_t v333 = (uint64_t) v44;
      TASSIGN(v332, v333);
      // pto: %comb_inv_inline1483__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v334 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v72);
      // pto: %comb_inv_inline1483__ssa_v0
      ;
      uint64_t v335 = (uint64_t) v44;
      TASSIGN(v334, v335);
      // pto: %t__tmp_v362
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v336 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v77);
      // pto: %t__tmp_v362
      ;
      uint64_t v337 = (uint64_t) v45;
      TASSIGN(v336, v337);
      // pto: %slice_view
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v338;
      // pto: %slice_view
      ;
      uint64_t v339 = (uint64_t) v45;
      TASSIGN(v338, v339);
      // pto: %raw_inline1346__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v340 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v77);
      // pto: %raw_inline1346__ssa_v0
      ;
      uint64_t v341 = (uint64_t) v40;
      TASSIGN(v340, v341);
      TADDS(v340, v338, v82);
      // pto: %base_inline1460__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v342 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v77);
      // pto: %base_inline1460__ssa_v0
      ;
      uint64_t v343 = (uint64_t) v42;
      TASSIGN(v342, v343);
      // pto: %hc_base_2d_inline1433__ssa_v0_pview
      ;
      __gm__ float* v344 = PTOAS__GLOBAL_TENSOR_DATA(v148);
      // pto: %hc_base_2d_inline1433__ssa_v0_pview
      ;
      unsigned v345 = 8;
      // pto: %hc_base_2d_inline1433__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 16> v346 = pto::Shape<1, 1, 1, 1, 16>();
      // pto: %hc_base_2d_inline1433__ssa_v0_pview
      ;
      pto::Stride<24, 24, 24, 24, 1> v347 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %hc_base_2d_inline1433__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v348 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v344 + v345, v346, v347);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      TLOAD(v342, v348);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      // pto: %t__tmp_v363
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v349 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v77);
      // pto: %t__tmp_v363
      ;
      uint64_t v350 = (uint64_t) v46;
      TASSIGN(v349, v350);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v349, v340, v334);
      // pto: %t__tmp_v364
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v351 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v77);
      // pto: %t__tmp_v364
      ;
      uint64_t v352 = (uint64_t) v46;
      TASSIGN(v351, v352);
      pipe_barrier(PIPE_V);
      TMULS(v351, v349, v282);
      // pto: %t__tmp_v365
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v353 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v77);
      // pto: %t__tmp_v365
      ;
      uint64_t v354 = (uint64_t) v40;
      TASSIGN(v353, v354);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      TCOLEXPAND(v353, v342);
      // pto: %logits_inline1463__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v355 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v77);
      // pto: %logits_inline1463__ssa_v0
      ;
      uint64_t v356 = (uint64_t) v47;
      TASSIGN(v355, v356);
      pipe_barrier(PIPE_V);
      TADD(v355, v351, v353);
      // pto: %t__tmp_v366
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v357 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v77);
      // pto: %t__tmp_v366
      ;
      uint64_t v358 = (uint64_t) v47;
      TASSIGN(v357, v358);
      // pto: %regular_logits_inline1397__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v359 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %regular_logits_inline1397__ssa_v0
      ;
      uint64_t v360 = (uint64_t) v47;
      TASSIGN(v359, v360);
      // pto: %ids_inline1450__ci_tmp_v0
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v361 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v91);
      // pto: %ids_inline1450__ci_tmp_v0
      ;
      uint64_t v362 = (uint64_t) v40;
      TASSIGN(v361, v362);
      // pto: %ids_inline1450__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v363 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %ids_inline1450__ssa_v1
      ;
      uint64_t v364 = (uint64_t) v43;
      TASSIGN(v363, v364);
      // pto: %ci_tmp_view
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v365;
      TRESHAPE(v365, v361);
      // pto: %ci_dst_view
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v366;
      TRESHAPE(v366, v363);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v366, v79, v365);
      // pto: %tmp_inline1468__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v367 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %tmp_inline1468__ssa_v1
      ;
      uint64_t v368 = (uint64_t) v40;
      TASSIGN(v367, v368);
      // pto: %idx_float_inline1414__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v369 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %idx_float_inline1414__ssa_v0
      ;
      uint64_t v370 = (uint64_t) v43;
      TASSIGN(v369, v370);
      pipe_barrier(PIPE_V);
      RoundMode v371 = RoundMode::CAST_ROUND;
      SaturationMode v372 = SaturationMode::ON;
      TCVT(v369, v363, v371, v372);
      // pto: %t__tmp_v367
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v373 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v367
      ;
      uint64_t v374 = (uint64_t) v48;
      TASSIGN(v373, v374);
      pipe_barrier(PIPE_V);
      TMULS(v373, v369, v92);
      // pto: %t__tmp_v368
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v375 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v368
      ;
      uint64_t v376 = (uint64_t) v48;
      TASSIGN(v375, v376);
      pipe_barrier(PIPE_V);
      RoundMode v377 = RoundMode::CAST_FLOOR;
      SaturationMode v378 = SaturationMode::ON;
      TCVT(v375, v373, v377, v378);
      // pto: %ri1_low_inline1515__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v379 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %ri1_low_inline1515__ssa_v0
      ;
      uint64_t v380 = (uint64_t) v48;
      TASSIGN(v379, v380);
      pipe_barrier(PIPE_V);
      RoundMode v381 = RoundMode::CAST_ROUND;
      SaturationMode v382 = SaturationMode::ON;
      TCVT(v379, v375, v381, v382);
      // pto: %t__tmp_v369
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v383 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v369
      ;
      uint64_t v384 = (uint64_t) v49;
      TASSIGN(v383, v384);
      TMULS(v383, v369, v93);
      // pto: %t__tmp_v370
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v385 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v370
      ;
      uint64_t v386 = (uint64_t) v49;
      TASSIGN(v385, v386);
      pipe_barrier(PIPE_V);
      RoundMode v387 = RoundMode::CAST_FLOOR;
      SaturationMode v388 = SaturationMode::ON;
      TCVT(v385, v383, v387, v388);
      // pto: %ri1_high_inline1418__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v389 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %ri1_high_inline1418__ssa_v0
      ;
      uint64_t v390 = (uint64_t) v49;
      TASSIGN(v389, v390);
      pipe_barrier(PIPE_V);
      RoundMode v391 = RoundMode::CAST_ROUND;
      SaturationMode v392 = SaturationMode::ON;
      TCVT(v389, v385, v391, v392);
      // pto: %t__tmp_v371
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v393 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v371
      ;
      uint64_t v394 = (uint64_t) v49;
      TASSIGN(v393, v394);
      pipe_barrier(PIPE_V);
      TMULS(v393, v389, v94);
      // pto: %ri1_bit_inline1347__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v395 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %ri1_bit_inline1347__ssa_v0
      ;
      uint64_t v396 = (uint64_t) v48;
      TASSIGN(v395, v396);
      pipe_barrier(PIPE_V);
      TSUB(v395, v379, v393);
      // pto: %t__tmp_v372
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v397 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v372
      ;
      uint64_t v398 = (uint64_t) v48;
      TASSIGN(v397, v398);
      pipe_barrier(PIPE_V);
      TMULS(v397, v395, v94);
      // pto: %t__tmp_v373
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v399 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v373
      ;
      uint64_t v400 = (uint64_t) v48;
      TASSIGN(v399, v400);
      pipe_barrier(PIPE_V);
      TSUB(v399, v369, v397);
      // pto: %t__tmp_v374
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v401 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v374
      ;
      uint64_t v402 = (uint64_t) v48;
      TASSIGN(v401, v402);
      pipe_barrier(PIPE_V);
      TADDS(v401, v399, v92);
      // pto: %ri1_inline1372__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v403 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %ri1_inline1372__ssa_v1
      ;
      uint64_t v404 = (uint64_t) v48;
      TASSIGN(v403, v404);
      pipe_barrier(PIPE_V);
      RoundMode v405 = RoundMode::CAST_ROUND;
      SaturationMode v406 = SaturationMode::ON;
      TCVT(v403, v401, v405, v406);
      // pto: %t__tmp_v375
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v407 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v375
      ;
      uint64_t v408 = (uint64_t) v49;
      TASSIGN(v407, v408);
      TMULS(v407, v369, v93);
      // pto: %t__tmp_v376
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v409 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v376
      ;
      uint64_t v410 = (uint64_t) v49;
      TASSIGN(v409, v410);
      pipe_barrier(PIPE_V);
      RoundMode v411 = RoundMode::CAST_FLOOR;
      SaturationMode v412 = SaturationMode::ON;
      TCVT(v409, v407, v411, v412);
      // pto: %ri2_low_inline1516__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v413 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %ri2_low_inline1516__ssa_v0
      ;
      uint64_t v414 = (uint64_t) v49;
      TASSIGN(v413, v414);
      pipe_barrier(PIPE_V);
      RoundMode v415 = RoundMode::CAST_ROUND;
      SaturationMode v416 = SaturationMode::ON;
      TCVT(v413, v409, v415, v416);
      // pto: %t__tmp_v377
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v417 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v377
      ;
      uint64_t v418 = (uint64_t) v50;
      TASSIGN(v417, v418);
      TMULS(v417, v369, v95);
      // pto: %t__tmp_v378
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v419 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v378
      ;
      uint64_t v420 = (uint64_t) v50;
      TASSIGN(v419, v420);
      pipe_barrier(PIPE_V);
      RoundMode v421 = RoundMode::CAST_FLOOR;
      SaturationMode v422 = SaturationMode::ON;
      TCVT(v419, v417, v421, v422);
      // pto: %ri2_high_inline1517__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v423 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %ri2_high_inline1517__ssa_v0
      ;
      uint64_t v424 = (uint64_t) v50;
      TASSIGN(v423, v424);
      pipe_barrier(PIPE_V);
      RoundMode v425 = RoundMode::CAST_ROUND;
      SaturationMode v426 = SaturationMode::ON;
      TCVT(v423, v419, v425, v426);
      // pto: %t__tmp_v379
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v427 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v379
      ;
      uint64_t v428 = (uint64_t) v50;
      TASSIGN(v427, v428);
      pipe_barrier(PIPE_V);
      TMULS(v427, v423, v94);
      // pto: %ri2_bit_inline1499__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v429 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %ri2_bit_inline1499__ssa_v0
      ;
      uint64_t v430 = (uint64_t) v49;
      TASSIGN(v429, v430);
      pipe_barrier(PIPE_V);
      TSUB(v429, v413, v427);
      // pto: %t__tmp_v380
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v431 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v380
      ;
      uint64_t v432 = (uint64_t) v49;
      TASSIGN(v431, v432);
      pipe_barrier(PIPE_V);
      TMULS(v431, v429, v96);
      // pto: %t__tmp_v381
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v433 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v381
      ;
      uint64_t v434 = (uint64_t) v49;
      TASSIGN(v433, v434);
      pipe_barrier(PIPE_V);
      TSUB(v433, v369, v431);
      // pto: %t__tmp_v382
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v435 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v382
      ;
      uint64_t v436 = (uint64_t) v49;
      TASSIGN(v435, v436);
      pipe_barrier(PIPE_V);
      TADDS(v435, v433, v94);
      // pto: %ri2_inline1519__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v437 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %ri2_inline1519__ssa_v0
      ;
      uint64_t v438 = (uint64_t) v49;
      TASSIGN(v437, v438);
      pipe_barrier(PIPE_V);
      RoundMode v439 = RoundMode::CAST_ROUND;
      SaturationMode v440 = SaturationMode::ON;
      TCVT(v437, v435, v439, v440);
      // pto: %t__tmp_v383
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v441 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v383
      ;
      uint64_t v442 = (uint64_t) v50;
      TASSIGN(v441, v442);
      TMULS(v441, v369, v95);
      // pto: %t__tmp_v384
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v443 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v384
      ;
      uint64_t v444 = (uint64_t) v50;
      TASSIGN(v443, v444);
      pipe_barrier(PIPE_V);
      RoundMode v445 = RoundMode::CAST_FLOOR;
      SaturationMode v446 = SaturationMode::ON;
      TCVT(v443, v441, v445, v446);
      // pto: %ci1_low_inline1376__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v447 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %ci1_low_inline1376__ssa_v0
      ;
      uint64_t v448 = (uint64_t) v50;
      TASSIGN(v447, v448);
      pipe_barrier(PIPE_V);
      RoundMode v449 = RoundMode::CAST_ROUND;
      SaturationMode v450 = SaturationMode::ON;
      TCVT(v447, v443, v449, v450);
      // pto: %t__tmp_v385
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v451 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v385
      ;
      uint64_t v452 = (uint64_t) v51;
      TASSIGN(v451, v452);
      TMULS(v451, v369, v97);
      // pto: %t__tmp_v386
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v453 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v386
      ;
      uint64_t v454 = (uint64_t) v51;
      TASSIGN(v453, v454);
      pipe_barrier(PIPE_V);
      RoundMode v455 = RoundMode::CAST_FLOOR;
      SaturationMode v456 = SaturationMode::ON;
      TCVT(v453, v451, v455, v456);
      // pto: %ci1_high_inline1506__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v457 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %ci1_high_inline1506__ssa_v0
      ;
      uint64_t v458 = (uint64_t) v51;
      TASSIGN(v457, v458);
      pipe_barrier(PIPE_V);
      RoundMode v459 = RoundMode::CAST_ROUND;
      SaturationMode v460 = SaturationMode::ON;
      TCVT(v457, v453, v459, v460);
      // pto: %t__tmp_v387
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v461 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v387
      ;
      uint64_t v462 = (uint64_t) v51;
      TASSIGN(v461, v462);
      pipe_barrier(PIPE_V);
      TMULS(v461, v457, v94);
      // pto: %ci1_bit_inline1299__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v463 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %ci1_bit_inline1299__ssa_v0
      ;
      uint64_t v464 = (uint64_t) v50;
      TASSIGN(v463, v464);
      pipe_barrier(PIPE_V);
      TSUB(v463, v447, v461);
      // pto: %t__tmp_v388
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v465 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v388
      ;
      uint64_t v466 = (uint64_t) v50;
      TASSIGN(v465, v466);
      pipe_barrier(PIPE_V);
      TMULS(v465, v463, v98);
      // pto: %t__tmp_v389
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v467 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v389
      ;
      uint64_t v468 = (uint64_t) v50;
      TASSIGN(v467, v468);
      pipe_barrier(PIPE_V);
      TSUB(v467, v369, v465);
      // pto: %t__tmp_v390
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v469 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v390
      ;
      uint64_t v470 = (uint64_t) v50;
      TASSIGN(v469, v470);
      pipe_barrier(PIPE_V);
      TADDS(v469, v467, v96);
      // pto: %ci1_inline1523__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v471 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %ci1_inline1523__ssa_v0
      ;
      uint64_t v472 = (uint64_t) v50;
      TASSIGN(v471, v472);
      pipe_barrier(PIPE_V);
      RoundMode v473 = RoundMode::CAST_ROUND;
      SaturationMode v474 = SaturationMode::ON;
      TCVT(v471, v469, v473, v474);
      // pto: %t__tmp_v391
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v475 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v391
      ;
      uint64_t v476 = (uint64_t) v51;
      TASSIGN(v475, v476);
      TMULS(v475, v369, v97);
      // pto: %t__tmp_v392
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v477 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v392
      ;
      uint64_t v478 = (uint64_t) v51;
      TASSIGN(v477, v478);
      pipe_barrier(PIPE_V);
      RoundMode v479 = RoundMode::CAST_FLOOR;
      SaturationMode v480 = SaturationMode::ON;
      TCVT(v477, v475, v479, v480);
      // pto: %ci2_low_inline1524__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v481 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %ci2_low_inline1524__ssa_v0
      ;
      uint64_t v482 = (uint64_t) v51;
      TASSIGN(v481, v482);
      pipe_barrier(PIPE_V);
      RoundMode v483 = RoundMode::CAST_ROUND;
      SaturationMode v484 = SaturationMode::ON;
      TCVT(v481, v477, v483, v484);
      // pto: %t__tmp_v393
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v485 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v393
      ;
      uint64_t v486 = (uint64_t) v52;
      TASSIGN(v485, v486);
      TMULS(v485, v369, v99);
      // pto: %t__tmp_v394
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v487 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v394
      ;
      uint64_t v488 = (uint64_t) v52;
      TASSIGN(v487, v488);
      pipe_barrier(PIPE_V);
      RoundMode v489 = RoundMode::CAST_FLOOR;
      SaturationMode v490 = SaturationMode::ON;
      TCVT(v487, v485, v489, v490);
      // pto: %ci2_high_inline1466__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v491 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %ci2_high_inline1466__ssa_v0
      ;
      uint64_t v492 = (uint64_t) v52;
      TASSIGN(v491, v492);
      pipe_barrier(PIPE_V);
      RoundMode v493 = RoundMode::CAST_ROUND;
      SaturationMode v494 = SaturationMode::ON;
      TCVT(v491, v487, v493, v494);
      // pto: %t__tmp_v395
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v495 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v395
      ;
      uint64_t v496 = (uint64_t) v52;
      TASSIGN(v495, v496);
      pipe_barrier(PIPE_V);
      TMULS(v495, v491, v94);
      // pto: %ci2_bit_inline1526__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v497 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %ci2_bit_inline1526__ssa_v0
      ;
      uint64_t v498 = (uint64_t) v51;
      TASSIGN(v497, v498);
      pipe_barrier(PIPE_V);
      TSUB(v497, v481, v495);
      // pto: %t__tmp_v396
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v499 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v396
      ;
      uint64_t v500 = (uint64_t) v51;
      TASSIGN(v499, v500);
      pipe_barrier(PIPE_V);
      TMULS(v499, v497, v100);
      // pto: %t__tmp_v397
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v501 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v397
      ;
      uint64_t v502 = (uint64_t) v43;
      TASSIGN(v501, v502);
      pipe_barrier(PIPE_V);
      TSUB(v501, v369, v499);
      // pto: %t__tmp_v398
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v503 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v398
      ;
      uint64_t v504 = (uint64_t) v43;
      TASSIGN(v503, v504);
      pipe_barrier(PIPE_V);
      TADDS(v503, v501, v98);
      // pto: %ci2_inline1383__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v505 = Tile<TileType::Vec, int32_t, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %ci2_inline1383__ssa_v0
      ;
      uint64_t v506 = (uint64_t) v43;
      TASSIGN(v505, v506);
      pipe_barrier(PIPE_V);
      RoundMode v507 = RoundMode::CAST_ROUND;
      SaturationMode v508 = SaturationMode::ON;
      TCVT(v505, v503, v507, v508);
      // pto: %t__tmp_v399
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v509 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v399
      ;
      uint64_t v510 = (uint64_t) v51;
      TASSIGN(v509, v510);
      TGATHER(v509, v359, v403, v367);
      // pto: %max2_inline1520__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v511 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %max2_inline1520__ssa_v1
      ;
      uint64_t v512 = (uint64_t) v51;
      TASSIGN(v511, v512);
      pipe_barrier(PIPE_V);
      TMAX(v511, v359, v509);
      // pto: %t__tmp_v400
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v513 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v400
      ;
      uint64_t v514 = (uint64_t) v52;
      TASSIGN(v513, v514);
      pipe_barrier(PIPE_V);
      TGATHER(v513, v511, v437, v367);
      // pto: %max4_inline1496__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v515 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %max4_inline1496__ssa_v1
      ;
      uint64_t v516 = (uint64_t) v51;
      TASSIGN(v515, v516);
      pipe_barrier(PIPE_V);
      TMAX(v515, v511, v513);
      // pto: %t__tmp_v401
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v517 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v401
      ;
      uint64_t v518 = (uint64_t) v46;
      TASSIGN(v517, v518);
      pipe_barrier(PIPE_V);
      TSUB(v517, v359, v515);
      // pto: %exps_inline1498__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v519 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %exps_inline1498__ssa_v1
      ;
      uint64_t v520 = (uint64_t) v46;
      TASSIGN(v519, v520);
      pipe_barrier(PIPE_V);
      TEXP(v519, v517);
      // pto: %t__tmp_v402
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v521 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v402
      ;
      uint64_t v522 = (uint64_t) v51;
      TASSIGN(v521, v522);
      pipe_barrier(PIPE_V);
      TGATHER(v521, v519, v403, v367);
      // pto: %sum2_inline1362__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v523 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %sum2_inline1362__ssa_v1
      ;
      uint64_t v524 = (uint64_t) v51;
      TASSIGN(v523, v524);
      pipe_barrier(PIPE_V);
      TADD(v523, v519, v521);
      // pto: %t__tmp_v403
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v525 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v403
      ;
      uint64_t v526 = (uint64_t) v52;
      TASSIGN(v525, v526);
      pipe_barrier(PIPE_V);
      TGATHER(v525, v523, v437, v367);
      // pto: %sum4_inline1401__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v527 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %sum4_inline1401__ssa_v1
      ;
      uint64_t v528 = (uint64_t) v51;
      TASSIGN(v527, v528);
      pipe_barrier(PIPE_V);
      TADD(v527, v523, v525);
      // pto: %t__tmp_v404
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v529 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v404
      ;
      uint64_t v530 = (uint64_t) v46;
      TASSIGN(v529, v530);
      pipe_barrier(PIPE_V);
      TDIV(v529, v519, v527);
      // pto: %probs_inline1501__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v531 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %probs_inline1501__ssa_v1
      ;
      uint64_t v532 = (uint64_t) v46;
      TASSIGN(v531, v532);
      pipe_barrier(PIPE_V);
      TADDS(v531, v529, v89);
      // pto: %t__tmp_v405
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v533 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v405
      ;
      uint64_t v534 = (uint64_t) v51;
      TASSIGN(v533, v534);
      pipe_barrier(PIPE_V);
      TGATHER(v533, v531, v471, v367);
      // pto: %col2_inline1567__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v535 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %col2_inline1567__ssa_v4
      ;
      uint64_t v536 = (uint64_t) v51;
      TASSIGN(v535, v536);
      pipe_barrier(PIPE_V);
      TADD(v535, v531, v533);
      // pto: %t__tmp_v406
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v537 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v406
      ;
      uint64_t v538 = (uint64_t) v52;
      TASSIGN(v537, v538);
      pipe_barrier(PIPE_V);
      TGATHER(v537, v535, v505, v367);
      // pto: %t__tmp_v407
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v539 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %t__tmp_v407
      ;
      uint64_t v540 = (uint64_t) v51;
      TASSIGN(v539, v540);
      pipe_barrier(PIPE_V);
      TADD(v539, v535, v537);
      // pto: %col4_inline1507__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v541 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %col4_inline1507__ssa_v4
      ;
      uint64_t v542 = (uint64_t) v51;
      TASSIGN(v541, v542);
      pipe_barrier(PIPE_V);
      TADDS(v541, v539, v89);
      // pto: %cur_inline1382__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v543 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
      // pto: %cur_inline1382__ssa_v4
      ;
      uint64_t v544 = (uint64_t) v46;
      TASSIGN(v543, v544);
      pipe_barrier(PIPE_V);
      TDIV(v543, v531, v541);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      for (int64_t v545 = v83; v545 < v101; v545 += v72) {
        // pto: %t__tmp_v408
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v546 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
        // pto: %t__tmp_v408
        ;
        uint64_t v547 = (uint64_t) v51;
        TASSIGN(v546, v547);
        pipe_barrier(PIPE_V);
        TGATHER(v546, v543, v403, v367);
        // pto: %row2_inline1509__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v548 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
        // pto: %row2_inline1509__ssa_v1
        ;
        uint64_t v549 = (uint64_t) v51;
        TASSIGN(v548, v549);
        pipe_barrier(PIPE_V);
        TADD(v548, v543, v546);
        // pto: %t__tmp_v409
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v550 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
        // pto: %t__tmp_v409
        ;
        uint64_t v551 = (uint64_t) v52;
        TASSIGN(v550, v551);
        pipe_barrier(PIPE_V);
        TGATHER(v550, v548, v437, v367);
        // pto: %t__tmp_v410
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v552 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
        // pto: %t__tmp_v410
        ;
        uint64_t v553 = (uint64_t) v51;
        TASSIGN(v552, v553);
        pipe_barrier(PIPE_V);
        TADD(v552, v548, v550);
        // pto: %row4_inline1378__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v554 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
        // pto: %row4_inline1378__ssa_v1
        ;
        uint64_t v555 = (uint64_t) v51;
        TASSIGN(v554, v555);
        pipe_barrier(PIPE_V);
        TADDS(v554, v552, v89);
        // pto: %norm_inline1410__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v556 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
        // pto: %norm_inline1410__ssa_v1
        ;
        uint64_t v557 = (uint64_t) v51;
        TASSIGN(v556, v557);
        pipe_barrier(PIPE_V);
        TDIV(v556, v543, v554);
        // pto: %t__tmp_v411
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v558 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
        // pto: %t__tmp_v411
        ;
        uint64_t v559 = (uint64_t) v52;
        TASSIGN(v558, v559);
        pipe_barrier(PIPE_V);
        TGATHER(v558, v556, v471, v367);
        // pto: %col2_inline1567__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v560 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
        // pto: %col2_inline1567__ssa_v7
        ;
        uint64_t v561 = (uint64_t) v52;
        TASSIGN(v560, v561);
        pipe_barrier(PIPE_V);
        TADD(v560, v556, v558);
        // pto: %t__tmp_v412
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v562 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
        // pto: %t__tmp_v412
        ;
        uint64_t v563 = (uint64_t) v53;
        TASSIGN(v562, v563);
        pipe_barrier(PIPE_V);
        TGATHER(v562, v560, v505, v367);
        // pto: %t__tmp_v413
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v564 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
        // pto: %t__tmp_v413
        ;
        uint64_t v565 = (uint64_t) v52;
        TASSIGN(v564, v565);
        pipe_barrier(PIPE_V);
        TADD(v564, v560, v562);
        // pto: %col4_inline1507__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v566 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
        // pto: %col4_inline1507__ssa_v7
        ;
        uint64_t v567 = (uint64_t) v52;
        TASSIGN(v566, v567);
        pipe_barrier(PIPE_V);
        TADDS(v566, v564, v89);
        // pto: %cur_inline1382__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v568 = Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v90);
        // pto: %cur_inline1382__ssa_v7
        ;
        uint64_t v569 = (uint64_t) v46;
        TASSIGN(v568, v569);
        pipe_barrier(PIPE_V);
        TDIV(v568, v556, v566);
      };
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      // pto: %t__tmp_v414
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v570 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v77);
      // pto: %t__tmp_v414
      ;
      uint64_t v571 = (uint64_t) v46;
      TASSIGN(v570, v571);
      v570.SetValidShape(v284, v77);
      // pto: %comb_inline1359__ssa_v0_pview
      ;
      const int64_t v572 = 0;
      // pto: %comb_inline1359__ssa_v0_pview
      ;
      __gm__ float* v573 = PTOAS__GLOBAL_TENSOR_DATA(v157);
      // pto: %comb_inline1359__ssa_v0_pview
      ;
      const int64_t v574 = 1;
      // pto: %comb_inline1359__ssa_v0_pview
      ;
      const int64_t v575 = 1;
      // pto: %comb_inline1359__ssa_v0_pview
      ;
      const int64_t v576 = 1;
      // pto: %comb_inline1359__ssa_v0_pview
      ;
      int64_t v577 = v284 * v77;
      // pto: %comb_inline1359__ssa_v0_pview
      ;
      int64_t v578 = v576 * v577;
      // pto: %comb_inline1359__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, 16> v579 = pto::Shape<1, 1, 1, -1, 16>(v574, v575, v576, v284, v77);
      // pto: %comb_inline1359__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v580 = pto::Stride<-1, -1, -1, -1, -1>(v575 * v578, v578, v577, v77, v72);
      // pto: %comb_inline1359__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v581 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v573 + (v572 + v308 * v77 + v83 * v72), v579, v580);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      TSTORE(v581, v570);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    };
  } else {
    // pto: %98
    ;
    int64_t v582 = (int64_t) ((uint64_t) v274 - (uint64_t) v18);
    // pto: %100
    ;
    int64_t v583 = v582 < v22 ? v582 : v22;
    // pto: %99, %101, %102
    ;
    int64_t v584 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v582 * (uint64_t) v21) + (uint64_t) v583) * (uint64_t) v85);
    // pto: %103
    ;
    int64_t v585 = (int64_t) ((uint64_t) v22 - (uint64_t) v582);
    // pto: %104
    ;
    int64_t v586 = v585 < v83 ? v83 : v585;
    // pto: %105
    ;
    int64_t v587 = v586 < v72 ? v586 : v72;
    // pto: %108, %106, %107
    ;
    int64_t v588 = (int64_t) ((uint64_t) v584 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v21 + (uint64_t) v587) * (uint64_t) v85));
    // pto: %109
    ;
    // pto: %110
    ;
    // pto: %d_begin_inline1545__phi_v2
    ;
    int64_t v589;
    // pto: %d_end_inline1546__phi_v2
    ;
    int64_t v590;
    // pto: %work_begin_inline1311__phi_v3
    ;
    int64_t v591;
    // pto: %work_end_inline1307__phi_v3
    ;
    int64_t v592;
    if (v24 < v25) {
      // pto: %111
      ;
      // pto: %d_worker_inline1476__phi_v2
      ;
      int64_t v593;
      // pto: %d_workers_inline1548__phi_v2
      ;
      int64_t v594;
      // pto: %pair_id_inline1547__phi_v2
      ;
      int64_t v595;
      if (v582 < v26) {
        // pto: %112
        ;
        int64_t v596 = (int64_t) ((uint64_t) v27 + (uint64_t) v72);
        // pto: %113
        ;
        // pto: %115
        ;
        v593 = v582 % v596;
        v594 = v596;
        v595 = v582 / v596;
      } else {
        // pto: %117
        ;
        int64_t v597 = (int64_t) ((uint64_t) v582 - (uint64_t) v26);
        // pto: %119, %118
        ;
        // pto: %120
        ;
        v593 = v597 % v27;
        v594 = v27;
        v595 = (int64_t) ((uint64_t) v28 + (uint64_t) (v597 / v27));
      };
      // pto: %121
      ;
      uint64_t v598 = (uint64_t) v595;
      // pto: %121
      ;
      int64_t v599 = (int64_t) (v598 * (uint64_t) v85);
      // pto: %122
      ;
      int64_t v600 = v23 < v599 ? v23 : v599;
      // pto: %123
      ;
      int64_t v601 = (int64_t) ((uint64_t) v600 + (uint64_t) v85);
      // pto: %124
      ;
      // pto: %125
      ;
      int64_t v602 = v76 / v594;
      // pto: %126
      ;
      int64_t v603 = v76 % v594;
      // pto: %127
      ;
      uint64_t v604 = (uint64_t) v593;
      // pto: %128
      ;
      bool v605 = v593 < v603;
      // pto: %128
      ;
      int64_t v606 = v605 ? v593 : v603;
      // pto: %127, %129
      ;
      int64_t v607 = (int64_t) ((uint64_t) ((int64_t) v604 * (uint64_t) v602) + (uint64_t) v606);
      // pto: %131
      ;
      uint64_t v608 = (uint64_t) v593;
      // pto: %131
      ;
      int64_t v609 = (int64_t) ((uint64_t) v603 - v608);
      // pto: %132
      ;
      int64_t v610 = v609 < v83 ? v83 : v609;
      // pto: %133
      ;
      int64_t v611 = v610 < v72 ? v610 : v72;
      // pto: %130, %134
      ;
      v589 = v607;
      v590 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v607 + (uint64_t) v602) + (uint64_t) v611);
      v591 = v600;
      v592 = v23 < v601 ? v23 : v601;
    } else {
      v589 = v83;
      v590 = v76;
      v591 = v584;
      v592 = v23 < v588 ? v23 : v588;
    };
    // pto: %135
    ;
    bool v612 = v25 <= v582;
    // pto: %work_begin_inline1311__phi_v5
    ;
    int64_t v613 = v612 ? v83 : v591;
    // pto: %work_end_inline1307__phi_v5
    ;
    int64_t v614 = v612 ? v83 : v592;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
    for (int64_t v615 = v613; v615 < v614; v615 += v76) {
      // pto: %136
      ;
      int64_t v616 = (int64_t) ((uint64_t) v614 - (uint64_t) v615);
      // pto: %137
      ;
      int64_t v617 = v616 < v76 ? v616 : v76;
      // pto: %138
      ;
      int64_t v618 = (int64_t) ((uint64_t) v617 * (uint64_t) v71);
      // pto: %m_parts_load_inline1518__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v619 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v618);
      // pto: %m_parts_load_inline1518__ssa_v0
      ;
      uint64_t v620 = (uint64_t) v40;
      TASSIGN(v619, v620);
      // pto: %139
      ;
      int64_t v621 = (int64_t) ((uint64_t) v615 * (uint64_t) v71);
      // pto: %140
      ;
      int64_t v622 = v621 < v83 ? v83 : v621;
      // pto: %142
      ;
      const int64_t v623 = 0;
      // pto: %142
      ;
      __gm__ float* v624 = PTOAS__GLOBAL_TENSOR_DATA(v131);
      // pto: %142
      ;
      const int64_t v625 = 1;
      // pto: %142
      ;
      const int64_t v626 = 1;
      // pto: %142
      ;
      const int64_t v627 = 1;
      // pto: %142
      ;
      int64_t v628 = v76 * v123;
      // pto: %142
      ;
      int64_t v629 = v627 * v628;
      // pto: %142
      ;
      pto::Shape<1, 1, 1, 8, -1> v630 = pto::Shape<1, 1, 1, 8, -1>(v625, v626, v627, v76, v618);
      // pto: %142
      ;
      pto::Stride<-1, -1, -1, -1, -1> v631 = pto::Stride<-1, -1, -1, -1, -1>(v626 * v629, v629, v628, v123, v72);
      // pto: %142
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v632 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v624 + (v623 + v83 * v123 + v622 * v72), v630, v631);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      TLOAD(v619, v632);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      // pto: %t__tmp_v415
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v633 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v76, v87);
      // pto: %t__tmp_v415
      ;
      uint64_t v634 = (uint64_t) v40;
      TASSIGN(v633, v634);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      TFILLPAD<pto::TFillPadMode::InPlace>(v633, v619);
      v633.SetValidShape(v76, v87);
      // pto: %t__tmp_v416
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v635 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v87);
      // pto: %t__tmp_v416
      ;
      uint64_t v636 = (uint64_t) v40;
      TASSIGN(v635, v636);
      pipe_barrier(PIPE_V);
      TCOLSUM(v635, v633);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      // pto: %m_total_inline1559__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v637 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v71);
      // pto: %m_total_inline1559__ssa_v0
      ;
      uint64_t v638 = (uint64_t) v40;
      TASSIGN(v637, v638);
      // pto: %m_sq_load_inline1512__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v639 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v617);
      // pto: %m_sq_load_inline1512__ssa_v0
      ;
      uint64_t v640 = (uint64_t) v42;
      TASSIGN(v639, v640);
      // pto: %143
      ;
      int64_t v641 = v615 < v83 ? v83 : v615;
      // pto: %144
      ;
      int64_t v642 = (int64_t) v29;
      // pto: %144
      ;
      const int64_t v643 = 0;
      // pto: %144
      ;
      __gm__ float* v644 = PTOAS__GLOBAL_TENSOR_DATA(v140);
      // pto: %144
      ;
      const int64_t v645 = 1;
      // pto: %144
      ;
      const int64_t v646 = 1;
      // pto: %144
      ;
      const int64_t v647 = 1;
      // pto: %144
      ;
      int64_t v648 = v76 * v642;
      // pto: %144
      ;
      int64_t v649 = v647 * v648;
      // pto: %144
      ;
      pto::Shape<1, 1, 1, 8, -1> v650 = pto::Shape<1, 1, 1, 8, -1>(v645, v646, v647, v76, v617);
      // pto: %144
      ;
      pto::Stride<-1, -1, -1, -1, -1> v651 = pto::Stride<-1, -1, -1, -1, -1>(v646 * v649, v649, v648, v642, v72);
      // pto: %144
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v652 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v644 + (v643 + v83 * v642 + v641 * v72), v650, v651);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      TLOAD(v639, v652);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      // pto: %t__tmp_v417
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v653 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v76, v76);
      // pto: %t__tmp_v417
      ;
      uint64_t v654 = (uint64_t) v42;
      TASSIGN(v653, v654);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      TFILLPAD<pto::TFillPadMode::InPlace>(v653, v639);
      v653.SetValidShape(v76, v76);
      // pto: %m_rms_tmp_inline1554__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v655 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %m_rms_tmp_inline1554__ssa_v0
      ;
      uint64_t v656 = (uint64_t) v43;
      TASSIGN(v655, v656);
      // pto: %t__tmp_v418
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v657 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %t__tmp_v418
      ;
      uint64_t v658 = (uint64_t) v42;
      TASSIGN(v657, v658);
      pipe_barrier(PIPE_V);
      TCOLSUM(v657, v653);
      // pto: %t__tmp_v419
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v659 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %t__tmp_v419
      ;
      uint64_t v660 = (uint64_t) v42;
      TASSIGN(v659, v660);
      pipe_barrier(PIPE_V);
      TMULS(v659, v657, v88);
      // pto: %t__tmp_v420
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v661 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %t__tmp_v420
      ;
      uint64_t v662 = (uint64_t) v42;
      TASSIGN(v661, v662);
      pipe_barrier(PIPE_V);
      TADDS(v661, v659, v89);
      // pto: %t__tmp_v421
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v663 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %t__tmp_v421
      ;
      uint64_t v664 = (uint64_t) v48;
      TASSIGN(v663, v664);
      pipe_barrier(PIPE_V);
      TRSQRT(v663, v661, v655);
      // pto: %m_inv_inline1440__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v665 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v72);
      // pto: %m_inv_inline1440__ssa_v0
      ;
      uint64_t v666 = (uint64_t) v48;
      TASSIGN(v665, v666);
      // pto: %gate_base_inline1557__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v667 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %gate_base_inline1557__ssa_v0
      ;
      uint64_t v668 = (uint64_t) v54;
      TASSIGN(v667, v668);
      // pto: %145
      ;
      __gm__ float* v669 = PTOAS__GLOBAL_TENSOR_DATA(v148);
      // pto: %145
      ;
      pto::Shape<1, 1, 1, 1, 8> v670 = pto::Shape<1, 1, 1, 1, 8>();
      // pto: %145
      ;
      pto::Stride<24, 24, 24, 24, 1> v671 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %145
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v672 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v669, v670, v671);
      TLOAD(v667, v672);
      // pto: %gate_scales_inline1555__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v673 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v75);
      // pto: %gate_scales_inline1555__ssa_v0
      ;
      uint64_t v674 = (uint64_t) v55;
      TASSIGN(v673, v674);
      // pto: %hc_scale_2d_inline1412__ssa_v0_pview
      ;
      __gm__ float* v675 = PTOAS__GLOBAL_TENSOR_DATA(v165);
      // pto: %hc_scale_2d_inline1412__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 3> v676 = pto::Shape<1, 1, 1, 1, 3>();
      // pto: %hc_scale_2d_inline1412__ssa_v0_pview
      ;
      pto::Stride<3, 3, 3, 3, 1> v677 = pto::Stride<3, 3, 3, 3, 1>();
      // pto: %hc_scale_2d_inline1412__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND> v678 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND>(v675, v676, v677);
      TLOAD(v673, v678);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      // pto: %t__ci_tmp_v1
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v679 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v91);
      // pto: %t__ci_tmp_v1
      ;
      uint64_t v680 = (uint64_t) v42;
      TASSIGN(v679, v680);
      // pto: %t__tmp_v422
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v681 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %t__tmp_v422
      ;
      uint64_t v682 = (uint64_t) v56;
      TASSIGN(v681, v682);
      // pto: %146
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v683;
      TRESHAPE(v683, v679);
      // pto: %147
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v684;
      TRESHAPE(v684, v681);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v684, v79, v683);
      // pto: %gate_ids_inline1556__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v685 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %gate_ids_inline1556__ssa_v0
      ;
      uint64_t v686 = (uint64_t) v42;
      TASSIGN(v685, v686);
      pipe_barrier(PIPE_V);
      RoundMode v687 = RoundMode::CAST_ROUND;
      SaturationMode v688 = SaturationMode::ON;
      TCVT(v685, v681, v687, v688);
      // pto: %t__tmp_v423
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v689 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %t__tmp_v423
      ;
      uint64_t v690 = (uint64_t) v42;
      TASSIGN(v689, v690);
      pipe_barrier(PIPE_V);
      TMULS(v689, v685, v95);
      // pto: %gate_scale_ids_inline1558__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v691 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %gate_scale_ids_inline1558__ssa_v0
      ;
      uint64_t v692 = (uint64_t) v42;
      TASSIGN(v691, v692);
      pipe_barrier(PIPE_V);
      RoundMode v693 = RoundMode::CAST_FLOOR;
      SaturationMode v694 = SaturationMode::ON;
      TCVT(v691, v689, v693, v694);
      // pto: %gate_tmp_inline1561__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v695 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %gate_tmp_inline1561__ssa_v0
      ;
      uint64_t v696 = (uint64_t) v57;
      TASSIGN(v695, v696);
      // pto: %gate_scale_vec_inline1459__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v697 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %gate_scale_vec_inline1459__ssa_v0
      ;
      uint64_t v698 = (uint64_t) v44;
      TASSIGN(v697, v698);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      TGATHER(v697, v673, v691, v695);
      // pto: %t__tmp_v424
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v699 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v76);
      // pto: %t__tmp_v424
      ;
      uint64_t v700 = (uint64_t) v40;
      TASSIGN(v699, v700);
      // pto: %148
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v701;
      // pto: %148
      ;
      uint64_t v702 = (uint64_t) v40;
      TASSIGN(v701, v702);
      // pto: %gate_raw_inline1560__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v703 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v76);
      // pto: %gate_raw_inline1560__ssa_v0
      ;
      uint64_t v704 = (uint64_t) v40;
      TASSIGN(v703, v704);
      TADDS(v703, v701, v82);
      // pto: %gate_norm_inline1327__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v705 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v76);
      // pto: %gate_norm_inline1327__ssa_v0
      ;
      uint64_t v706 = (uint64_t) v40;
      TASSIGN(v705, v706);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v705, v703, v665);
      // pto: %gate_scaled_inline1562__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v707 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v76);
      // pto: %gate_scaled_inline1562__ssa_v0
      ;
      uint64_t v708 = (uint64_t) v40;
      TASSIGN(v707, v708);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v707, v705, v697);
      // pto: %t__tmp_v425
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v709 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v76);
      // pto: %t__tmp_v425
      ;
      uint64_t v710 = (uint64_t) v58;
      TASSIGN(v709, v710);
      TCOLEXPAND(v709, v667);
      // pto: %gate_logits_inline1551__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v711 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v76);
      // pto: %gate_logits_inline1551__ssa_v0
      ;
      uint64_t v712 = (uint64_t) v40;
      TASSIGN(v711, v712);
      pipe_barrier(PIPE_V);
      TADD(v711, v707, v709);
      // pto: %t__tmp_v426
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v713 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v76);
      // pto: %t__tmp_v426
      ;
      uint64_t v714 = (uint64_t) v40;
      TASSIGN(v713, v714);
      pipe_barrier(PIPE_V);
      TNEG(v713, v711);
      // pto: %t__tmp_v427
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v715 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v76);
      // pto: %t__tmp_v427
      ;
      uint64_t v716 = (uint64_t) v40;
      TASSIGN(v715, v716);
      pipe_barrier(PIPE_V);
      TEXP(v715, v713);
      // pto: %t__tmp_v428
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v717 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v76);
      // pto: %t__tmp_v428
      ;
      uint64_t v718 = (uint64_t) v40;
      TASSIGN(v717, v718);
      pipe_barrier(PIPE_V);
      TADDS(v717, v715, v92);
      // pto: %gate_sigmoid_inline1565__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v719 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v76);
      // pto: %gate_sigmoid_inline1565__ssa_v0
      ;
      uint64_t v720 = (uint64_t) v47;
      TASSIGN(v719, v720);
      pipe_barrier(PIPE_V);
      TRECIP(v719, v717);
      // pto: %transpose_tmp
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v721 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v76);
      // pto: %transpose_tmp
      ;
      uint64_t v722 = (uint64_t) v40;
      TASSIGN(v721, v722);
      // pto: %gate_transposed_inline1566__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v723 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v76);
      // pto: %gate_transposed_inline1566__ssa_v0
      ;
      uint64_t v724 = (uint64_t) v42;
      TASSIGN(v723, v724);
      pipe_barrier(PIPE_V);
      TTRANS(v723, v719, v721);
      // pto: %t__ci_tmp_v2
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v725 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v91);
      // pto: %t__ci_tmp_v2
      ;
      uint64_t v726 = (uint64_t) v40;
      TASSIGN(v725, v726);
      // pto: %t__tmp_v429
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v727 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v71);
      // pto: %t__tmp_v429
      ;
      uint64_t v728 = (uint64_t) v43;
      TASSIGN(v727, v728);
      // pto: %149
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v729;
      TRESHAPE(v729, v725);
      // pto: %150
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v730;
      TRESHAPE(v730, v727);
      TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v730, v79, v729);
      // pto: %post_ids_float_inline1568__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v731 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v71);
      // pto: %post_ids_float_inline1568__ssa_v0
      ;
      uint64_t v732 = (uint64_t) v40;
      TASSIGN(v731, v732);
      pipe_barrier(PIPE_V);
      RoundMode v733 = RoundMode::CAST_ROUND;
      SaturationMode v734 = SaturationMode::ON;
      TCVT(v731, v727, v733, v734);
      // pto: %t__tmp_v430
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v735 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v71);
      // pto: %t__tmp_v430
      ;
      uint64_t v736 = (uint64_t) v59;
      TASSIGN(v735, v736);
      pipe_barrier(PIPE_V);
      TMULS(v735, v731, v95);
      // pto: %t__tmp_v431
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v737 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v71);
      // pto: %t__tmp_v431
      ;
      uint64_t v738 = (uint64_t) v59;
      TASSIGN(v737, v738);
      pipe_barrier(PIPE_V);
      RoundMode v739 = RoundMode::CAST_FLOOR;
      SaturationMode v740 = SaturationMode::ON;
      TCVT(v737, v735, v739, v740);
      // pto: %post_rows_inline1296__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v741 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v71);
      // pto: %post_rows_inline1296__ssa_v0
      ;
      uint64_t v742 = (uint64_t) v59;
      TASSIGN(v741, v742);
      pipe_barrier(PIPE_V);
      RoundMode v743 = RoundMode::CAST_ROUND;
      SaturationMode v744 = SaturationMode::ON;
      TCVT(v741, v737, v743, v744);
      // pto: %t__tmp_v432
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v745 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v71);
      // pto: %t__tmp_v432
      ;
      uint64_t v746 = (uint64_t) v59;
      TASSIGN(v745, v746);
      pipe_barrier(PIPE_V);
      TMULS(v745, v741, v96);
      // pto: %t__tmp_v433
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v747 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v71);
      // pto: %t__tmp_v433
      ;
      uint64_t v748 = (uint64_t) v40;
      TASSIGN(v747, v748);
      pipe_barrier(PIPE_V);
      TADD(v747, v731, v745);
      // pto: %t__tmp_v434
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v749 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v71);
      // pto: %t__tmp_v434
      ;
      uint64_t v750 = (uint64_t) v40;
      TASSIGN(v749, v750);
      pipe_barrier(PIPE_V);
      TADDS(v749, v747, v96);
      // pto: %post_ids_inline1570__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v751 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v71);
      // pto: %post_ids_inline1570__ssa_v0
      ;
      uint64_t v752 = (uint64_t) v40;
      TASSIGN(v751, v752);
      pipe_barrier(PIPE_V);
      RoundMode v753 = RoundMode::CAST_ROUND;
      SaturationMode v754 = SaturationMode::ON;
      TCVT(v751, v749, v753, v754);
      // pto: %post_tmp_inline1572__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v755 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v71);
      // pto: %post_tmp_inline1572__ssa_v0
      ;
      uint64_t v756 = (uint64_t) v59;
      TASSIGN(v755, v756);
      // pto: %t__tmp_v435
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v757 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v102);
      // pto: %t__tmp_v435
      ;
      uint64_t v758 = (uint64_t) v47;
      TASSIGN(v757, v758);
      // pto: %post_values_inline1574__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v759 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v71);
      // pto: %post_values_inline1574__ssa_v0
      ;
      uint64_t v760 = (uint64_t) v58;
      TASSIGN(v759, v760);
      pipe_barrier(PIPE_V);
      TGATHER(v759, v757, v751, v755);
      // pto: %t__tmp_v436
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v761 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v71);
      // pto: %t__tmp_v436
      ;
      uint64_t v762 = (uint64_t) v40;
      TASSIGN(v761, v762);
      pipe_barrier(PIPE_V);
      TMULS(v761, v759, v94);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
      // pto: %151
      ;
      int64_t v763 = (int64_t) ((uint64_t) v617 * (uint64_t) v84);
      v761.SetValidShape(v72, v763);
      // pto: %152
      ;
      bool v764 = v589 == v83;
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
      if (v764) {
        // pto: %154
        ;
        int64_t v765 = (int64_t) ((uint64_t) v615 * (uint64_t) v84);
        // pto: %155
        ;
        int64_t v766 = v765 < v83 ? v83 : v765;
        // pto: %post_flat_inline1380__ssa_v0_pview
        ;
        const int64_t v767 = 0;
        // pto: %post_flat_inline1380__ssa_v0_pview
        ;
        __gm__ float* v768 = PTOAS__GLOBAL_TENSOR_DATA(v174);
        // pto: %post_flat_inline1380__ssa_v0_pview
        ;
        const int64_t v769 = 1;
        // pto: %post_flat_inline1380__ssa_v0_pview
        ;
        const int64_t v770 = 1;
        // pto: %post_flat_inline1380__ssa_v0_pview
        ;
        const int64_t v771 = 1;
        // pto: %post_flat_inline1380__ssa_v0_pview
        ;
        int64_t v772 = v72 * v166;
        // pto: %post_flat_inline1380__ssa_v0_pview
        ;
        int64_t v773 = v771 * v772;
        // pto: %post_flat_inline1380__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, -1> v774 = pto::Shape<1, 1, 1, 1, -1>(v769, v770, v771, v72, v763);
        // pto: %post_flat_inline1380__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v775 = pto::Stride<-1, -1, -1, -1, -1>(v770 * v773, v773, v772, v166, v72);
        // pto: %post_flat_inline1380__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v776 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v768 + (v767 + v83 * v166 + v766 * v72), v774, v775);
        TSTORE(v776, v761);
      };
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      // pto: %t__tmp_v437
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v777 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %t__tmp_v437
      ;
      uint64_t v778 = (uint64_t) v42;
      TASSIGN(v777, v778);
      // pto: %156
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v779;
      // pto: %156
      ;
      uint64_t v780 = (uint64_t) v42;
      TASSIGN(v779, v780);
      // pto: %t__tmp_v438
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v781 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %t__tmp_v438
      ;
      uint64_t v782 = (uint64_t) v60;
      TASSIGN(v781, v782);
      TADDS(v781, v779, v89);
      // pto: %pre0_inline1533__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v783 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v72);
      // pto: %pre0_inline1533__ssa_v0
      ;
      uint64_t v784 = (uint64_t) v60;
      TASSIGN(v783, v784);
      // pto: %t__tmp_v439
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v785 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %t__tmp_v439
      ;
      uint64_t v786 = (uint64_t) v57;
      TASSIGN(v785, v786);
      // pto: %157
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v787;
      // pto: %157
      ;
      uint64_t v788 = (uint64_t) v57;
      TASSIGN(v787, v788);
      // pto: %t__tmp_v440
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v789 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %t__tmp_v440
      ;
      uint64_t v790 = (uint64_t) v61;
      TASSIGN(v789, v790);
      TADDS(v789, v787, v89);
      // pto: %pre1_inline1273__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v791 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v72);
      // pto: %pre1_inline1273__ssa_v0
      ;
      uint64_t v792 = (uint64_t) v61;
      TASSIGN(v791, v792);
      // pto: %t__tmp_v441
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v793 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %t__tmp_v441
      ;
      uint64_t v794 = (uint64_t) v44;
      TASSIGN(v793, v794);
      // pto: %158
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v795;
      // pto: %158
      ;
      uint64_t v796 = (uint64_t) v44;
      TASSIGN(v795, v796);
      // pto: %t__tmp_v442
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v797 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %t__tmp_v442
      ;
      uint64_t v798 = (uint64_t) v62;
      TASSIGN(v797, v798);
      TADDS(v797, v795, v89);
      // pto: %pre2_inline1272__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v799 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v72);
      // pto: %pre2_inline1272__ssa_v0
      ;
      uint64_t v800 = (uint64_t) v62;
      TASSIGN(v799, v800);
      // pto: %t__tmp_v443
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v801 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %t__tmp_v443
      ;
      uint64_t v802 = (uint64_t) v63;
      TASSIGN(v801, v802);
      // pto: %159
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v803;
      // pto: %159
      ;
      uint64_t v804 = (uint64_t) v63;
      TASSIGN(v803, v804);
      // pto: %t__tmp_v444
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v805 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v72, v76);
      // pto: %t__tmp_v444
      ;
      uint64_t v806 = (uint64_t) v64;
      TASSIGN(v805, v806);
      TADDS(v805, v803, v89);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      // pto: %pre3_inline1271__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v807 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v72);
      // pto: %pre3_inline1271__ssa_v0
      ;
      uint64_t v808 = (uint64_t) v64;
      TASSIGN(v807, v808);
      // pto: %160
      ;
      uint64_t v809 = (uint64_t) v590;
      // pto: %160
      ;
      uint64_t v810 = (uint64_t) v589;
      // pto: %163
      ;
      uint64_t v811 = (uint64_t) v589;
      // pto: %160, %161, %162, %163
      ;
      int64_t v812 = (int64_t) (v811 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (v809 - v810) / v85) * (uint64_t) v85));
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      for (int64_t v813 = v589; v813 < v812; v813 += v85) {
        // pto: %164
        ;
        int64_t v814 = (int64_t) ((uint64_t) v813 * (uint64_t) v86);
        // pto: %167
        ;
        int64_t v815 = (int64_t) ((uint64_t) v814 + (uint64_t) v86);
        // pto: %x0_bf16_inline1269__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v816 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %x0_bf16_inline1269__ssa_v0
        ;
        uint64_t v817 = (uint64_t) v65;
        TASSIGN(v816, v817);
        // pto: %170
        ;
        int64_t v818 = v814 < v83 ? v83 : v814;
        // pto: %171
        ;
        const int64_t v819 = 0;
        // pto: %171
        ;
        __gm__ bfloat16_t* v820 = PTOAS__GLOBAL_TENSOR_DATA(v122);
        // pto: %171
        ;
        const int64_t v821 = 1;
        // pto: %171
        ;
        const int64_t v822 = 1;
        // pto: %171
        ;
        const int64_t v823 = 1;
        // pto: %171
        ;
        int64_t v824 = v617 * v74;
        // pto: %171
        ;
        int64_t v825 = v823 * v824;
        // pto: %171
        ;
        pto::Shape<1, 1, 1, -1, 512> v826 = pto::Shape<1, 1, 1, -1, 512>(v821, v822, v823, v617, v86);
        // pto: %171
        ;
        pto::Stride<-1, -1, -1, -1, -1> v827 = pto::Stride<-1, -1, -1, -1, -1>(v822 * v825, v825, v824, v74, v72);
        // pto: %171
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v828 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v820 + (v819 + v641 * v74 + v818 * v72), v826, v827);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v816, v828);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %x1_bf16_inline1446__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v829 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %x1_bf16_inline1446__ssa_v0
        ;
        uint64_t v830 = (uint64_t) v66;
        TASSIGN(v829, v830);
        // pto: %173
        ;
        int64_t v831 = (int64_t) ((uint64_t) v814 + (uint64_t) v78);
        // pto: %174
        ;
        int64_t v832 = v831 < v83 ? v83 : v831;
        // pto: %175
        ;
        const int64_t v833 = 0;
        // pto: %175
        ;
        __gm__ bfloat16_t* v834 = PTOAS__GLOBAL_TENSOR_DATA(v122);
        // pto: %175
        ;
        const int64_t v835 = 1;
        // pto: %175
        ;
        const int64_t v836 = 1;
        // pto: %175
        ;
        const int64_t v837 = 1;
        // pto: %175
        ;
        int64_t v838 = v617 * v74;
        // pto: %175
        ;
        int64_t v839 = v837 * v838;
        // pto: %175
        ;
        pto::Shape<1, 1, 1, -1, 512> v840 = pto::Shape<1, 1, 1, -1, 512>(v835, v836, v837, v617, v86);
        // pto: %175
        ;
        pto::Stride<-1, -1, -1, -1, -1> v841 = pto::Stride<-1, -1, -1, -1, -1>(v836 * v839, v839, v838, v74, v72);
        // pto: %175
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v842 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v834 + (v833 + v641 * v74 + v832 * v72), v840, v841);
        TLOAD(v829, v842);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %x2_bf16_inline1267__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v843 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %x2_bf16_inline1267__ssa_v0
        ;
        uint64_t v844 = (uint64_t) v38;
        TASSIGN(v843, v844);
        // pto: %177
        ;
        int64_t v845 = (int64_t) ((uint64_t) v814 + (uint64_t) v103);
        // pto: %178
        ;
        int64_t v846 = v845 < v83 ? v83 : v845;
        // pto: %179
        ;
        const int64_t v847 = 0;
        // pto: %179
        ;
        __gm__ bfloat16_t* v848 = PTOAS__GLOBAL_TENSOR_DATA(v122);
        // pto: %179
        ;
        const int64_t v849 = 1;
        // pto: %179
        ;
        const int64_t v850 = 1;
        // pto: %179
        ;
        const int64_t v851 = 1;
        // pto: %179
        ;
        int64_t v852 = v617 * v74;
        // pto: %179
        ;
        int64_t v853 = v851 * v852;
        // pto: %179
        ;
        pto::Shape<1, 1, 1, -1, 512> v854 = pto::Shape<1, 1, 1, -1, 512>(v849, v850, v851, v617, v86);
        // pto: %179
        ;
        pto::Stride<-1, -1, -1, -1, -1> v855 = pto::Stride<-1, -1, -1, -1, -1>(v850 * v853, v853, v852, v74, v72);
        // pto: %179
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v856 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v848 + (v847 + v641 * v74 + v846 * v72), v854, v855);
        TLOAD(v843, v856);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %x3_bf16_inline1266__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v857 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %x3_bf16_inline1266__ssa_v0
        ;
        uint64_t v858 = (uint64_t) v67;
        TASSIGN(v857, v858);
        // pto: %181
        ;
        int64_t v859 = (int64_t) ((uint64_t) v814 + (uint64_t) v104);
        // pto: %182
        ;
        int64_t v860 = v859 < v83 ? v83 : v859;
        // pto: %183
        ;
        const int64_t v861 = 0;
        // pto: %183
        ;
        __gm__ bfloat16_t* v862 = PTOAS__GLOBAL_TENSOR_DATA(v122);
        // pto: %183
        ;
        const int64_t v863 = 1;
        // pto: %183
        ;
        const int64_t v864 = 1;
        // pto: %183
        ;
        const int64_t v865 = 1;
        // pto: %183
        ;
        int64_t v866 = v617 * v74;
        // pto: %183
        ;
        int64_t v867 = v865 * v866;
        // pto: %183
        ;
        pto::Shape<1, 1, 1, -1, 512> v868 = pto::Shape<1, 1, 1, -1, 512>(v863, v864, v865, v617, v86);
        // pto: %183
        ;
        pto::Stride<-1, -1, -1, -1, -1> v869 = pto::Stride<-1, -1, -1, -1, -1>(v864 * v867, v867, v866, v74, v72);
        // pto: %183
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v870 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v862 + (v861 + v641 * v74 + v860 * v72), v868, v869);
        TLOAD(v857, v870);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %14
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v871 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %14
        ;
        uint64_t v872 = (uint64_t) v68;
        TASSIGN(v871, v872);
        // pto: %185
        ;
        int64_t v873 = v815 < v83 ? v83 : v815;
        // pto: %186
        ;
        const int64_t v874 = 0;
        // pto: %186
        ;
        __gm__ bfloat16_t* v875 = PTOAS__GLOBAL_TENSOR_DATA(v122);
        // pto: %186
        ;
        const int64_t v876 = 1;
        // pto: %186
        ;
        const int64_t v877 = 1;
        // pto: %186
        ;
        const int64_t v878 = 1;
        // pto: %186
        ;
        int64_t v879 = v617 * v74;
        // pto: %186
        ;
        int64_t v880 = v878 * v879;
        // pto: %186
        ;
        pto::Shape<1, 1, 1, -1, 512> v881 = pto::Shape<1, 1, 1, -1, 512>(v876, v877, v878, v617, v86);
        // pto: %186
        ;
        pto::Stride<-1, -1, -1, -1, -1> v882 = pto::Stride<-1, -1, -1, -1, -1>(v877 * v880, v880, v879, v74, v72);
        // pto: %186
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v883 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v875 + (v874 + v641 * v74 + v873 * v72), v881, v882);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        TLOAD(v871, v883);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %15
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v884 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %15
        ;
        uint64_t v885 = (uint64_t) v39;
        TASSIGN(v884, v885);
        int64_t v886 = (int64_t) ((uint64_t) v814 + (uint64_t) v36);
        // pto: %189
        ;
        int64_t v887 = v886 < v83 ? v83 : v886;
        // pto: %190
        ;
        const int64_t v888 = 0;
        // pto: %190
        ;
        __gm__ bfloat16_t* v889 = PTOAS__GLOBAL_TENSOR_DATA(v122);
        // pto: %190
        ;
        const int64_t v890 = 1;
        // pto: %190
        ;
        const int64_t v891 = 1;
        // pto: %190
        ;
        const int64_t v892 = 1;
        // pto: %190
        ;
        int64_t v893 = v617 * v74;
        // pto: %190
        ;
        int64_t v894 = v892 * v893;
        // pto: %190
        ;
        pto::Shape<1, 1, 1, -1, 512> v895 = pto::Shape<1, 1, 1, -1, 512>(v890, v891, v892, v617, v86);
        // pto: %190
        ;
        pto::Stride<-1, -1, -1, -1, -1> v896 = pto::Stride<-1, -1, -1, -1, -1>(v891 * v894, v894, v893, v74, v72);
        // pto: %190
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v897 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v889 + (v888 + v641 * v74 + v887 * v72), v895, v896);
        TLOAD(v884, v897);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %16
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v898 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %16
        ;
        uint64_t v899 = (uint64_t) v37;
        TASSIGN(v898, v899);
        int64_t v900 = (int64_t) ((uint64_t) v814 + (uint64_t) v35);
        // pto: %193
        ;
        int64_t v901 = v900 < v83 ? v83 : v900;
        // pto: %194
        ;
        const int64_t v902 = 0;
        // pto: %194
        ;
        __gm__ bfloat16_t* v903 = PTOAS__GLOBAL_TENSOR_DATA(v122);
        // pto: %194
        ;
        const int64_t v904 = 1;
        // pto: %194
        ;
        const int64_t v905 = 1;
        // pto: %194
        ;
        const int64_t v906 = 1;
        // pto: %194
        ;
        int64_t v907 = v617 * v74;
        // pto: %194
        ;
        int64_t v908 = v906 * v907;
        // pto: %194
        ;
        pto::Shape<1, 1, 1, -1, 512> v909 = pto::Shape<1, 1, 1, -1, 512>(v904, v905, v906, v617, v86);
        // pto: %194
        ;
        pto::Stride<-1, -1, -1, -1, -1> v910 = pto::Stride<-1, -1, -1, -1, -1>(v905 * v908, v908, v907, v74, v72);
        // pto: %194
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v911 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v903 + (v902 + v641 * v74 + v901 * v72), v909, v910);
        TLOAD(v898, v911);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        // pto: %17
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v912 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %17
        ;
        uint64_t v913 = (uint64_t) v69;
        TASSIGN(v912, v913);
        int64_t v914 = (int64_t) ((uint64_t) v814 + (uint64_t) v34);
        // pto: %197
        ;
        int64_t v915 = v914 < v83 ? v83 : v914;
        // pto: %198
        ;
        const int64_t v916 = 0;
        // pto: %198
        ;
        __gm__ bfloat16_t* v917 = PTOAS__GLOBAL_TENSOR_DATA(v122);
        // pto: %198
        ;
        const int64_t v918 = 1;
        // pto: %198
        ;
        const int64_t v919 = 1;
        // pto: %198
        ;
        const int64_t v920 = 1;
        // pto: %198
        ;
        int64_t v921 = v617 * v74;
        // pto: %198
        ;
        int64_t v922 = v920 * v921;
        // pto: %198
        ;
        pto::Shape<1, 1, 1, -1, 512> v923 = pto::Shape<1, 1, 1, -1, 512>(v918, v919, v920, v617, v86);
        // pto: %198
        ;
        pto::Stride<-1, -1, -1, -1, -1> v924 = pto::Stride<-1, -1, -1, -1, -1>(v919 * v922, v922, v921, v74, v72);
        // pto: %198
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v925 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v917 + (v916 + v641 * v74 + v915 * v72), v923, v924);
        TLOAD(v912, v925);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        // pto: %x0_inline1268__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v926 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %x0_inline1268__ssa_v0
        ;
        uint64_t v927 = (uint64_t) v40;
        TASSIGN(v926, v927);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        RoundMode v928 = RoundMode::CAST_ROUND;
        SaturationMode v929 = SaturationMode::ON;
        TCVT(v926, v816, v928, v929);
        // pto: %x1_inline1331__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v930 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %x1_inline1331__ssa_v0
        ;
        uint64_t v931 = (uint64_t) v65;
        TASSIGN(v930, v931);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        RoundMode v932 = RoundMode::CAST_ROUND;
        SaturationMode v933 = SaturationMode::ON;
        TCVT(v930, v829, v932, v933);
        // pto: %x2_inline1358__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v934 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %x2_inline1358__ssa_v0
        ;
        uint64_t v935 = (uint64_t) v66;
        TASSIGN(v934, v935);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        RoundMode v936 = RoundMode::CAST_ROUND;
        SaturationMode v937 = SaturationMode::ON;
        TCVT(v934, v843, v936, v937);
        // pto: %x3_inline1265__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v938 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %x3_inline1265__ssa_v0
        ;
        uint64_t v939 = (uint64_t) v38;
        TASSIGN(v938, v939);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        RoundMode v940 = RoundMode::CAST_ROUND;
        SaturationMode v941 = SaturationMode::ON;
        TCVT(v938, v857, v940, v941);
        // pto: %y0_inline1264__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v942 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %y0_inline1264__ssa_v0
        ;
        uint64_t v943 = (uint64_t) v40;
        TASSIGN(v942, v943);
        TROWEXPANDMUL(v942, v926, v783);
        // pto: %y1_inline1489__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v944 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %y1_inline1489__ssa_v0
        ;
        uint64_t v945 = (uint64_t) v65;
        TASSIGN(v944, v945);
        TROWEXPANDMUL(v944, v930, v791);
        // pto: %y2_inline1263__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v946 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %y2_inline1263__ssa_v0
        ;
        uint64_t v947 = (uint64_t) v66;
        TASSIGN(v946, v947);
        TROWEXPANDMUL(v946, v934, v799);
        // pto: %y3_inline1262__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v948 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %y3_inline1262__ssa_v0
        ;
        uint64_t v949 = (uint64_t) v38;
        TASSIGN(v948, v949);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v948, v938, v807);
        // pto: %t__tmp_v445
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v950 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %t__tmp_v445
        ;
        uint64_t v951 = (uint64_t) v40;
        TASSIGN(v950, v951);
        TADD(v950, v942, v944);
        // pto: %t__tmp_v446
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v952 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %t__tmp_v446
        ;
        uint64_t v953 = (uint64_t) v65;
        TASSIGN(v952, v953);
        pipe_barrier(PIPE_V);
        TADD(v952, v946, v948);
        // pto: %y_tile_inline1573__ssa_v0
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v954 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %y_tile_inline1573__ssa_v0
        ;
        uint64_t v955 = (uint64_t) v40;
        TASSIGN(v954, v955);
        pipe_barrier(PIPE_V);
        TADD(v954, v950, v952);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %y_bf16_inline1261__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v956 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %y_bf16_inline1261__ssa_v0
        ;
        uint64_t v957 = (uint64_t) v40;
        TASSIGN(v956, v957);
        pipe_barrier(PIPE_V);
        RoundMode v958 = RoundMode::CAST_RINT;
        SaturationMode v959 = SaturationMode::ON;
        TCVT(v956, v954, v958, v959);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        v956.SetValidShape(v617, v86);
        // pto: %mixed_view_inline1406__ssa_v0_pview
        ;
        const int64_t v960 = 0;
        // pto: %mixed_view_inline1406__ssa_v0_pview
        ;
        __gm__ bfloat16_t* v961 = PTOAS__GLOBAL_TENSOR_DATA(v183);
        // pto: %mixed_view_inline1406__ssa_v0_pview
        ;
        const int64_t v962 = 1;
        // pto: %mixed_view_inline1406__ssa_v0_pview
        ;
        const int64_t v963 = 1;
        // pto: %mixed_view_inline1406__ssa_v0_pview
        ;
        const int64_t v964 = 1;
        // pto: %mixed_view_inline1406__ssa_v0_pview
        ;
        int64_t v965 = v617 * v78;
        // pto: %mixed_view_inline1406__ssa_v0_pview
        ;
        int64_t v966 = v964 * v965;
        // pto: %mixed_view_inline1406__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, -1, 512> v967 = pto::Shape<1, 1, 1, -1, 512>(v962, v963, v964, v617, v86);
        // pto: %mixed_view_inline1406__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v968 = pto::Stride<-1, -1, -1, -1, -1>(v963 * v966, v966, v965, v78, v72);
        // pto: %mixed_view_inline1406__ssa_v0_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v969 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v961 + (v960 + v641 * v78 + v818 * v72), v967, v968);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v969, v956);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        // pto: %18
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v970 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %18
        ;
        uint64_t v971 = (uint64_t) v70;
        TASSIGN(v970, v971);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        RoundMode v972 = RoundMode::CAST_ROUND;
        SaturationMode v973 = SaturationMode::ON;
        TCVT(v970, v871, v972, v973);
        // pto: %19
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v974 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %19
        ;
        uint64_t v975 = (uint64_t) v68;
        TASSIGN(v974, v975);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        RoundMode v976 = RoundMode::CAST_ROUND;
        SaturationMode v977 = SaturationMode::ON;
        TCVT(v974, v884, v976, v977);
        // pto: %20
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v978 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %20
        ;
        uint64_t v979 = (uint64_t) v39;
        TASSIGN(v978, v979);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        RoundMode v980 = RoundMode::CAST_ROUND;
        SaturationMode v981 = SaturationMode::ON;
        TCVT(v978, v898, v980, v981);
        // pto: %21
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v982 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %21
        ;
        uint64_t v983 = (uint64_t) v37;
        TASSIGN(v982, v983);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        RoundMode v984 = RoundMode::CAST_ROUND;
        SaturationMode v985 = SaturationMode::ON;
        TCVT(v982, v912, v984, v985);
        // pto: %22
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v986 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %22
        ;
        uint64_t v987 = (uint64_t) v70;
        TASSIGN(v986, v987);
        TROWEXPANDMUL(v986, v970, v783);
        // pto: %23
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v988 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %23
        ;
        uint64_t v989 = (uint64_t) v68;
        TASSIGN(v988, v989);
        TROWEXPANDMUL(v988, v974, v791);
        // pto: %24
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v990 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %24
        ;
        uint64_t v991 = (uint64_t) v39;
        TASSIGN(v990, v991);
        TROWEXPANDMUL(v990, v978, v799);
        // pto: %25
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v992 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %25
        ;
        uint64_t v993 = (uint64_t) v37;
        TASSIGN(v992, v993);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v992, v982, v807);
        // pto: %26
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v994 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %26
        ;
        uint64_t v995 = (uint64_t) v70;
        TASSIGN(v994, v995);
        TADD(v994, v986, v988);
        // pto: %27
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v996 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %27
        ;
        uint64_t v997 = (uint64_t) v68;
        TASSIGN(v996, v997);
        pipe_barrier(PIPE_V);
        TADD(v996, v990, v992);
        // pto: %28
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v998 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %28
        ;
        uint64_t v999 = (uint64_t) v70;
        TASSIGN(v998, v999);
        pipe_barrier(PIPE_V);
        TADD(v998, v994, v996);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        // pto: %29
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1000 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %29
        ;
        uint64_t v1001 = (uint64_t) v70;
        TASSIGN(v1000, v1001);
        pipe_barrier(PIPE_V);
        RoundMode v1002 = RoundMode::CAST_RINT;
        SaturationMode v1003 = SaturationMode::ON;
        TCVT(v1000, v998, v1002, v1003);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        v1000.SetValidShape(v617, v86);
        // pto: %204
        ;
        const int64_t v1004 = 0;
        // pto: %204
        ;
        __gm__ bfloat16_t* v1005 = PTOAS__GLOBAL_TENSOR_DATA(v183);
        // pto: %204
        ;
        const int64_t v1006 = 1;
        // pto: %204
        ;
        const int64_t v1007 = 1;
        // pto: %204
        ;
        const int64_t v1008 = 1;
        // pto: %204
        ;
        int64_t v1009 = v617 * v78;
        // pto: %204
        ;
        int64_t v1010 = v1008 * v1009;
        // pto: %204
        ;
        pto::Shape<1, 1, 1, -1, 512> v1011 = pto::Shape<1, 1, 1, -1, 512>(v1006, v1007, v1008, v617, v86);
        // pto: %204
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1012 = pto::Stride<-1, -1, -1, -1, -1>(v1007 * v1010, v1010, v1009, v78, v72);
        // pto: %204
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1013 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1005 + (v1004 + v641 * v78 + v873 * v72), v1011, v1012);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v1013, v1000);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
      };
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      // pto: %205
      ;
      uint64_t v1014 = (uint64_t) v590;
      // pto: %205, %206
      ;
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID6);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      if ((int64_t) (v1014 - (uint64_t) v812) == v72) {
        // pto: %207
        ;
        int64_t v1015 = (int64_t) ((uint64_t) v812 * (uint64_t) v86);
        // pto: %31
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1016 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %31
        ;
        uint64_t v1017 = (uint64_t) v65;
        TASSIGN(v1016, v1017);
        // pto: %210
        ;
        int64_t v1018 = v1015 < v83 ? v83 : v1015;
        // pto: %211
        ;
        const int64_t v1019 = 0;
        // pto: %211
        ;
        __gm__ bfloat16_t* v1020 = PTOAS__GLOBAL_TENSOR_DATA(v122);
        // pto: %211
        ;
        const int64_t v1021 = 1;
        // pto: %211
        ;
        const int64_t v1022 = 1;
        // pto: %211
        ;
        const int64_t v1023 = 1;
        // pto: %211
        ;
        int64_t v1024 = v617 * v74;
        // pto: %211
        ;
        int64_t v1025 = v1023 * v1024;
        // pto: %211
        ;
        pto::Shape<1, 1, 1, -1, 512> v1026 = pto::Shape<1, 1, 1, -1, 512>(v1021, v1022, v1023, v617, v86);
        // pto: %211
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1027 = pto::Stride<-1, -1, -1, -1, -1>(v1022 * v1025, v1025, v1024, v74, v72);
        // pto: %211
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1028 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1020 + (v1019 + v641 * v74 + v1018 * v72), v1026, v1027);
        TLOAD(v1016, v1028);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %32
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1029 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %32
        ;
        uint64_t v1030 = (uint64_t) v40;
        TASSIGN(v1029, v1030);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v1031 = RoundMode::CAST_ROUND;
        SaturationMode v1032 = SaturationMode::ON;
        TCVT(v1029, v1016, v1031, v1032);
        // pto: %33
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1033 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %33
        ;
        uint64_t v1034 = (uint64_t) v66;
        TASSIGN(v1033, v1034);
        // pto: %213
        ;
        int64_t v1035 = (int64_t) ((uint64_t) v1015 + (uint64_t) v78);
        // pto: %214
        ;
        int64_t v1036 = v1035 < v83 ? v83 : v1035;
        // pto: %215
        ;
        const int64_t v1037 = 0;
        // pto: %215
        ;
        __gm__ bfloat16_t* v1038 = PTOAS__GLOBAL_TENSOR_DATA(v122);
        // pto: %215
        ;
        const int64_t v1039 = 1;
        // pto: %215
        ;
        const int64_t v1040 = 1;
        // pto: %215
        ;
        const int64_t v1041 = 1;
        // pto: %215
        ;
        int64_t v1042 = v617 * v74;
        // pto: %215
        ;
        int64_t v1043 = v1041 * v1042;
        // pto: %215
        ;
        pto::Shape<1, 1, 1, -1, 512> v1044 = pto::Shape<1, 1, 1, -1, 512>(v1039, v1040, v1041, v617, v86);
        // pto: %215
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1045 = pto::Stride<-1, -1, -1, -1, -1>(v1040 * v1043, v1043, v1042, v74, v72);
        // pto: %215
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1046 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1038 + (v1037 + v641 * v74 + v1036 * v72), v1044, v1045);
        TLOAD(v1033, v1046);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %34
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1047 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %34
        ;
        uint64_t v1048 = (uint64_t) v65;
        TASSIGN(v1047, v1048);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v1049 = RoundMode::CAST_ROUND;
        SaturationMode v1050 = SaturationMode::ON;
        TCVT(v1047, v1033, v1049, v1050);
        // pto: %35
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1051 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %35
        ;
        uint64_t v1052 = (uint64_t) v38;
        TASSIGN(v1051, v1052);
        // pto: %217
        ;
        int64_t v1053 = (int64_t) ((uint64_t) v1015 + (uint64_t) v103);
        // pto: %218
        ;
        int64_t v1054 = v1053 < v83 ? v83 : v1053;
        // pto: %219
        ;
        const int64_t v1055 = 0;
        // pto: %219
        ;
        __gm__ bfloat16_t* v1056 = PTOAS__GLOBAL_TENSOR_DATA(v122);
        // pto: %219
        ;
        const int64_t v1057 = 1;
        // pto: %219
        ;
        const int64_t v1058 = 1;
        // pto: %219
        ;
        const int64_t v1059 = 1;
        // pto: %219
        ;
        int64_t v1060 = v617 * v74;
        // pto: %219
        ;
        int64_t v1061 = v1059 * v1060;
        // pto: %219
        ;
        pto::Shape<1, 1, 1, -1, 512> v1062 = pto::Shape<1, 1, 1, -1, 512>(v1057, v1058, v1059, v617, v86);
        // pto: %219
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1063 = pto::Stride<-1, -1, -1, -1, -1>(v1058 * v1061, v1061, v1060, v74, v72);
        // pto: %219
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1064 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1056 + (v1055 + v641 * v74 + v1054 * v72), v1062, v1063);
        TLOAD(v1051, v1064);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %36
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1065 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %36
        ;
        uint64_t v1066 = (uint64_t) v66;
        TASSIGN(v1065, v1066);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v1067 = RoundMode::CAST_ROUND;
        SaturationMode v1068 = SaturationMode::ON;
        TCVT(v1065, v1051, v1067, v1068);
        // pto: %37
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1069 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %37
        ;
        uint64_t v1070 = (uint64_t) v70;
        TASSIGN(v1069, v1070);
        // pto: %221
        ;
        int64_t v1071 = (int64_t) ((uint64_t) v1015 + (uint64_t) v104);
        // pto: %222
        ;
        int64_t v1072 = v1071 < v83 ? v83 : v1071;
        // pto: %223
        ;
        const int64_t v1073 = 0;
        // pto: %223
        ;
        __gm__ bfloat16_t* v1074 = PTOAS__GLOBAL_TENSOR_DATA(v122);
        // pto: %223
        ;
        const int64_t v1075 = 1;
        // pto: %223
        ;
        const int64_t v1076 = 1;
        // pto: %223
        ;
        const int64_t v1077 = 1;
        // pto: %223
        ;
        int64_t v1078 = v617 * v74;
        // pto: %223
        ;
        int64_t v1079 = v1077 * v1078;
        // pto: %223
        ;
        pto::Shape<1, 1, 1, -1, 512> v1080 = pto::Shape<1, 1, 1, -1, 512>(v1075, v1076, v1077, v617, v86);
        // pto: %223
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1081 = pto::Stride<-1, -1, -1, -1, -1>(v1076 * v1079, v1079, v1078, v74, v72);
        // pto: %223
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1082 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1074 + (v1073 + v641 * v74 + v1072 * v72), v1080, v1081);
        TLOAD(v1069, v1082);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %38
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1083 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %38
        ;
        uint64_t v1084 = (uint64_t) v38;
        TASSIGN(v1083, v1084);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v1085 = RoundMode::CAST_ROUND;
        SaturationMode v1086 = SaturationMode::ON;
        TCVT(v1083, v1069, v1085, v1086);
        // pto: %39
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1087 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %39
        ;
        uint64_t v1088 = (uint64_t) v40;
        TASSIGN(v1087, v1088);
        TROWEXPANDMUL(v1087, v1029, v783);
        // pto: %40
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1089 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %40
        ;
        uint64_t v1090 = (uint64_t) v65;
        TASSIGN(v1089, v1090);
        TROWEXPANDMUL(v1089, v1047, v791);
        // pto: %41
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1091 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %41
        ;
        uint64_t v1092 = (uint64_t) v66;
        TASSIGN(v1091, v1092);
        TROWEXPANDMUL(v1091, v1065, v799);
        // pto: %42
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1093 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %42
        ;
        uint64_t v1094 = (uint64_t) v38;
        TASSIGN(v1093, v1094);
        pipe_barrier(PIPE_V);
        TROWEXPANDMUL(v1093, v1083, v807);
        // pto: %43
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1095 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %43
        ;
        uint64_t v1096 = (uint64_t) v40;
        TASSIGN(v1095, v1096);
        TADD(v1095, v1087, v1089);
        // pto: %44
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1097 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %44
        ;
        uint64_t v1098 = (uint64_t) v65;
        TASSIGN(v1097, v1098);
        pipe_barrier(PIPE_V);
        TADD(v1097, v1091, v1093);
        // pto: %45
        ;
        Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1099 = Tile<TileType::Vec, float, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %45
        ;
        uint64_t v1100 = (uint64_t) v40;
        TASSIGN(v1099, v1100);
        pipe_barrier(PIPE_V);
        TADD(v1099, v1095, v1097);
        // pto: %46
        ;
        Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1101 = Tile<TileType::Vec, bfloat16_t, 8, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v617, v86);
        // pto: %46
        ;
        uint64_t v1102 = (uint64_t) v40;
        TASSIGN(v1101, v1102);
        pipe_barrier(PIPE_V);
        RoundMode v1103 = RoundMode::CAST_RINT;
        SaturationMode v1104 = SaturationMode::ON;
        TCVT(v1101, v1099, v1103, v1104);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        v1101.SetValidShape(v617, v86);
        // pto: %227
        ;
        const int64_t v1105 = 0;
        // pto: %227
        ;
        __gm__ bfloat16_t* v1106 = PTOAS__GLOBAL_TENSOR_DATA(v183);
        // pto: %227
        ;
        const int64_t v1107 = 1;
        // pto: %227
        ;
        const int64_t v1108 = 1;
        // pto: %227
        ;
        const int64_t v1109 = 1;
        // pto: %227
        ;
        int64_t v1110 = v617 * v78;
        // pto: %227
        ;
        int64_t v1111 = v1109 * v1110;
        // pto: %227
        ;
        pto::Shape<1, 1, 1, -1, 512> v1112 = pto::Shape<1, 1, 1, -1, 512>(v1107, v1108, v1109, v617, v86);
        // pto: %227
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1113 = pto::Stride<-1, -1, -1, -1, -1>(v1108 * v1111, v1111, v1110, v78, v72);
        // pto: %227
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1114 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, 512>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1106 + (v1105 + v641 * v78 + v1018 * v72), v1112, v1113);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v1114, v1101);
      };
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}