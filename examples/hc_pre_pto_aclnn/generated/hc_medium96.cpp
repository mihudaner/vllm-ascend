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

AICORE void hc_medium96_aic(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ bfloat16_t* v12, __gm__ float* v13, int64_t v14, int32_t v15, int64_t v16, int64_t v17, int64_t v18, int64_t v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int64_t v28, int64_t v29, int64_t v30, int32_t v31, int32_t v32) {
  using T = float;

  #if defined(__DAV_CUBE__)
  const int64_t v33 = 768;
  const int32_t v34 = 0;
  // pto: %c0_i64
  const int64_t v35 = 0;
  // pto: %c262144_i64
  const int64_t v36 = 262144;
  // pto: %c32768_i64
  const int64_t v37 = 32768;
  // pto: %c32_index
  const int64_t v38 = 32;
  // pto: %c1_index
  const int64_t v39 = 1;
  // pto: %c24_index
  const int64_t v40 = 24;
  // pto: %c16384_index
  const int64_t v41 = 16384;
  // pto: %c8_index
  const int64_t v42 = 8;
  // pto: %c2048_index
  const int64_t v43 = 2048;
  // pto: %c0_index
  const int64_t v44 = 0;
  // pto: %c2_index
  const int64_t v45 = 2;
  // pto: %c1024_index
  const int64_t v46 = 1024;
  // pto: %c512_index
  const int64_t v47 = 512;
  // pto: %c256_index
  const int64_t v48 = 256;
  // pto: %cn256_index
  const int64_t v49 = -256;
  // pto: %40
  int64_t v50 = (int64_t) ((uint64_t) v29 * (uint64_t) v42);
  // pto: %mixes_partials_inline693__ssa_v0_view
  const int64_t v51 = 1;
  // pto: %mixes_partials_inline693__ssa_v0_view
  const int64_t v52 = 1;
  // pto: %mixes_partials_inline693__ssa_v0_view
  const int64_t v53 = 1;
  // pto: %mixes_partials_inline693__ssa_v0_view
  int64_t v54 = v50 * v38;
  // pto: %mixes_partials_inline693__ssa_v0_view
  int64_t v55 = v53 * v54;
  // pto: %mixes_partials_inline693__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v56 = pto::Shape<1, 1, 1, -1, -1>(v51, v52, v53, v50, v38);
  // pto: %mixes_partials_inline693__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v57 = pto::Stride<-1, -1, -1, -1, -1>(v52 * v55, v55, v54, v38, v39);
  // pto: %mixes_partials_inline693__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v58 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v56, v57);
  // pto: %hc_fn__ssa_v0_view
  const int64_t v59 = 1;
  // pto: %hc_fn__ssa_v0_view
  const int64_t v60 = 1;
  // pto: %hc_fn__ssa_v0_view
  const int64_t v61 = 1;
  // pto: %hc_fn__ssa_v0_view
  int64_t v62 = v40 * v41;
  // pto: %hc_fn__ssa_v0_view
  int64_t v63 = v61 * v62;
  // pto: %hc_fn__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v64 = pto::Shape<1, 1, 1, -1, -1>(v59, v60, v61, v40, v41);
  // pto: %hc_fn__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v65 = pto::Stride<-1, -1, -1, -1, -1>(v60 * v63, v63, v62, v41, v39);
  // pto: %hc_fn__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v66 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v64, v65);
  // pto: %mid_stream_product_inline785__phi_v10
  Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v67 = Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v38, v40);
  // pto: %mid_stream_product_inline785__phi_v10
  uint64_t v68 = (uint64_t) v35;
  TASSIGN(v67, v68);
  const int32_t v69 = 0;
  auto v70 = TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>(v13, v69, v34);
  // pto: %worker_inline757__ssa_v0
  // pto: %linear_workers_inline771__ssa_v0_idx
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  for (int64_t v71 = (int64_t) v31; v71 < v14; v71 += (int64_t) v15) {
    // pto: %12
    ;
    int64_t v72 = v71 % v42;
    // pto: %14, %15, %10, %11
    ;
    int64_t v73 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v72 * (uint64_t) v16) + (uint64_t) ((int64_t) (uint64_t) (v71 / v42) * (uint64_t) v38));
    // pto: %mid_stream_product_inline785__tile
    ;
    Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v74 = Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v38, v38);
    // pto: %mid_stream_product_inline785__tile
    ;
    uint64_t v75 = (uint64_t) v35;
    TASSIGN(v74, v75);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int64_t v76 = v44; v76 < v45; v76 += v39) {
      // pto: %13, %17, %16
      ;
      int64_t v77 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v72 * (uint64_t) v43) + (uint64_t) ((int64_t) (uint64_t) v76 * (uint64_t) v46));
      // pto: %mid_weight_chunk_inline701__tile
      ;
      Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v78 = Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v40, v46);
      // pto: %mid_weight_chunk_inline701__tile
      ;
      uint64_t v79 = (uint64_t) v36;
      TASSIGN(v78, v79);
      // pto: %18
      ;
      int64_t v80 = v77 < v44 ? v44 : v77;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v81 = 0;
      // pto: %hc_fn__ssa_v0_pview
      ;
      __gm__ float* v82 = PTOAS__GLOBAL_TENSOR_DATA(v66);
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v83 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v84 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      const int64_t v85 = 1;
      // pto: %hc_fn__ssa_v0_pview
      ;
      int64_t v86 = v40 * v41;
      // pto: %hc_fn__ssa_v0_pview
      ;
      int64_t v87 = v85 * v86;
      // pto: %hc_fn__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 24, -1> v88 = pto::Shape<1, 1, 1, 24, -1>(v83, v84, v85, v40, v46);
      // pto: %hc_fn__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v89 = pto::Stride<-1, -1, -1, -1, -1>(v84 * v87, v87, v86, v41, v39);
      // pto: %hc_fn__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v90 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v82 + (v81 + v44 * v41 + v80 * v39), v88, v89);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v78, v90);
      // pto: %mid_gathered_chunk_inline682__tile
      ;
      const int32_t v91 = 32;
      // pto: %mid_gathered_chunk_inline682__tile
      ;
      const int32_t v92 = 1024;
      // pto: %mid_gathered_chunk_inline682__tile
      ;
      Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v93 = Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v91, v92);
      pipe_barrier(PIPE_MTE2);
      TPOP<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v70, v93);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      // pto: %19
      ;
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      if (v76 == v44) {
        // pto: %mid_weight_chunk_inline701__tile_t
        ;
        Tile<TileType::Mat, float, 1024, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v94 = Tile<TileType::Mat, float, 1024, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v46, v40);
        // pto: %mid_weight_chunk_inline701__tile_t
        ;
        uint64_t v95 = (uint64_t) v36;
        TASSIGN(v94, v95);
        // pto: %mid_stream_product_inline785__tile_l0_init_storage
        ;
        Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v96 = Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v38, v38);
        // pto: %mid_stream_product_inline785__tile_l0_init_storage
        ;
        uint64_t v97 = (uint64_t) v35;
        TASSIGN(v96, v97);
        v96.SetValidShape(v38, v40);
        for (int64_t v98 = v44; v98 < v46; v98 += v47) {
          // pto: %mid_stream_product_inline785__tile_l0_a
          ;
          Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v99 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v38, v48);
          // pto: %mid_stream_product_inline785__tile_l0_a
          ;
          uint64_t v100 = (uint64_t) v35;
          TASSIGN(v99, v100);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          pipe_barrier(PIPE_MTE1);
          uint16_t v101 = static_cast<uint16_t>(v44);
          uint16_t v102 = static_cast<uint16_t>(v98);
          TEXTRACT(v99, v93, v101, v102);
          // pto: %20
          ;
          int64_t v103 = (int64_t) ((uint64_t) v46 - (uint64_t) v98);
          // pto: %21
          ;
          int64_t v104 = v103 < v48 ? v103 : v48;
          // pto: %mid_stream_product_inline785__tile_l0_b
          ;
          Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v105 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v104, v40);
          // pto: %mid_stream_product_inline785__tile_l0_b
          ;
          uint64_t v106 = (uint64_t) v35;
          TASSIGN(v105, v106);
          uint16_t v107 = static_cast<uint16_t>(v98);
          uint16_t v108 = static_cast<uint16_t>(v44);
          TEXTRACT(v105, v94, v107, v108);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          // pto: %0
          ;
          Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v109 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v38, v48);
          // pto: %0
          ;
          uint64_t v110 = (uint64_t) v37;
          TASSIGN(v109, v110);
          // pto: %22
          ;
          int64_t v111 = (int64_t) ((uint64_t) v98 + (uint64_t) v48);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
          uint16_t v112 = static_cast<uint16_t>(v44);
          uint16_t v113 = static_cast<uint16_t>(v111);
          TEXTRACT(v109, v93, v112, v113);
          int64_t v114 = (int64_t) ((uint64_t) v33 - (uint64_t) v98);
          // pto: %25
          ;
          int64_t v115 = v114 < v44 ? v44 : v114;
          // pto: %26
          ;
          int64_t v116 = v115 < v48 ? v115 : v48;
          // pto: %1
          ;
          Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v117 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v116, v40);
          // pto: %1
          ;
          uint64_t v118 = (uint64_t) v37;
          TASSIGN(v117, v118);
          uint16_t v119 = static_cast<uint16_t>(v111);
          uint16_t v120 = static_cast<uint16_t>(v44);
          TEXTRACT(v117, v94, v119, v120);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          // pto: %28
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          if (v98 == v44) {
            pipe_barrier(PIPE_M);
            TMATMUL(v96, v99, v105);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v96, v96, v99, v105);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
          // pto: %29
          ;
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
          if (v98 == v49) {
            pipe_barrier(PIPE_M);
            TMATMUL(v96, v109, v117);
          } else {
            pipe_barrier(PIPE_M);
            TMATMUL_ACC(v96, v96, v109, v117);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        };
      } else {
        // pto: %3
        ;
        Tile<TileType::Mat, float, 1024, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v121 = Tile<TileType::Mat, float, 1024, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v46, v40);
        // pto: %3
        ;
        uint64_t v122 = (uint64_t) v36;
        TASSIGN(v121, v122);
        for (int64_t v123 = v44; v123 < v46; v123 += v47) {
          // pto: %4
          ;
          Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v124 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v38, v48);
          // pto: %4
          ;
          uint64_t v125 = (uint64_t) v35;
          TASSIGN(v124, v125);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
          uint16_t v126 = static_cast<uint16_t>(v44);
          uint16_t v127 = static_cast<uint16_t>(v123);
          TEXTRACT(v124, v93, v126, v127);
          // pto: %31
          ;
          int64_t v128 = (int64_t) ((uint64_t) v46 - (uint64_t) v123);
          // pto: %32
          ;
          int64_t v129 = v128 < v48 ? v128 : v48;
          // pto: %5
          ;
          Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v130 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v129, v40);
          // pto: %5
          ;
          uint64_t v131 = (uint64_t) v35;
          TASSIGN(v130, v131);
          uint16_t v132 = static_cast<uint16_t>(v123);
          uint16_t v133 = static_cast<uint16_t>(v44);
          TEXTRACT(v130, v121, v132, v133);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
          // pto: %6
          ;
          Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v134 = Tile<TileType::Left, float, 32, 256, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v38, v48);
          // pto: %6
          ;
          uint64_t v135 = (uint64_t) v37;
          TASSIGN(v134, v135);
          // pto: %33
          ;
          int64_t v136 = (int64_t) ((uint64_t) v123 + (uint64_t) v48);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
          uint16_t v137 = static_cast<uint16_t>(v44);
          uint16_t v138 = static_cast<uint16_t>(v136);
          TEXTRACT(v134, v93, v137, v138);
          int64_t v139 = (int64_t) ((uint64_t) v33 - (uint64_t) v123);
          // pto: %36
          ;
          int64_t v140 = v139 < v44 ? v44 : v139;
          // pto: %37
          ;
          int64_t v141 = v140 < v48 ? v140 : v48;
          // pto: %7
          ;
          Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v142 = Tile<TileType::Right, float, 256, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v141, v40);
          // pto: %7
          ;
          uint64_t v143 = (uint64_t) v37;
          TASSIGN(v142, v143);
          uint16_t v144 = static_cast<uint16_t>(v136);
          uint16_t v145 = static_cast<uint16_t>(v44);
          TEXTRACT(v142, v121, v144, v145);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
          pipe_barrier(PIPE_M);
          TMATMUL_ACC(v74, v74, v124, v130);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
          pipe_barrier(PIPE_M);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
          TMATMUL_ACC(v74, v74, v134, v142);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
        };
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TFREE<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v70);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    // pto: %39
    ;
    int64_t v146 = v73 < v44 ? v44 : v73;
    // pto: %mixes_partials_inline693__iter_v1_pview
    ;
    __gm__ float* v147 = PTOAS__GLOBAL_TENSOR_DATA(v58);
    // pto: %mixes_partials_inline693__iter_v1_pview
    ;
    const int64_t v148 = 0;
    // pto: %mixes_partials_inline693__iter_v1_pview
    ;
    const int64_t v149 = 32;
    // pto: %mixes_partials_inline693__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 32, 32> v150 = pto::Shape<1, 1, 1, 32, 32>();
    // pto: %mixes_partials_inline693__iter_v1_pview
    ;
    pto::Stride<1024, 1024, 1024, 32, 1> v151 = pto::Stride<1024, 1024, 1024, 32, 1>();
    // pto: %mixes_partials_inline693__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v152 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v147 + (v148 + v146 * v149), v150, v151);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v152, v74);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
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

AICORE void hc_medium96_aiv(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ bfloat16_t* v12, __gm__ float* v13, int64_t v14, int32_t v15, int64_t v16, int64_t v17, int64_t v18, int64_t v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int64_t v28, int64_t v29, int64_t v30, int32_t v31, int32_t v32, int32_t v33) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v34 = 0;
  // pto: %c65600_i64
  const int64_t v35 = 65600;
  // pto: %c64_i64
  const int64_t v36 = 64;
  // pto: %c131136_i64
  const int64_t v37 = 131136;
  // pto: %c1088_i64
  const int64_t v38 = 1088;
  // pto: %c1344_i64
  const int64_t v39 = 1344;
  // pto: %c1152_i64
  const int64_t v40 = 1152;
  const int64_t v41 = 96;
  // pto: %c576_i64
  const int64_t v42 = 576;
  // pto: %c832_i64
  const int64_t v43 = 832;
  // pto: %c320_i64
  const int64_t v44 = 320;
  // pto: %c1600_i64
  const int64_t v45 = 1600;
  // pto: %c1856_i64
  const int64_t v46 = 1856;
  // pto: %c2112_i64
  const int64_t v47 = 2112;
  // pto: %c1888_i64
  const int64_t v48 = 1888;
  // pto: %c1920_i64
  const int64_t v49 = 1920;
  // pto: %c1952_i64
  const int64_t v50 = 1952;
  // pto: %c1120_i64
  const int64_t v51 = 1120;
  // pto: %c114752_i64
  const int64_t v52 = 114752;
  // pto: %c192_i64
  const int64_t v53 = 192;
  // pto: %c16448_i64
  const int64_t v54 = 16448;
  // pto: %c32832_i64
  const int64_t v55 = 32832;
  // pto: %c41024_i64
  const int64_t v56 = 41024;
  // pto: %c98368_i64
  const int64_t v57 = 98368;
  // pto: %c81984_i64
  const int64_t v58 = 81984;
  // pto: %c90176_i64
  const int64_t v59 = 90176;
  // pto: %c106560_i64
  const int64_t v60 = 106560;
  // pto: %c49216_i64
  const int64_t v61 = 49216;
  // pto: %c32_index
  const int64_t v62 = 32;
  // pto: %c1_index
  const int64_t v63 = 1;
  // pto: %c24_index
  const int64_t v64 = 24;
  // pto: %c16384_index
  const int64_t v65 = 16384;
  // pto: %c3_index
  const int64_t v66 = 3;
  // pto: %c8_index
  const int64_t v67 = 8;
  const int64_t v68 = 16;
  // pto: %c4096_index
  const int64_t v69 = 4096;
  // pto: %c0_i32
  const int32_t v70 = 0;
  // pto: %c2048_index
  const int64_t v71 = 2048;
  // pto: %cst_38
  const float v72 = 0.0f;
  // pto: %c0_index
  const int64_t v73 = 0;
  // pto: %c2_index
  const int64_t v74 = 2;
  // pto: %c1024_index
  const int64_t v75 = 1024;
  // pto: %c4_index
  const int64_t v76 = 4;
  // pto: %c256_index
  const int64_t v77 = 256;
  // pto: %cst_44
  const float v78 = 6.10351563E-5f;
  // pto: %cst_45
  const float v79 = 9.99999997E-7f;
  // pto: %c64_index
  const int64_t v80 = 64;
  // pto: %c192_index
  const int64_t v81 = 192;
  // pto: %cst_48
  const float v82 = 1.0f;
  // pto: %cst_49
  const float v83 = 0.5f;
  // pto: %cst_50
  const float v84 = 2.0f;
  // pto: %cst_51
  const float v85 = 0.25f;
  // pto: %cst_52
  const float v86 = 4.0f;
  // pto: %cst_53
  const float v87 = 0.125f;
  // pto: %cst_54
  const float v88 = 8.0f;
  // pto: %cst_55
  const float v89 = 0.0625f;
  // pto: %cst_56
  const float v90 = 16.0f;
  // pto: %c19_index
  const int64_t v91 = 19;
  // pto: %212
  int64_t v92 = (int64_t) ((uint64_t) v29 * (uint64_t) v67);
  // pto: %rms_partials_inline811__ssa_v0_view
  const int64_t v93 = 1;
  // pto: %rms_partials_inline811__ssa_v0_view
  const int64_t v94 = 1;
  // pto: %rms_partials_inline811__ssa_v0_view
  const int64_t v95 = 1;
  // pto: %rms_partials_inline811__ssa_v0_view
  int64_t v96 = v92 * v63;
  // pto: %rms_partials_inline811__ssa_v0_view
  int64_t v97 = v95 * v96;
  // pto: %rms_partials_inline811__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v98 = pto::Shape<1, 1, 1, -1, -1>(v93, v94, v95, v92, v63);
  // pto: %rms_partials_inline811__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v99 = pto::Stride<-1, -1, -1, -1, -1>(v94 * v97, v97, v96, v63, v92);
  // pto: %rms_partials_inline811__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v100 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v98, v99);
  // pto: %x_flat_inline744__ssa_v0_view
  const int64_t v101 = 1;
  // pto: %x_flat_inline744__ssa_v0_view
  const int64_t v102 = 1;
  // pto: %x_flat_inline744__ssa_v0_view
  const int64_t v103 = 1;
  // pto: %x_flat_inline744__ssa_v0_view
  int64_t v104 = (int64_t) v30;
  // pto: %x_flat_inline744__ssa_v0_view
  int64_t v105 = v104 * v65;
  // pto: %x_flat_inline744__ssa_v0_view
  int64_t v106 = v103 * v105;
  // pto: %x_flat_inline744__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v107 = pto::Shape<1, 1, 1, -1, -1>(v101, v102, v103, v104, v65);
  // pto: %x_flat_inline744__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v108 = pto::Stride<-1, -1, -1, -1, -1>(v102 * v106, v106, v105, v65, v63);
  // pto: %x_flat_inline744__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v109 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v107, v108);
  // pto: %213
  int64_t v110 = (int64_t) ((uint64_t) v29 * (uint64_t) v62);
  // pto: %parts_2d_inline707__ssa_v0_view
  const int64_t v111 = 1;
  // pto: %parts_2d_inline707__ssa_v0_view
  const int64_t v112 = 1;
  // pto: %parts_2d_inline707__ssa_v0_view
  const int64_t v113 = 1;
  // pto: %parts_2d_inline707__ssa_v0_view
  int64_t v114 = v67 * v110;
  // pto: %parts_2d_inline707__ssa_v0_view
  int64_t v115 = v113 * v114;
  // pto: %parts_2d_inline707__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v116 = pto::Shape<1, 1, 1, -1, -1>(v111, v112, v113, v67, v110);
  // pto: %parts_2d_inline707__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v117 = pto::Stride<-1, -1, -1, -1, -1>(v112 * v115, v115, v114, v110, v63);
  // pto: %parts_2d_inline707__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v118 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v116, v117);
  // pto: %rms_2d_inline749__ssa_v0_view
  const int64_t v119 = 1;
  // pto: %rms_2d_inline749__ssa_v0_view
  const int64_t v120 = 1;
  // pto: %rms_2d_inline749__ssa_v0_view
  const int64_t v121 = 1;
  // pto: %rms_2d_inline749__ssa_v0_view
  int64_t v122 = (int64_t) v29;
  // pto: %rms_2d_inline749__ssa_v0_view
  int64_t v123 = v67 * v122;
  // pto: %rms_2d_inline749__ssa_v0_view
  int64_t v124 = v121 * v123;
  // pto: %rms_2d_inline749__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v125 = pto::Shape<1, 1, 1, -1, -1>(v119, v120, v121, v67, (int64_t) v29);
  // pto: %rms_2d_inline749__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v126 = pto::Stride<-1, -1, -1, -1, -1>(v120 * v124, v124, v123, v122, v63);
  // pto: %rms_2d_inline749__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v127 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v125, v126);
  // pto: %hc_base_2d_inline788__ssa_v0_view
  const int64_t v128 = 1;
  // pto: %hc_base_2d_inline788__ssa_v0_view
  const int64_t v129 = 1;
  // pto: %hc_base_2d_inline788__ssa_v0_view
  const int64_t v130 = 1;
  // pto: %hc_base_2d_inline788__ssa_v0_view
  int64_t v131 = v63 * v64;
  // pto: %hc_base_2d_inline788__ssa_v0_view
  int64_t v132 = v130 * v131;
  // pto: %hc_base_2d_inline788__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v133 = pto::Shape<1, 1, 1, -1, -1>(v128, v129, v130, v63, v64);
  // pto: %hc_base_2d_inline788__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v134 = pto::Stride<-1, -1, -1, -1, -1>(v129 * v132, v132, v131, v64, v63);
  // pto: %hc_base_2d_inline788__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v135 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v8, v133, v134);
  // pto: %comb_inline769__ssa_v0_view
  const int64_t v136 = 1;
  // pto: %comb_inline769__ssa_v0_view
  const int64_t v137 = 1;
  // pto: %comb_inline769__ssa_v0_view
  const int64_t v138 = 1;
  // pto: %comb_inline769__ssa_v0_view
  int64_t v139 = (int64_t) v30;
  // pto: %comb_inline769__ssa_v0_view
  int64_t v140 = v139 * v68;
  // pto: %comb_inline769__ssa_v0_view
  int64_t v141 = v138 * v140;
  // pto: %comb_inline769__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v142 = pto::Shape<1, 1, 1, -1, -1>(v136, v137, v138, v139, v68);
  // pto: %comb_inline769__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v143 = pto::Stride<-1, -1, -1, -1, -1>(v137 * v141, v141, v140, v68, v63);
  // pto: %comb_inline769__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v144 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v9, v142, v143);
  // pto: %hc_scale_2d_inline758__ssa_v0_view
  const int64_t v145 = 1;
  // pto: %hc_scale_2d_inline758__ssa_v0_view
  const int64_t v146 = 1;
  // pto: %hc_scale_2d_inline758__ssa_v0_view
  const int64_t v147 = 1;
  // pto: %hc_scale_2d_inline758__ssa_v0_view
  int64_t v148 = v63 * v66;
  // pto: %hc_scale_2d_inline758__ssa_v0_view
  int64_t v149 = v147 * v148;
  // pto: %hc_scale_2d_inline758__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v150 = pto::Shape<1, 1, 1, -1, -1>(v145, v146, v147, v63, v66);
  // pto: %hc_scale_2d_inline758__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v151 = pto::Stride<-1, -1, -1, -1, -1>(v146 * v149, v149, v148, v66, v63);
  // pto: %hc_scale_2d_inline758__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v152 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v10, v150, v151);
  // pto: %214
  int64_t v153 = (int64_t) ((uint64_t) v30 * (uint64_t) v76);
  // pto: %post_flat_inline735__ssa_v0_view
  const int64_t v154 = 1;
  // pto: %post_flat_inline735__ssa_v0_view
  const int64_t v155 = 1;
  // pto: %post_flat_inline735__ssa_v0_view
  const int64_t v156 = 1;
  // pto: %post_flat_inline735__ssa_v0_view
  int64_t v157 = v63 * v153;
  // pto: %post_flat_inline735__ssa_v0_view
  int64_t v158 = v156 * v157;
  // pto: %post_flat_inline735__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v159 = pto::Shape<1, 1, 1, -1, -1>(v154, v155, v156, v63, v153);
  // pto: %post_flat_inline735__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v160 = pto::Stride<-1, -1, -1, -1, -1>(v155 * v158, v158, v157, v153, v63);
  // pto: %post_flat_inline735__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v161 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v11, v159, v160);
  // pto: %mixed_view_inline813__ssa_v0_view
  const int64_t v162 = 1;
  // pto: %mixed_view_inline813__ssa_v0_view
  const int64_t v163 = 1;
  // pto: %mixed_view_inline813__ssa_v0_view
  const int64_t v164 = 1;
  // pto: %mixed_view_inline813__ssa_v0_view
  int64_t v165 = (int64_t) v30;
  // pto: %mixed_view_inline813__ssa_v0_view
  int64_t v166 = v165 * v69;
  // pto: %mixed_view_inline813__ssa_v0_view
  int64_t v167 = v164 * v166;
  // pto: %mixed_view_inline813__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v168 = pto::Shape<1, 1, 1, -1, -1>(v162, v163, v164, v165, v69);
  // pto: %mixed_view_inline813__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v169 = pto::Stride<-1, -1, -1, -1, -1>(v163 * v167, v167, v166, v69, v63);
  // pto: %mixed_view_inline813__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v170 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v12, v168, v169);
  const int32_t v171 = 0;
  auto v172 = TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>(v13, v171, v70);
  // pto: %worker_inline757__ssa_v0
  int64_t v173 = (int64_t) v31;
  // pto: %linear_workers_inline771__ssa_v0_idx
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  set_flag(PIPE_S, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  for (int64_t v174 = v173; v174 < v14; v174 += (int64_t) v15) {
    // pto: %36, %37
    ;
    int64_t v175 = (int64_t) ((uint64_t) (v174 / v67) * (uint64_t) v62);
    // pto: %38
    ;
    int64_t v176 = v174 % v67;
    // pto: %mid_stream_rms_total_inline732__tile
    ;
    Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v177 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v68);
    // pto: %mid_stream_rms_total_inline732__tile
    ;
    uint64_t v178 = (uint64_t) v34;
    TASSIGN(v177, v178);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TEXPANDS(v177, v72);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    for (int64_t v179 = v73; v179 < v74; v179 += v63) {
      // pto: %39, %43, %42
      ;
      int64_t v180 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v176 * (uint64_t) v71) + (uint64_t) ((int64_t) (uint64_t) v179 * (uint64_t) v75));
      // pto: %45, %lane_inline689__ssa_v0, %44
      ;
      int64_t v181 = (int64_t) ((uint64_t) v175 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) * (uint64_t) v68));
      // pto: %46
      ;
      int64_t v182 = (int64_t) ((uint64_t) v17 - (uint64_t) v181);
      // pto: %47
      ;
      int64_t v183 = v182 < v68 ? v182 : v68;
      // pto: %48
      ;
      int64_t v184 = v183 < v73 ? v73 : v183;
      // pto: %mid_stream_source_inline839__tile
      ;
      Tile<TileType::Vec, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v185 = Tile<TileType::Vec, bfloat16_t, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v184, v75);
      // pto: %mid_stream_source_inline839__tile
      ;
      uint64_t v186 = (uint64_t) v35;
      TASSIGN(v185, v186);
      // pto: %49
      ;
      int64_t v187 = v181 < v73 ? v73 : v181;
      // pto: %50
      ;
      int64_t v188 = v180 < v73 ? v73 : v180;
      // pto: %x_flat_inline744__ssa_v0_pview
      ;
      const int64_t v189 = 0;
      // pto: %x_flat_inline744__ssa_v0_pview
      ;
      __gm__ bfloat16_t* v190 = PTOAS__GLOBAL_TENSOR_DATA(v109);
      // pto: %x_flat_inline744__ssa_v0_pview
      ;
      const int64_t v191 = 1;
      // pto: %x_flat_inline744__ssa_v0_pview
      ;
      const int64_t v192 = 1;
      // pto: %x_flat_inline744__ssa_v0_pview
      ;
      const int64_t v193 = 1;
      // pto: %x_flat_inline744__ssa_v0_pview
      ;
      int64_t v194 = v184 * v65;
      // pto: %x_flat_inline744__ssa_v0_pview
      ;
      int64_t v195 = v193 * v194;
      // pto: %x_flat_inline744__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, -1> v196 = pto::Shape<1, 1, 1, -1, -1>(v191, v192, v193, v184, v75);
      // pto: %x_flat_inline744__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v197 = pto::Stride<-1, -1, -1, -1, -1>(v192 * v195, v195, v194, v65, v63);
      // pto: %x_flat_inline744__ssa_v0_pview
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v198 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v190 + (v189 + v187 * v65 + v188 * v63), v196, v197);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v185, v198);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      // pto: %mid_stream_widened_inline692__tile
      ;
      Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v199 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v184, v75);
      // pto: %mid_stream_widened_inline692__tile
      ;
      uint64_t v200 = (uint64_t) v36;
      TASSIGN(v199, v200);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      RoundMode v201 = RoundMode::CAST_ROUND;
      SaturationMode v202 = SaturationMode::ON;
      TCVT(v199, v185, v201, v202);
      v199.SetValidShape(v184, v75);
      // pto: %mid_stream_clean_inline691__tile
      ;
      Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v203 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v68, v75);
      // pto: %mid_stream_clean_inline691__tile
      ;
      uint64_t v204 = (uint64_t) v36;
      TASSIGN(v203, v204);
      pipe_barrier(PIPE_V);
      TFILLPAD<pto::TFillPadMode::InPlace>(v203, v199);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      v203.SetValidShape(v68, v75);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TPUSH<TPipe<0, Direction::DIR_V2C, 131072, 2, 2, false>, Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v172, v203);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      // pto: %0
      ;
      Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v205 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v68, v75);
      // pto: %0
      ;
      uint64_t v206 = (uint64_t) v36;
      TASSIGN(v205, v206);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
      TMUL(v205, v203, v203);
      // pto: %tmp_tile
      ;
      Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v207 = Tile<TileType::Vec, float, 16, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v75);
      // pto: %tmp_tile
      ;
      uint64_t v208 = (uint64_t) v35;
      TASSIGN(v207, v208);
      // pto: %1
      ;
      Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v209 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v63);
      // pto: %1
      ;
      uint64_t v210 = (uint64_t) v37;
      TASSIGN(v209, v210);
      pipe_barrier(PIPE_V);
      TROWSUM(v209, v205, v207);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      // pto: %mid_stream_sq_inline685__tile
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v211 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v68);
      // pto: %mid_stream_sq_inline685__tile
      ;
      uint64_t v212 = (uint64_t) v37;
      TASSIGN(v211, v212);
      // pto: %2
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v213 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v68);
      // pto: %2
      ;
      uint64_t v214 = (uint64_t) v34;
      TASSIGN(v213, v214);
      pipe_barrier(PIPE_V);
      TADD(v213, v177, v211);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %40, %41, %52, %rms_lane_inline680__ssa_v0, %51
    ;
    int64_t v215 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v176 * (uint64_t) v16) + (uint64_t) v175) + (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) * (uint64_t) v68));
    // pto: %3
    ;
    Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v216 = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v63);
    // pto: %3
    ;
    uint64_t v217 = (uint64_t) v34;
    TASSIGN(v216, v217);
    // pto: %53
    ;
    int64_t v218 = v215 < v73 ? v73 : v215;
    // pto: %rms_partials_inline811__iter_v1_pview
    ;
    const int64_t v219 = 0;
    // pto: %rms_partials_inline811__iter_v1_pview
    ;
    __gm__ float* v220 = PTOAS__GLOBAL_TENSOR_DATA(v100);
    // pto: %rms_partials_inline811__iter_v1_pview
    ;
    const int64_t v221 = 1;
    // pto: %rms_partials_inline811__iter_v1_pview
    ;
    const int64_t v222 = 1;
    // pto: %rms_partials_inline811__iter_v1_pview
    ;
    const int64_t v223 = 1;
    // pto: %rms_partials_inline811__iter_v1_pview
    ;
    int64_t v224 = v68 * v63;
    // pto: %rms_partials_inline811__iter_v1_pview
    ;
    int64_t v225 = v223 * v224;
    // pto: %rms_partials_inline811__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 16, 1> v226 = pto::Shape<1, 1, 1, 16, 1>(v221, v222, v223, v68, v63);
    // pto: %rms_partials_inline811__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v227 = pto::Stride<-1, -1, -1, -1, -1>(v222 * v225, v225, v224, v63, v92);
    // pto: %rms_partials_inline811__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v228 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v220 + (v219 + v218 * v63 + v73 * v92), v226, v227);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    TSTORE(v228, v216);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  pipe_barrier(PIPE_ALL);
  dsb(DSB_DDR);
  SYNCALL<SyncCoreType::Mix>();
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  // pto: %54, %55, %tail_lane_inline766__ssa_v0
  int64_t v229 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v173 * (uint64_t) v74) + (uint64_t) ((int64_t) v33));
  // pto: %56
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  if (v229 < v18) {
    // pto: %58
    ;
    int64_t v230 = v229 < v20 ? v229 : v20;
    // pto: %57, %59
    ;
    int64_t v231 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v229 * (uint64_t) v19) + (uint64_t) v230);
    // pto: %61
    ;
    int64_t v232 = (int64_t) ((uint64_t) v20 - (uint64_t) v229);
    // pto: %62
    ;
    int64_t v233 = v232 < v73 ? v73 : v232;
    // pto: %63
    ;
    int64_t v234 = v233 < v63 ? v233 : v63;
    // pto: %60, %64
    ;
    int64_t v235 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v231 + (uint64_t) v19) + (uint64_t) v234);
    for (int64_t v236 = v231; v236 < v235; v236 += v76) {
      // pto: %scale2_inline818__tile
      ;
      float v237 = (v5)[v74];
      // pto: %65
      ;
      int64_t v238 = (int64_t) ((uint64_t) v235 - (uint64_t) v236);
      // pto: %66
      ;
      int64_t v239 = v238 < v76 ? v238 : v76;
      // pto: %67
      ;
      int64_t v240 = (int64_t) ((uint64_t) v239 * (uint64_t) v62);
      // pto: %c_parts_load_inline668__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v241 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v240);
      // pto: %c_parts_load_inline668__ssa_v0
      ;
      uint64_t v242 = (uint64_t) v36;
      TASSIGN(v241, v242);
      // pto: %68
      ;
      int64_t v243 = (int64_t) ((uint64_t) v236 * (uint64_t) v62);
      // pto: %69
      ;
      int64_t v244 = v243 < v73 ? v73 : v243;
      // pto: %parts_2d_inline707__ssa_v0_pview
      ;
      const int64_t v245 = 0;
      // pto: %parts_2d_inline707__ssa_v0_pview
      ;
      __gm__ float* v246 = PTOAS__GLOBAL_TENSOR_DATA(v118);
      // pto: %parts_2d_inline707__ssa_v0_pview
      ;
      const int64_t v247 = 1;
      // pto: %parts_2d_inline707__ssa_v0_pview
      ;
      const int64_t v248 = 1;
      // pto: %parts_2d_inline707__ssa_v0_pview
      ;
      const int64_t v249 = 1;
      // pto: %parts_2d_inline707__ssa_v0_pview
      ;
      int64_t v250 = v67 * v110;
      // pto: %parts_2d_inline707__ssa_v0_pview
      ;
      int64_t v251 = v249 * v250;
      // pto: %parts_2d_inline707__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, -1> v252 = pto::Shape<1, 1, 1, 8, -1>(v247, v248, v249, v67, v240);
      // pto: %parts_2d_inline707__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v253 = pto::Stride<-1, -1, -1, -1, -1>(v248 * v251, v251, v250, v110, v63);
      // pto: %parts_2d_inline707__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v254 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v246 + (v245 + v73 * v110 + v244 * v63), v252, v253);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(v241, v254);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %t__tmp_v191
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v255 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v67, v77);
      // pto: %t__tmp_v191
      ;
      uint64_t v256 = (uint64_t) v36;
      TASSIGN(v255, v256);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TFILLPAD<pto::TFillPadMode::InPlace>(v255, v241);
      v255.SetValidShape(v67, v77);
      // pto: %t__tmp_v192
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v257 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v77);
      // pto: %t__tmp_v192
      ;
      uint64_t v258 = (uint64_t) v36;
      TASSIGN(v257, v258);
      pipe_barrier(PIPE_V);
      TCOLSUM(v257, v255);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
      // pto: %c_total_inline866__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v259 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v62);
      // pto: %c_total_inline866__ssa_v0
      ;
      uint64_t v260 = (uint64_t) v36;
      TASSIGN(v259, v260);
      // pto: %c_sq_load_inline721__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v261 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v239);
      // pto: %c_sq_load_inline721__ssa_v0
      ;
      uint64_t v262 = (uint64_t) v38;
      TASSIGN(v261, v262);
      // pto: %71
      ;
      int64_t v263 = v236 < v73 ? v73 : v236;
      // pto: %rms_2d_inline749__ssa_v0_pview
      ;
      int64_t v264 = (int64_t) v29;
      // pto: %rms_2d_inline749__ssa_v0_pview
      ;
      const int64_t v265 = 0;
      // pto: %rms_2d_inline749__ssa_v0_pview
      ;
      __gm__ float* v266 = PTOAS__GLOBAL_TENSOR_DATA(v127);
      // pto: %rms_2d_inline749__ssa_v0_pview
      ;
      const int64_t v267 = 1;
      // pto: %rms_2d_inline749__ssa_v0_pview
      ;
      const int64_t v268 = 1;
      // pto: %rms_2d_inline749__ssa_v0_pview
      ;
      const int64_t v269 = 1;
      // pto: %rms_2d_inline749__ssa_v0_pview
      ;
      int64_t v270 = v67 * v264;
      // pto: %rms_2d_inline749__ssa_v0_pview
      ;
      int64_t v271 = v269 * v270;
      // pto: %rms_2d_inline749__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 8, -1> v272 = pto::Shape<1, 1, 1, 8, -1>(v267, v268, v269, v67, v239);
      // pto: %rms_2d_inline749__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v273 = pto::Stride<-1, -1, -1, -1, -1>(v268 * v271, v271, v270, v264, v63);
      // pto: %rms_2d_inline749__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v274 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v266 + (v265 + v73 * v264 + v263 * v63), v272, v273);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
      TLOAD(v261, v274);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %t__tmp_v193
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v275 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v67, v67);
      // pto: %t__tmp_v193
      ;
      uint64_t v276 = (uint64_t) v38;
      TASSIGN(v275, v276);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      TFILLPAD<pto::TFillPadMode::InPlace>(v275, v261);
      v275.SetValidShape(v67, v67);
      // pto: %c_rms_tmp_inline723__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v277 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v67);
      // pto: %c_rms_tmp_inline723__ssa_v0
      ;
      uint64_t v278 = (uint64_t) v39;
      TASSIGN(v277, v278);
      // pto: %t__tmp_v194
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v279 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v67);
      // pto: %t__tmp_v194
      ;
      uint64_t v280 = (uint64_t) v38;
      TASSIGN(v279, v280);
      pipe_barrier(PIPE_V);
      TCOLSUM(v279, v275);
      // pto: %t__tmp_v195
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v281 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v67);
      // pto: %t__tmp_v195
      ;
      uint64_t v282 = (uint64_t) v38;
      TASSIGN(v281, v282);
      pipe_barrier(PIPE_V);
      TMULS(v281, v279, v78);
      // pto: %t__tmp_v196
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v283 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v67);
      // pto: %t__tmp_v196
      ;
      uint64_t v284 = (uint64_t) v38;
      TASSIGN(v283, v284);
      pipe_barrier(PIPE_V);
      TADDS(v283, v281, v79);
      // pto: %t__tmp_v197
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v285 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v67);
      // pto: %t__tmp_v197
      ;
      uint64_t v286 = (uint64_t) v40;
      TASSIGN(v285, v286);
      pipe_barrier(PIPE_V);
      TRSQRT(v285, v283, v277);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      // pto: %c_inv_inline716__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v287 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v63);
      // pto: %c_inv_inline716__ssa_v0
      ;
      uint64_t v288 = (uint64_t) v40;
      TASSIGN(v287, v288);
      // pto: %comb_inv_inline751__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v289 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v63);
      // pto: %comb_inv_inline751__ssa_v0
      ;
      uint64_t v290 = (uint64_t) v40;
      TASSIGN(v289, v290);
      // pto: %t__tmp_v198
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v291 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v68);
      // pto: %t__tmp_v198
      ;
      uint64_t v292 = (uint64_t) v41;
      TASSIGN(v291, v292);
      // pto: %slice_view
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v293;
      // pto: %slice_view
      ;
      uint64_t v294 = (uint64_t) v41;
      TASSIGN(v293, v294);
      // pto: %raw_inline851__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v295 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v68);
      // pto: %raw_inline851__ssa_v0
      ;
      uint64_t v296 = (uint64_t) v36;
      TASSIGN(v295, v296);
      TADDS(v295, v293, v72);
      // pto: %base_inline825__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v297 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v68);
      // pto: %base_inline825__ssa_v0
      ;
      uint64_t v298 = (uint64_t) v38;
      TASSIGN(v297, v298);
      // pto: %hc_base_2d_inline788__ssa_v0_pview
      ;
      __gm__ float* v299 = PTOAS__GLOBAL_TENSOR_DATA(v135);
      // pto: %hc_base_2d_inline788__ssa_v0_pview
      ;
      unsigned v300 = 8;
      // pto: %hc_base_2d_inline788__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 16> v301 = pto::Shape<1, 1, 1, 1, 16>();
      // pto: %hc_base_2d_inline788__ssa_v0_pview
      ;
      pto::Stride<24, 24, 24, 24, 1> v302 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %hc_base_2d_inline788__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v303 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v299 + v300, v301, v302);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v297, v303);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %t__tmp_v199
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v304 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v68);
      // pto: %t__tmp_v199
      ;
      uint64_t v305 = (uint64_t) v42;
      TASSIGN(v304, v305);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v304, v295, v289);
      // pto: %t__tmp_v200
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v306 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v68);
      // pto: %t__tmp_v200
      ;
      uint64_t v307 = (uint64_t) v42;
      TASSIGN(v306, v307);
      pipe_barrier(PIPE_V);
      TMULS(v306, v304, v237);
      // pto: %t__tmp_v201
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v308 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v68);
      // pto: %t__tmp_v201
      ;
      uint64_t v309 = (uint64_t) v36;
      TASSIGN(v308, v309);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      TCOLEXPAND(v308, v297);
      // pto: %logits_inline764__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v310 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v68);
      // pto: %logits_inline764__ssa_v0
      ;
      uint64_t v311 = (uint64_t) v43;
      TASSIGN(v310, v311);
      pipe_barrier(PIPE_V);
      TADD(v310, v306, v308);
      // pto: %t__tmp_v202
      ;
      Tile<TileType::Vec, float, 4, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v312 = Tile<TileType::Vec, float, 4, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v68);
      // pto: %t__tmp_v202
      ;
      uint64_t v313 = (uint64_t) v43;
      TASSIGN(v312, v313);
      // pto: %regular_logits_inline882__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v314 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %regular_logits_inline882__ssa_v0
      ;
      uint64_t v315 = (uint64_t) v43;
      TASSIGN(v314, v315);
      // pto: %ids_inline823__ci_tmp_v0
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v316 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v81);
      // pto: %ids_inline823__ci_tmp_v0
      ;
      uint64_t v317 = (uint64_t) v36;
      TASSIGN(v316, v317);
      // pto: %ids_inline823__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v318 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %ids_inline823__ssa_v1
      ;
      uint64_t v319 = (uint64_t) v39;
      TASSIGN(v318, v319);
      // pto: %ci_tmp_view
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v320;
      TRESHAPE(v320, v316);
      // pto: %ci_dst_view
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v321;
      TRESHAPE(v321, v318);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v321, v70, v320);
      // pto: %tmp_inline828__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v322 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %tmp_inline828__ssa_v1
      ;
      uint64_t v323 = (uint64_t) v36;
      TASSIGN(v322, v323);
      // pto: %idx_float_inline883__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v324 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %idx_float_inline883__ssa_v0
      ;
      uint64_t v325 = (uint64_t) v44;
      TASSIGN(v324, v325);
      pipe_barrier(PIPE_V);
      RoundMode v326 = RoundMode::CAST_ROUND;
      SaturationMode v327 = SaturationMode::ON;
      TCVT(v324, v318, v326, v327);
      // pto: %t__tmp_v203
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v328 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v203
      ;
      uint64_t v329 = (uint64_t) v42;
      TASSIGN(v328, v329);
      pipe_barrier(PIPE_V);
      TMULS(v328, v324, v82);
      // pto: %t__tmp_v204
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v330 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v204
      ;
      uint64_t v331 = (uint64_t) v42;
      TASSIGN(v330, v331);
      pipe_barrier(PIPE_V);
      RoundMode v332 = RoundMode::CAST_FLOOR;
      SaturationMode v333 = SaturationMode::ON;
      TCVT(v330, v328, v332, v333);
      // pto: %ri1_low_inline711__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v334 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %ri1_low_inline711__ssa_v0
      ;
      uint64_t v335 = (uint64_t) v42;
      TASSIGN(v334, v335);
      pipe_barrier(PIPE_V);
      RoundMode v336 = RoundMode::CAST_ROUND;
      SaturationMode v337 = SaturationMode::ON;
      TCVT(v334, v330, v336, v337);
      // pto: %t__tmp_v205
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v338 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v205
      ;
      uint64_t v339 = (uint64_t) v39;
      TASSIGN(v338, v339);
      TMULS(v338, v324, v83);
      // pto: %t__tmp_v206
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v340 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v206
      ;
      uint64_t v341 = (uint64_t) v39;
      TASSIGN(v340, v341);
      pipe_barrier(PIPE_V);
      RoundMode v342 = RoundMode::CAST_FLOOR;
      SaturationMode v343 = SaturationMode::ON;
      TCVT(v340, v338, v342, v343);
      // pto: %ri1_high_inline884__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v344 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %ri1_high_inline884__ssa_v0
      ;
      uint64_t v345 = (uint64_t) v39;
      TASSIGN(v344, v345);
      pipe_barrier(PIPE_V);
      RoundMode v346 = RoundMode::CAST_ROUND;
      SaturationMode v347 = SaturationMode::ON;
      TCVT(v344, v340, v346, v347);
      // pto: %t__tmp_v207
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v348 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v207
      ;
      uint64_t v349 = (uint64_t) v39;
      TASSIGN(v348, v349);
      pipe_barrier(PIPE_V);
      TMULS(v348, v344, v84);
      // pto: %ri1_bit_inline747__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v350 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %ri1_bit_inline747__ssa_v0
      ;
      uint64_t v351 = (uint64_t) v42;
      TASSIGN(v350, v351);
      pipe_barrier(PIPE_V);
      TSUB(v350, v334, v348);
      // pto: %t__tmp_v208
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v352 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v208
      ;
      uint64_t v353 = (uint64_t) v42;
      TASSIGN(v352, v353);
      pipe_barrier(PIPE_V);
      TMULS(v352, v350, v84);
      // pto: %t__tmp_v209
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v354 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v209
      ;
      uint64_t v355 = (uint64_t) v42;
      TASSIGN(v354, v355);
      pipe_barrier(PIPE_V);
      TSUB(v354, v324, v352);
      // pto: %t__tmp_v210
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v356 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v210
      ;
      uint64_t v357 = (uint64_t) v42;
      TASSIGN(v356, v357);
      pipe_barrier(PIPE_V);
      TADDS(v356, v354, v82);
      // pto: %ri1_inline726__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v358 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %ri1_inline726__ssa_v1
      ;
      uint64_t v359 = (uint64_t) v42;
      TASSIGN(v358, v359);
      pipe_barrier(PIPE_V);
      RoundMode v360 = RoundMode::CAST_ROUND;
      SaturationMode v361 = SaturationMode::ON;
      TCVT(v358, v356, v360, v361);
      // pto: %t__tmp_v211
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v362 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v211
      ;
      uint64_t v363 = (uint64_t) v39;
      TASSIGN(v362, v363);
      TMULS(v362, v324, v83);
      // pto: %t__tmp_v212
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v364 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v212
      ;
      uint64_t v365 = (uint64_t) v39;
      TASSIGN(v364, v365);
      pipe_barrier(PIPE_V);
      RoundMode v366 = RoundMode::CAST_FLOOR;
      SaturationMode v367 = SaturationMode::ON;
      TCVT(v364, v362, v366, v367);
      // pto: %ri2_low_inline885__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v368 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %ri2_low_inline885__ssa_v0
      ;
      uint64_t v369 = (uint64_t) v39;
      TASSIGN(v368, v369);
      pipe_barrier(PIPE_V);
      RoundMode v370 = RoundMode::CAST_ROUND;
      SaturationMode v371 = SaturationMode::ON;
      TCVT(v368, v364, v370, v371);
      // pto: %t__tmp_v213
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v372 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v213
      ;
      uint64_t v373 = (uint64_t) v45;
      TASSIGN(v372, v373);
      TMULS(v372, v324, v85);
      // pto: %t__tmp_v214
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v374 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v214
      ;
      uint64_t v375 = (uint64_t) v45;
      TASSIGN(v374, v375);
      pipe_barrier(PIPE_V);
      RoundMode v376 = RoundMode::CAST_FLOOR;
      SaturationMode v377 = SaturationMode::ON;
      TCVT(v374, v372, v376, v377);
      // pto: %ri2_high_inline888__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v378 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %ri2_high_inline888__ssa_v0
      ;
      uint64_t v379 = (uint64_t) v45;
      TASSIGN(v378, v379);
      pipe_barrier(PIPE_V);
      RoundMode v380 = RoundMode::CAST_ROUND;
      SaturationMode v381 = SaturationMode::ON;
      TCVT(v378, v374, v380, v381);
      // pto: %t__tmp_v215
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v382 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v215
      ;
      uint64_t v383 = (uint64_t) v45;
      TASSIGN(v382, v383);
      pipe_barrier(PIPE_V);
      TMULS(v382, v378, v84);
      // pto: %ri2_bit_inline896__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v384 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %ri2_bit_inline896__ssa_v0
      ;
      uint64_t v385 = (uint64_t) v39;
      TASSIGN(v384, v385);
      pipe_barrier(PIPE_V);
      TSUB(v384, v368, v382);
      // pto: %t__tmp_v216
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v386 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v216
      ;
      uint64_t v387 = (uint64_t) v39;
      TASSIGN(v386, v387);
      pipe_barrier(PIPE_V);
      TMULS(v386, v384, v86);
      // pto: %t__tmp_v217
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v388 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v217
      ;
      uint64_t v389 = (uint64_t) v39;
      TASSIGN(v388, v389);
      pipe_barrier(PIPE_V);
      TSUB(v388, v324, v386);
      // pto: %t__tmp_v218
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v390 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v218
      ;
      uint64_t v391 = (uint64_t) v39;
      TASSIGN(v390, v391);
      pipe_barrier(PIPE_V);
      TADDS(v390, v388, v84);
      // pto: %ri2_inline897__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v392 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %ri2_inline897__ssa_v0
      ;
      uint64_t v393 = (uint64_t) v39;
      TASSIGN(v392, v393);
      pipe_barrier(PIPE_V);
      RoundMode v394 = RoundMode::CAST_ROUND;
      SaturationMode v395 = SaturationMode::ON;
      TCVT(v392, v390, v394, v395);
      // pto: %t__tmp_v219
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v396 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v219
      ;
      uint64_t v397 = (uint64_t) v45;
      TASSIGN(v396, v397);
      TMULS(v396, v324, v85);
      // pto: %t__tmp_v220
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v398 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v220
      ;
      uint64_t v399 = (uint64_t) v45;
      TASSIGN(v398, v399);
      pipe_barrier(PIPE_V);
      RoundMode v400 = RoundMode::CAST_FLOOR;
      SaturationMode v401 = SaturationMode::ON;
      TCVT(v398, v396, v400, v401);
      // pto: %ci1_low_inline898__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v402 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %ci1_low_inline898__ssa_v0
      ;
      uint64_t v403 = (uint64_t) v45;
      TASSIGN(v402, v403);
      pipe_barrier(PIPE_V);
      RoundMode v404 = RoundMode::CAST_ROUND;
      SaturationMode v405 = SaturationMode::ON;
      TCVT(v402, v398, v404, v405);
      // pto: %t__tmp_v221
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v406 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v221
      ;
      uint64_t v407 = (uint64_t) v46;
      TASSIGN(v406, v407);
      TMULS(v406, v324, v87);
      // pto: %t__tmp_v222
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v408 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v222
      ;
      uint64_t v409 = (uint64_t) v46;
      TASSIGN(v408, v409);
      pipe_barrier(PIPE_V);
      RoundMode v410 = RoundMode::CAST_FLOOR;
      SaturationMode v411 = SaturationMode::ON;
      TCVT(v408, v406, v410, v411);
      // pto: %ci1_high_inline899__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v412 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %ci1_high_inline899__ssa_v0
      ;
      uint64_t v413 = (uint64_t) v46;
      TASSIGN(v412, v413);
      pipe_barrier(PIPE_V);
      RoundMode v414 = RoundMode::CAST_ROUND;
      SaturationMode v415 = SaturationMode::ON;
      TCVT(v412, v408, v414, v415);
      // pto: %t__tmp_v223
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v416 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v223
      ;
      uint64_t v417 = (uint64_t) v46;
      TASSIGN(v416, v417);
      pipe_barrier(PIPE_V);
      TMULS(v416, v412, v84);
      // pto: %ci1_bit_inline900__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v418 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %ci1_bit_inline900__ssa_v0
      ;
      uint64_t v419 = (uint64_t) v45;
      TASSIGN(v418, v419);
      pipe_barrier(PIPE_V);
      TSUB(v418, v402, v416);
      // pto: %t__tmp_v224
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v420 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v224
      ;
      uint64_t v421 = (uint64_t) v45;
      TASSIGN(v420, v421);
      pipe_barrier(PIPE_V);
      TMULS(v420, v418, v88);
      // pto: %t__tmp_v225
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v422 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v225
      ;
      uint64_t v423 = (uint64_t) v45;
      TASSIGN(v422, v423);
      pipe_barrier(PIPE_V);
      TSUB(v422, v324, v420);
      // pto: %t__tmp_v226
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v424 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v226
      ;
      uint64_t v425 = (uint64_t) v45;
      TASSIGN(v424, v425);
      pipe_barrier(PIPE_V);
      TADDS(v424, v422, v86);
      // pto: %ci1_inline902__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v426 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %ci1_inline902__ssa_v0
      ;
      uint64_t v427 = (uint64_t) v45;
      TASSIGN(v426, v427);
      pipe_barrier(PIPE_V);
      RoundMode v428 = RoundMode::CAST_ROUND;
      SaturationMode v429 = SaturationMode::ON;
      TCVT(v426, v424, v428, v429);
      // pto: %t__tmp_v227
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v430 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v227
      ;
      uint64_t v431 = (uint64_t) v46;
      TASSIGN(v430, v431);
      TMULS(v430, v324, v87);
      // pto: %t__tmp_v228
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v432 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v228
      ;
      uint64_t v433 = (uint64_t) v46;
      TASSIGN(v432, v433);
      pipe_barrier(PIPE_V);
      RoundMode v434 = RoundMode::CAST_FLOOR;
      SaturationMode v435 = SaturationMode::ON;
      TCVT(v432, v430, v434, v435);
      // pto: %ci2_low_inline879__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v436 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %ci2_low_inline879__ssa_v0
      ;
      uint64_t v437 = (uint64_t) v46;
      TASSIGN(v436, v437);
      pipe_barrier(PIPE_V);
      RoundMode v438 = RoundMode::CAST_ROUND;
      SaturationMode v439 = SaturationMode::ON;
      TCVT(v436, v432, v438, v439);
      // pto: %t__tmp_v229
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v440 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v229
      ;
      uint64_t v441 = (uint64_t) v47;
      TASSIGN(v440, v441);
      TMULS(v440, v324, v89);
      // pto: %t__tmp_v230
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v442 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v230
      ;
      uint64_t v443 = (uint64_t) v47;
      TASSIGN(v442, v443);
      pipe_barrier(PIPE_V);
      RoundMode v444 = RoundMode::CAST_FLOOR;
      SaturationMode v445 = SaturationMode::ON;
      TCVT(v442, v440, v444, v445);
      // pto: %ci2_high_inline903__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v446 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %ci2_high_inline903__ssa_v0
      ;
      uint64_t v447 = (uint64_t) v47;
      TASSIGN(v446, v447);
      pipe_barrier(PIPE_V);
      RoundMode v448 = RoundMode::CAST_ROUND;
      SaturationMode v449 = SaturationMode::ON;
      TCVT(v446, v442, v448, v449);
      // pto: %t__tmp_v231
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v450 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v231
      ;
      uint64_t v451 = (uint64_t) v47;
      TASSIGN(v450, v451);
      pipe_barrier(PIPE_V);
      TMULS(v450, v446, v84);
      // pto: %ci2_bit_inline890__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v452 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %ci2_bit_inline890__ssa_v0
      ;
      uint64_t v453 = (uint64_t) v46;
      TASSIGN(v452, v453);
      pipe_barrier(PIPE_V);
      TSUB(v452, v436, v450);
      // pto: %t__tmp_v232
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v454 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v232
      ;
      uint64_t v455 = (uint64_t) v46;
      TASSIGN(v454, v455);
      pipe_barrier(PIPE_V);
      TMULS(v454, v452, v90);
      // pto: %t__tmp_v233
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v456 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v233
      ;
      uint64_t v457 = (uint64_t) v44;
      TASSIGN(v456, v457);
      pipe_barrier(PIPE_V);
      TSUB(v456, v324, v454);
      // pto: %t__tmp_v234
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v458 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v234
      ;
      uint64_t v459 = (uint64_t) v44;
      TASSIGN(v458, v459);
      pipe_barrier(PIPE_V);
      TADDS(v458, v456, v88);
      // pto: %ci2_inline867__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v460 = Tile<TileType::Vec, int32_t, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %ci2_inline867__ssa_v0
      ;
      uint64_t v461 = (uint64_t) v44;
      TASSIGN(v460, v461);
      pipe_barrier(PIPE_V);
      RoundMode v462 = RoundMode::CAST_ROUND;
      SaturationMode v463 = SaturationMode::ON;
      TCVT(v460, v458, v462, v463);
      // pto: %t__tmp_v235
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v464 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v235
      ;
      uint64_t v465 = (uint64_t) v46;
      TASSIGN(v464, v465);
      TGATHER(v464, v314, v358, v322);
      // pto: %max2_inline894__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v466 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %max2_inline894__ssa_v1
      ;
      uint64_t v467 = (uint64_t) v46;
      TASSIGN(v466, v467);
      pipe_barrier(PIPE_V);
      TMAX(v466, v314, v464);
      // pto: %t__tmp_v236
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v468 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v236
      ;
      uint64_t v469 = (uint64_t) v47;
      TASSIGN(v468, v469);
      pipe_barrier(PIPE_V);
      TGATHER(v468, v466, v392, v322);
      // pto: %max4_inline863__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v470 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %max4_inline863__ssa_v1
      ;
      uint64_t v471 = (uint64_t) v46;
      TASSIGN(v470, v471);
      pipe_barrier(PIPE_V);
      TMAX(v470, v466, v468);
      // pto: %t__tmp_v237
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v472 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v237
      ;
      uint64_t v473 = (uint64_t) v43;
      TASSIGN(v472, v473);
      pipe_barrier(PIPE_V);
      TSUB(v472, v314, v470);
      // pto: %exps_inline745__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v474 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %exps_inline745__ssa_v1
      ;
      uint64_t v475 = (uint64_t) v43;
      TASSIGN(v474, v475);
      pipe_barrier(PIPE_V);
      TEXP(v474, v472);
      // pto: %t__tmp_v238
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v476 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v238
      ;
      uint64_t v477 = (uint64_t) v38;
      TASSIGN(v476, v477);
      pipe_barrier(PIPE_V);
      TGATHER(v476, v474, v358, v322);
      // pto: %sum2_inline753__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v478 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %sum2_inline753__ssa_v1
      ;
      uint64_t v479 = (uint64_t) v38;
      TASSIGN(v478, v479);
      pipe_barrier(PIPE_V);
      TADD(v478, v474, v476);
      // pto: %t__tmp_v239
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v480 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v239
      ;
      uint64_t v481 = (uint64_t) v46;
      TASSIGN(v480, v481);
      pipe_barrier(PIPE_V);
      TGATHER(v480, v478, v392, v322);
      // pto: %sum4_inline870__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v482 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %sum4_inline870__ssa_v1
      ;
      uint64_t v483 = (uint64_t) v38;
      TASSIGN(v482, v483);
      pipe_barrier(PIPE_V);
      TADD(v482, v478, v480);
      // pto: %t__tmp_v240
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v484 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v240
      ;
      uint64_t v485 = (uint64_t) v43;
      TASSIGN(v484, v485);
      pipe_barrier(PIPE_V);
      TDIV(v484, v474, v482);
      // pto: %probs_inline871__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v486 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %probs_inline871__ssa_v1
      ;
      uint64_t v487 = (uint64_t) v43;
      TASSIGN(v486, v487);
      pipe_barrier(PIPE_V);
      TADDS(v486, v484, v79);
      // pto: %t__tmp_v241
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v488 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v241
      ;
      uint64_t v489 = (uint64_t) v38;
      TASSIGN(v488, v489);
      pipe_barrier(PIPE_V);
      TGATHER(v488, v486, v426, v322);
      // pto: %col2_inline872__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v490 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %col2_inline872__ssa_v4
      ;
      uint64_t v491 = (uint64_t) v38;
      TASSIGN(v490, v491);
      pipe_barrier(PIPE_V);
      TADD(v490, v486, v488);
      // pto: %t__tmp_v242
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v492 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v242
      ;
      uint64_t v493 = (uint64_t) v46;
      TASSIGN(v492, v493);
      pipe_barrier(PIPE_V);
      TGATHER(v492, v490, v460, v322);
      // pto: %t__tmp_v243
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v494 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v243
      ;
      uint64_t v495 = (uint64_t) v38;
      TASSIGN(v494, v495);
      pipe_barrier(PIPE_V);
      TADD(v494, v490, v492);
      // pto: %col4_inline806__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v496 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %col4_inline806__ssa_v4
      ;
      uint64_t v497 = (uint64_t) v38;
      TASSIGN(v496, v497);
      pipe_barrier(PIPE_V);
      TADDS(v496, v494, v79);
      // pto: %cur_inline670__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v498 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %cur_inline670__ssa_v4
      ;
      uint64_t v499 = (uint64_t) v43;
      TASSIGN(v498, v499);
      pipe_barrier(PIPE_V);
      TDIV(v498, v486, v496);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      for (int64_t v500 = v73; v500 < v91; v500 += v63) {
        // pto: %t__tmp_v244
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v501 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
        // pto: %t__tmp_v244
        ;
        uint64_t v502 = (uint64_t) v38;
        TASSIGN(v501, v502);
        pipe_barrier(PIPE_V);
        TGATHER(v501, v498, v358, v322);
        // pto: %row2_inline904__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v503 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
        // pto: %row2_inline904__ssa_v1
        ;
        uint64_t v504 = (uint64_t) v38;
        TASSIGN(v503, v504);
        pipe_barrier(PIPE_V);
        TADD(v503, v498, v501);
        // pto: %t__tmp_v245
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v505 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
        // pto: %t__tmp_v245
        ;
        uint64_t v506 = (uint64_t) v46;
        TASSIGN(v505, v506);
        pipe_barrier(PIPE_V);
        TGATHER(v505, v503, v392, v322);
        // pto: %t__tmp_v246
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v507 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
        // pto: %t__tmp_v246
        ;
        uint64_t v508 = (uint64_t) v38;
        TASSIGN(v507, v508);
        pipe_barrier(PIPE_V);
        TADD(v507, v503, v505);
        // pto: %row4_inline878__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v509 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
        // pto: %row4_inline878__ssa_v1
        ;
        uint64_t v510 = (uint64_t) v38;
        TASSIGN(v509, v510);
        pipe_barrier(PIPE_V);
        TADDS(v509, v507, v79);
        // pto: %norm_inline800__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v511 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
        // pto: %norm_inline800__ssa_v1
        ;
        uint64_t v512 = (uint64_t) v38;
        TASSIGN(v511, v512);
        pipe_barrier(PIPE_V);
        TDIV(v511, v498, v509);
        // pto: %t__tmp_v247
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v513 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
        // pto: %t__tmp_v247
        ;
        uint64_t v514 = (uint64_t) v46;
        TASSIGN(v513, v514);
        pipe_barrier(PIPE_V);
        TGATHER(v513, v511, v426, v322);
        // pto: %col2_inline872__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v515 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
        // pto: %col2_inline872__ssa_v7
        ;
        uint64_t v516 = (uint64_t) v46;
        TASSIGN(v515, v516);
        pipe_barrier(PIPE_V);
        TADD(v515, v511, v513);
        // pto: %t__tmp_v248
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v517 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
        // pto: %t__tmp_v248
        ;
        uint64_t v518 = (uint64_t) v47;
        TASSIGN(v517, v518);
        pipe_barrier(PIPE_V);
        TGATHER(v517, v515, v460, v322);
        // pto: %t__tmp_v249
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v519 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
        // pto: %t__tmp_v249
        ;
        uint64_t v520 = (uint64_t) v46;
        TASSIGN(v519, v520);
        pipe_barrier(PIPE_V);
        TADD(v519, v515, v517);
        // pto: %col4_inline806__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v521 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
        // pto: %col4_inline806__ssa_v7
        ;
        uint64_t v522 = (uint64_t) v46;
        TASSIGN(v521, v522);
        pipe_barrier(PIPE_V);
        TADDS(v521, v519, v79);
        // pto: %cur_inline670__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v523 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
        // pto: %cur_inline670__ssa_v7
        ;
        uint64_t v524 = (uint64_t) v43;
        TASSIGN(v523, v524);
        pipe_barrier(PIPE_V);
        TDIV(v523, v511, v521);
      };
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      // pto: %t__tmp_v250
      ;
      Tile<TileType::Vec, float, 4, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v525 = Tile<TileType::Vec, float, 4, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v76, v68);
      // pto: %t__tmp_v250
      ;
      uint64_t v526 = (uint64_t) v43;
      TASSIGN(v525, v526);
      v525.SetValidShape(v239, v68);
      // pto: %comb_inline769__ssa_v0_pview
      ;
      const int64_t v527 = 0;
      // pto: %comb_inline769__ssa_v0_pview
      ;
      __gm__ float* v528 = PTOAS__GLOBAL_TENSOR_DATA(v144);
      // pto: %comb_inline769__ssa_v0_pview
      ;
      const int64_t v529 = 1;
      // pto: %comb_inline769__ssa_v0_pview
      ;
      const int64_t v530 = 1;
      // pto: %comb_inline769__ssa_v0_pview
      ;
      const int64_t v531 = 1;
      // pto: %comb_inline769__ssa_v0_pview
      ;
      int64_t v532 = v239 * v68;
      // pto: %comb_inline769__ssa_v0_pview
      ;
      int64_t v533 = v531 * v532;
      // pto: %comb_inline769__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, 16> v534 = pto::Shape<1, 1, 1, -1, 16>(v529, v530, v531, v239, v68);
      // pto: %comb_inline769__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v535 = pto::Stride<-1, -1, -1, -1, -1>(v530 * v533, v533, v532, v68, v63);
      // pto: %comb_inline769__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v536 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v528 + (v527 + v263 * v68 + v73 * v63), v534, v535);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      TSTORE(v536, v525);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    };
  } else {
    // pto: %74
    ;
    int64_t v537 = (int64_t) ((uint64_t) v229 - (uint64_t) v18);
    // pto: %76
    ;
    int64_t v538 = v537 < v22 ? v537 : v22;
    // pto: %75, %77, %78
    ;
    int64_t v539 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v537 * (uint64_t) v21) + (uint64_t) v538) * (uint64_t) v74);
    // pto: %79
    ;
    int64_t v540 = (int64_t) ((uint64_t) v22 - (uint64_t) v537);
    // pto: %80
    ;
    int64_t v541 = v540 < v73 ? v73 : v540;
    // pto: %81
    ;
    int64_t v542 = v541 < v63 ? v541 : v63;
    // pto: %84, %82, %83
    ;
    int64_t v543 = (int64_t) ((uint64_t) v539 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v21 + (uint64_t) v542) * (uint64_t) v74));
    // pto: %85
    ;
    // pto: %86
    ;
    // pto: %d_begin_inline920__phi_v2
    ;
    int64_t v544;
    // pto: %work_begin_inline814__phi_v3
    ;
    int64_t v545;
    // pto: %work_end_inline666__phi_v3
    ;
    int64_t v546;
    if (v24 < v25) {
      // pto: %87
      ;
      // pto: %d_worker_inline864__phi_v2
      ;
      int64_t v547;
      // pto: %d_workers_inline921__phi_v2
      ;
      int64_t v548;
      // pto: %pair_id_inline790__phi_v2
      ;
      int64_t v549;
      if (v537 < v26) {
        // pto: %88
        ;
        int64_t v550 = (int64_t) ((uint64_t) v27 + (uint64_t) v63);
        // pto: %89
        ;
        // pto: %91
        ;
        v547 = v537 % v550;
        v548 = v550;
        v549 = v537 / v550;
      } else {
        // pto: %93
        ;
        int64_t v551 = (int64_t) ((uint64_t) v537 - (uint64_t) v26);
        // pto: %95, %94
        ;
        // pto: %96
        ;
        v547 = v551 % v27;
        v548 = v27;
        v549 = (int64_t) ((uint64_t) v28 + (uint64_t) (v551 / v27));
      };
      // pto: %97
      ;
      uint64_t v552 = (uint64_t) v549;
      // pto: %97
      ;
      int64_t v553 = (int64_t) (v552 * (uint64_t) v74);
      // pto: %98
      ;
      int64_t v554 = v23 < v553 ? v23 : v553;
      // pto: %99
      ;
      int64_t v555 = (int64_t) ((uint64_t) v554 + (uint64_t) v74);
      // pto: %100
      ;
      // pto: %101
      ;
      int64_t v556 = v67 / v548;
      // pto: %102
      ;
      int64_t v557 = v67 % v548;
      // pto: %103
      ;
      uint64_t v558 = (uint64_t) v547;
      // pto: %104
      ;
      bool v559 = v547 < v557;
      // pto: %104
      ;
      int64_t v560 = v559 ? v547 : v557;
      // pto: %103, %105
      ;
      v544 = (int64_t) ((uint64_t) ((int64_t) v558 * (uint64_t) v556) + (uint64_t) v560);
      v545 = v554;
      v546 = v23 < v555 ? v23 : v555;
    } else {
      v544 = v73;
      v545 = v539;
      v546 = v23 < v543 ? v23 : v543;
    };
    // pto: %106
    ;
    bool v561 = v25 <= v537;
    // pto: %work_begin_inline814__phi_v5
    ;
    int64_t v562 = v561 ? v73 : v545;
    // pto: %work_end_inline666__phi_v5
    ;
    int64_t v563 = v561 ? v73 : v546;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    for (int64_t v564 = v562; v564 < v563; v564 += v76) {
      // pto: %107
      ;
      int64_t v565 = (int64_t) ((uint64_t) v563 - (uint64_t) v564);
      // pto: %108
      ;
      int64_t v566 = v565 < v76 ? v565 : v76;
      // pto: %109
      ;
      int64_t v567 = (int64_t) ((uint64_t) v566 * (uint64_t) v62);
      // pto: %m_parts_load_inline740__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v568 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v567);
      // pto: %m_parts_load_inline740__ssa_v0
      ;
      uint64_t v569 = (uint64_t) v36;
      TASSIGN(v568, v569);
      // pto: %110
      ;
      int64_t v570 = (int64_t) ((uint64_t) v564 * (uint64_t) v62);
      // pto: %111
      ;
      int64_t v571 = v570 < v73 ? v73 : v570;
      // pto: %113
      ;
      const int64_t v572 = 0;
      // pto: %113
      ;
      __gm__ float* v573 = PTOAS__GLOBAL_TENSOR_DATA(v118);
      // pto: %113
      ;
      const int64_t v574 = 1;
      // pto: %113
      ;
      const int64_t v575 = 1;
      // pto: %113
      ;
      const int64_t v576 = 1;
      // pto: %113
      ;
      int64_t v577 = v67 * v110;
      // pto: %113
      ;
      int64_t v578 = v576 * v577;
      // pto: %113
      ;
      pto::Shape<1, 1, 1, 8, -1> v579 = pto::Shape<1, 1, 1, 8, -1>(v574, v575, v576, v67, v567);
      // pto: %113
      ;
      pto::Stride<-1, -1, -1, -1, -1> v580 = pto::Stride<-1, -1, -1, -1, -1>(v575 * v578, v578, v577, v110, v63);
      // pto: %113
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v581 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v573 + (v572 + v73 * v110 + v571 * v63), v579, v580);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      TLOAD(v568, v581);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      // pto: %t__tmp_v251
      ;
      Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v582 = Tile<TileType::Vec, float, 8, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v67, v77);
      // pto: %t__tmp_v251
      ;
      uint64_t v583 = (uint64_t) v36;
      TASSIGN(v582, v583);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      TFILLPAD<pto::TFillPadMode::InPlace>(v582, v568);
      v582.SetValidShape(v67, v77);
      // pto: %t__tmp_v252
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v584 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v77);
      // pto: %t__tmp_v252
      ;
      uint64_t v585 = (uint64_t) v36;
      TASSIGN(v584, v585);
      pipe_barrier(PIPE_V);
      TCOLSUM(v584, v582);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      // pto: %m_total_inline926__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v586 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v62);
      // pto: %m_total_inline926__ssa_v0
      ;
      uint64_t v587 = (uint64_t) v36;
      TASSIGN(v586, v587);
      // pto: %m_sq_load_inline886__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v588 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v566);
      // pto: %m_sq_load_inline886__ssa_v0
      ;
      uint64_t v589 = (uint64_t) v38;
      TASSIGN(v588, v589);
      // pto: %114
      ;
      int64_t v590 = v564 < v73 ? v73 : v564;
      // pto: %115
      ;
      int64_t v591 = (int64_t) v29;
      // pto: %115
      ;
      const int64_t v592 = 0;
      // pto: %115
      ;
      __gm__ float* v593 = PTOAS__GLOBAL_TENSOR_DATA(v127);
      // pto: %115
      ;
      const int64_t v594 = 1;
      // pto: %115
      ;
      const int64_t v595 = 1;
      // pto: %115
      ;
      const int64_t v596 = 1;
      // pto: %115
      ;
      int64_t v597 = v67 * v591;
      // pto: %115
      ;
      int64_t v598 = v596 * v597;
      // pto: %115
      ;
      pto::Shape<1, 1, 1, 8, -1> v599 = pto::Shape<1, 1, 1, 8, -1>(v594, v595, v596, v67, v566);
      // pto: %115
      ;
      pto::Stride<-1, -1, -1, -1, -1> v600 = pto::Stride<-1, -1, -1, -1, -1>(v595 * v598, v598, v597, v591, v63);
      // pto: %115
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v601 = GlobalTensor<float, pto::Shape<1, 1, 1, 8, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v593 + (v592 + v73 * v591 + v590 * v63), v599, v600);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      TLOAD(v588, v601);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      // pto: %t__tmp_v253
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v602 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v67, v67);
      // pto: %t__tmp_v253
      ;
      uint64_t v603 = (uint64_t) v38;
      TASSIGN(v602, v603);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      TFILLPAD<pto::TFillPadMode::InPlace>(v602, v588);
      v602.SetValidShape(v67, v67);
      // pto: %m_rms_tmp_inline928__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v604 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v67);
      // pto: %m_rms_tmp_inline928__ssa_v0
      ;
      uint64_t v605 = (uint64_t) v39;
      TASSIGN(v604, v605);
      // pto: %t__tmp_v254
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v606 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v67);
      // pto: %t__tmp_v254
      ;
      uint64_t v607 = (uint64_t) v38;
      TASSIGN(v606, v607);
      pipe_barrier(PIPE_V);
      TCOLSUM(v606, v602);
      // pto: %t__tmp_v255
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v608 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v67);
      // pto: %t__tmp_v255
      ;
      uint64_t v609 = (uint64_t) v38;
      TASSIGN(v608, v609);
      pipe_barrier(PIPE_V);
      TMULS(v608, v606, v78);
      // pto: %t__tmp_v256
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v610 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v67);
      // pto: %t__tmp_v256
      ;
      uint64_t v611 = (uint64_t) v38;
      TASSIGN(v610, v611);
      pipe_barrier(PIPE_V);
      TADDS(v610, v608, v79);
      // pto: %t__tmp_v257
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v612 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v67);
      // pto: %t__tmp_v257
      ;
      uint64_t v613 = (uint64_t) v46;
      TASSIGN(v612, v613);
      pipe_barrier(PIPE_V);
      TRSQRT(v612, v610, v604);
      // pto: %m_inv_inline868__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v614 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v63);
      // pto: %m_inv_inline868__ssa_v0
      ;
      uint64_t v615 = (uint64_t) v46;
      TASSIGN(v614, v615);
      // pto: %gate_base_inline930__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v616 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v67);
      // pto: %gate_base_inline930__ssa_v0
      ;
      uint64_t v617 = (uint64_t) v48;
      TASSIGN(v616, v617);
      // pto: %116
      ;
      __gm__ float* v618 = PTOAS__GLOBAL_TENSOR_DATA(v135);
      // pto: %116
      ;
      pto::Shape<1, 1, 1, 1, 8> v619 = pto::Shape<1, 1, 1, 1, 8>();
      // pto: %116
      ;
      pto::Stride<24, 24, 24, 24, 1> v620 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %116
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v621 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v618, v619, v620);
      TLOAD(v616, v621);
      // pto: %gate_scales_inline931__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v622 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v66);
      // pto: %gate_scales_inline931__ssa_v0
      ;
      uint64_t v623 = (uint64_t) v49;
      TASSIGN(v622, v623);
      // pto: %hc_scale_2d_inline758__ssa_v0_pview
      ;
      __gm__ float* v624 = PTOAS__GLOBAL_TENSOR_DATA(v152);
      // pto: %hc_scale_2d_inline758__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 3> v625 = pto::Shape<1, 1, 1, 1, 3>();
      // pto: %hc_scale_2d_inline758__ssa_v0_pview
      ;
      pto::Stride<3, 3, 3, 3, 1> v626 = pto::Stride<3, 3, 3, 3, 1>();
      // pto: %hc_scale_2d_inline758__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND> v627 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND>(v624, v625, v626);
      TLOAD(v622, v627);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      // pto: %t__ci_tmp_v1
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v628 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v81);
      // pto: %t__ci_tmp_v1
      ;
      uint64_t v629 = (uint64_t) v38;
      TASSIGN(v628, v629);
      // pto: %t__tmp_v258
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v630 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v67);
      // pto: %t__tmp_v258
      ;
      uint64_t v631 = (uint64_t) v50;
      TASSIGN(v630, v631);
      // pto: %117
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v632;
      TRESHAPE(v632, v628);
      // pto: %118
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v633;
      TRESHAPE(v633, v630);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v633, v70, v632);
      // pto: %gate_ids_inline802__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v634 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v67);
      // pto: %gate_ids_inline802__ssa_v0
      ;
      uint64_t v635 = (uint64_t) v38;
      TASSIGN(v634, v635);
      pipe_barrier(PIPE_V);
      RoundMode v636 = RoundMode::CAST_ROUND;
      SaturationMode v637 = SaturationMode::ON;
      TCVT(v634, v630, v636, v637);
      // pto: %t__tmp_v259
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v638 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v67);
      // pto: %t__tmp_v259
      ;
      uint64_t v639 = (uint64_t) v38;
      TASSIGN(v638, v639);
      pipe_barrier(PIPE_V);
      TMULS(v638, v634, v85);
      // pto: %gate_scale_ids_inline932__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v640 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v67);
      // pto: %gate_scale_ids_inline932__ssa_v0
      ;
      uint64_t v641 = (uint64_t) v38;
      TASSIGN(v640, v641);
      pipe_barrier(PIPE_V);
      RoundMode v642 = RoundMode::CAST_FLOOR;
      SaturationMode v643 = SaturationMode::ON;
      TCVT(v640, v638, v642, v643);
      // pto: %gate_tmp_inline935__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v644 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v67);
      // pto: %gate_tmp_inline935__ssa_v0
      ;
      uint64_t v645 = (uint64_t) v51;
      TASSIGN(v644, v645);
      // pto: %gate_scale_vec_inline675__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v646 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v67);
      // pto: %gate_scale_vec_inline675__ssa_v0
      ;
      uint64_t v647 = (uint64_t) v40;
      TASSIGN(v646, v647);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      TGATHER(v646, v622, v640, v644);
      // pto: %t__tmp_v260
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v648 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v67);
      // pto: %t__tmp_v260
      ;
      uint64_t v649 = (uint64_t) v36;
      TASSIGN(v648, v649);
      // pto: %119
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v650;
      // pto: %119
      ;
      uint64_t v651 = (uint64_t) v36;
      TASSIGN(v650, v651);
      // pto: %gate_raw_inline933__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v652 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v67);
      // pto: %gate_raw_inline933__ssa_v0
      ;
      uint64_t v653 = (uint64_t) v36;
      TASSIGN(v652, v653);
      TADDS(v652, v650, v72);
      // pto: %gate_norm_inline938__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v654 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v67);
      // pto: %gate_norm_inline938__ssa_v0
      ;
      uint64_t v655 = (uint64_t) v36;
      TASSIGN(v654, v655);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v654, v652, v614);
      // pto: %gate_scaled_inline940__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v656 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v67);
      // pto: %gate_scaled_inline940__ssa_v0
      ;
      uint64_t v657 = (uint64_t) v36;
      TASSIGN(v656, v657);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v656, v654, v646);
      // pto: %t__tmp_v261
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v658 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v67);
      // pto: %t__tmp_v261
      ;
      uint64_t v659 = (uint64_t) v44;
      TASSIGN(v658, v659);
      TCOLEXPAND(v658, v616);
      // pto: %gate_logits_inline941__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v660 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v67);
      // pto: %gate_logits_inline941__ssa_v0
      ;
      uint64_t v661 = (uint64_t) v36;
      TASSIGN(v660, v661);
      pipe_barrier(PIPE_V);
      TADD(v660, v656, v658);
      // pto: %t__tmp_v262
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v662 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v67);
      // pto: %t__tmp_v262
      ;
      uint64_t v663 = (uint64_t) v36;
      TASSIGN(v662, v663);
      pipe_barrier(PIPE_V);
      TNEG(v662, v660);
      // pto: %t__tmp_v263
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v664 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v67);
      // pto: %t__tmp_v263
      ;
      uint64_t v665 = (uint64_t) v36;
      TASSIGN(v664, v665);
      pipe_barrier(PIPE_V);
      TEXP(v664, v662);
      // pto: %t__tmp_v264
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v666 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v67);
      // pto: %t__tmp_v264
      ;
      uint64_t v667 = (uint64_t) v36;
      TASSIGN(v666, v667);
      pipe_barrier(PIPE_V);
      TADDS(v666, v664, v82);
      // pto: %gate_sigmoid_inline943__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v668 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v67, v67);
      // pto: %gate_sigmoid_inline943__ssa_v0
      ;
      uint64_t v669 = (uint64_t) v52;
      TASSIGN(v668, v669);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
      TRECIP(v668, v666);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      // pto: %t__ci_tmp_v2
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v670 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v81);
      // pto: %t__ci_tmp_v2
      ;
      uint64_t v671 = (uint64_t) v36;
      TASSIGN(v670, v671);
      // pto: %t__tmp_v265
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v672 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v62);
      // pto: %t__tmp_v265
      ;
      uint64_t v673 = (uint64_t) v43;
      TASSIGN(v672, v673);
      // pto: %120
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v674;
      TRESHAPE(v674, v670);
      // pto: %121
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v675;
      TRESHAPE(v675, v672);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v675, v70, v674);
      // pto: %post_ids_float_inline724__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v676 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v62);
      // pto: %post_ids_float_inline724__ssa_v0
      ;
      uint64_t v677 = (uint64_t) v36;
      TASSIGN(v676, v677);
      pipe_barrier(PIPE_V);
      RoundMode v678 = RoundMode::CAST_ROUND;
      SaturationMode v679 = SaturationMode::ON;
      TCVT(v676, v672, v678, v679);
      // pto: %t__tmp_v266
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v680 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v62);
      // pto: %t__tmp_v266
      ;
      uint64_t v681 = (uint64_t) v53;
      TASSIGN(v680, v681);
      pipe_barrier(PIPE_V);
      TMULS(v680, v676, v85);
      // pto: %t__tmp_v267
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v682 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v62);
      // pto: %t__tmp_v267
      ;
      uint64_t v683 = (uint64_t) v53;
      TASSIGN(v682, v683);
      pipe_barrier(PIPE_V);
      RoundMode v684 = RoundMode::CAST_FLOOR;
      SaturationMode v685 = SaturationMode::ON;
      TCVT(v682, v680, v684, v685);
      // pto: %post_rows_inline893__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v686 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v62);
      // pto: %post_rows_inline893__ssa_v0
      ;
      uint64_t v687 = (uint64_t) v53;
      TASSIGN(v686, v687);
      pipe_barrier(PIPE_V);
      RoundMode v688 = RoundMode::CAST_ROUND;
      SaturationMode v689 = SaturationMode::ON;
      TCVT(v686, v682, v688, v689);
      // pto: %t__tmp_v268
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v690 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v62);
      // pto: %t__tmp_v268
      ;
      uint64_t v691 = (uint64_t) v53;
      TASSIGN(v690, v691);
      pipe_barrier(PIPE_V);
      TMULS(v690, v686, v86);
      // pto: %t__tmp_v269
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v692 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v62);
      // pto: %t__tmp_v269
      ;
      uint64_t v693 = (uint64_t) v36;
      TASSIGN(v692, v693);
      pipe_barrier(PIPE_V);
      TADD(v692, v676, v690);
      // pto: %t__tmp_v270
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v694 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v62);
      // pto: %t__tmp_v270
      ;
      uint64_t v695 = (uint64_t) v36;
      TASSIGN(v694, v695);
      pipe_barrier(PIPE_V);
      TADDS(v694, v692, v86);
      // pto: %post_ids_inline836__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v696 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v62);
      // pto: %post_ids_inline836__ssa_v0
      ;
      uint64_t v697 = (uint64_t) v36;
      TASSIGN(v696, v697);
      pipe_barrier(PIPE_V);
      RoundMode v698 = RoundMode::CAST_ROUND;
      SaturationMode v699 = SaturationMode::ON;
      TCVT(v696, v694, v698, v699);
      // pto: %post_tmp_inline944__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v700 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v62);
      // pto: %post_tmp_inline944__ssa_v0
      ;
      uint64_t v701 = (uint64_t) v53;
      TASSIGN(v700, v701);
      // pto: %t__tmp_v271
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v702 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v80);
      // pto: %t__tmp_v271
      ;
      uint64_t v703 = (uint64_t) v52;
      TASSIGN(v702, v703);
      // pto: %post_values_inline761__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v704 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v62);
      // pto: %post_values_inline761__ssa_v0
      ;
      uint64_t v705 = (uint64_t) v44;
      TASSIGN(v704, v705);
      pipe_barrier(PIPE_V);
      TGATHER(v704, v702, v696, v700);
      // pto: %t__tmp_v272
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v706 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v62);
      // pto: %t__tmp_v272
      ;
      uint64_t v707 = (uint64_t) v36;
      TASSIGN(v706, v707);
      pipe_barrier(PIPE_V);
      TMULS(v706, v704, v84);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      // pto: %122
      ;
      int64_t v708 = (int64_t) ((uint64_t) v566 * (uint64_t) v76);
      v706.SetValidShape(v63, v708);
      // pto: %123
      ;
      bool v709 = v544 == v73;
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      if (v709) {
        // pto: %125
        ;
        int64_t v710 = (int64_t) ((uint64_t) v564 * (uint64_t) v76);
        // pto: %126
        ;
        int64_t v711 = v710 < v73 ? v73 : v710;
        // pto: %post_flat_inline735__ssa_v0_pview
        ;
        const int64_t v712 = 0;
        // pto: %post_flat_inline735__ssa_v0_pview
        ;
        __gm__ float* v713 = PTOAS__GLOBAL_TENSOR_DATA(v161);
        // pto: %post_flat_inline735__ssa_v0_pview
        ;
        const int64_t v714 = 1;
        // pto: %post_flat_inline735__ssa_v0_pview
        ;
        const int64_t v715 = 1;
        // pto: %post_flat_inline735__ssa_v0_pview
        ;
        const int64_t v716 = 1;
        // pto: %post_flat_inline735__ssa_v0_pview
        ;
        int64_t v717 = v63 * v153;
        // pto: %post_flat_inline735__ssa_v0_pview
        ;
        int64_t v718 = v716 * v717;
        // pto: %post_flat_inline735__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, -1> v719 = pto::Shape<1, 1, 1, 1, -1>(v714, v715, v716, v63, v708);
        // pto: %post_flat_inline735__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v720 = pto::Stride<-1, -1, -1, -1, -1>(v715 * v718, v718, v717, v153, v63);
        // pto: %post_flat_inline735__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v721 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v713 + (v712 + v73 * v153 + v711 * v63), v719, v720);
        TSTORE(v721, v706);
      };
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      // pto: %127
      ;
      int64_t v722 = v566 / v74;
      // pto: %128
      ;
      int64_t v723 = (int64_t) ((uint64_t) v722 * (uint64_t) v74);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      for (int64_t v724 = v73; v724 < v723; v724 += v74) {
        // pto: %129
        ;
        int64_t v725 = (int64_t) ((uint64_t) v564 + (uint64_t) v724);
        // pto: %130
        ;
        int64_t v726 = (int64_t) ((uint64_t) v724 + (uint64_t) v63);
        // pto: %131
        ;
        int64_t v727 = (int64_t) ((uint64_t) v564 + (uint64_t) v726);
        // pto: %flat_offset_mul
        ;
        int64_t v728 = (int64_t) ((uint64_t) v724 * (uint64_t) v67);
        // pto: %t__tmp_v273
        ;
        float v729 = v668.GetValue(v728);
        // pto: %132
        ;
        float v730 = v729 + v79;
        // pto: %134, %t__tmp_v274
        ;
        float v731 = v668.GetValue((int64_t) ((uint64_t) v728 + (uint64_t) v63));
        // pto: %135
        ;
        float v732 = v731 + v79;
        // pto: %137, %t__tmp_v275
        ;
        float v733 = v668.GetValue((int64_t) ((uint64_t) v728 + (uint64_t) v74));
        // pto: %138
        ;
        float v734 = v733 + v79;
        // pto: %140, %t__tmp_v276
        ;
        float v735 = v668.GetValue((int64_t) ((uint64_t) v728 + (uint64_t) v66));
        // pto: %141
        ;
        float v736 = v735 + v79;
        // pto: %t__tmp_v277
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v737 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %t__tmp_v277
        ;
        uint64_t v738 = (uint64_t) v54;
        TASSIGN(v737, v738);
        // pto: %142
        ;
        int64_t v739 = v725 < v73 ? v73 : v725;
        // pto: %143
        ;
        __gm__ bfloat16_t* v740 = PTOAS__GLOBAL_TENSOR_DATA(v109);
        // pto: %143
        ;
        const int64_t v741 = 0;
        // pto: %143
        ;
        const int64_t v742 = 16384;
        // pto: %143
        ;
        pto::Shape<1, 1, 1, 1, 4096> v743 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %143
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v744 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %143
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v745 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v740 + (v741 + v739 * v742), v743, v744);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        TLOAD(v737, v745);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        // pto: %t__tmp_v278
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v746 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %t__tmp_v278
        ;
        uint64_t v747 = (uint64_t) v55;
        TASSIGN(v746, v747);
        // pto: %145
        ;
        __gm__ bfloat16_t* v748 = PTOAS__GLOBAL_TENSOR_DATA(v109);
        // pto: %145
        ;
        const int64_t v749 = 4096;
        // pto: %145
        ;
        const int64_t v750 = 16384;
        // pto: %145
        ;
        pto::Shape<1, 1, 1, 1, 4096> v751 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %145
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v752 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %145
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v753 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v748 + (v749 + v739 * v750), v751, v752);
        TLOAD(v746, v753);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %t__tmp_v279
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v754 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %t__tmp_v279
        ;
        uint64_t v755 = (uint64_t) v56;
        TASSIGN(v754, v755);
        // pto: %147
        ;
        __gm__ bfloat16_t* v756 = PTOAS__GLOBAL_TENSOR_DATA(v109);
        // pto: %147
        ;
        const int64_t v757 = 8192;
        // pto: %147
        ;
        const int64_t v758 = 16384;
        // pto: %147
        ;
        pto::Shape<1, 1, 1, 1, 4096> v759 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %147
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v760 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %147
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v761 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v756 + (v757 + v739 * v758), v759, v760);
        TLOAD(v754, v761);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %t__tmp_v280
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v762 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %t__tmp_v280
        ;
        uint64_t v763 = (uint64_t) v57;
        TASSIGN(v762, v763);
        // pto: %149
        ;
        __gm__ bfloat16_t* v764 = PTOAS__GLOBAL_TENSOR_DATA(v109);
        // pto: %149
        ;
        const int64_t v765 = 12288;
        // pto: %149
        ;
        const int64_t v766 = 16384;
        // pto: %149
        ;
        pto::Shape<1, 1, 1, 1, 4096> v767 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %149
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v768 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %149
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v769 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v764 + (v765 + v739 * v766), v767, v768);
        TLOAD(v762, v769);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %152
        ;
        int64_t v770 = (int64_t) ((uint64_t) v726 * (uint64_t) v67);
        // pto: %150
        ;
        float v771 = v668.GetValue(v770);
        // pto: %154
        ;
        float v772 = v771 + v79;
        // pto: %158, %155
        ;
        float v773 = v668.GetValue((int64_t) ((uint64_t) v770 + (uint64_t) v63));
        // pto: %159
        ;
        float v774 = v773 + v79;
        // pto: %163, %160
        ;
        float v775 = v668.GetValue((int64_t) ((uint64_t) v770 + (uint64_t) v74));
        // pto: %164
        ;
        float v776 = v775 + v79;
        // pto: %168, %165
        ;
        float v777 = v668.GetValue((int64_t) ((uint64_t) v770 + (uint64_t) v66));
        // pto: %169
        ;
        float v778 = v777 + v79;
        // pto: %4
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v779 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %4
        ;
        uint64_t v780 = (uint64_t) v35;
        TASSIGN(v779, v780);
        // pto: %170
        ;
        int64_t v781 = v727 < v73 ? v73 : v727;
        // pto: %171
        ;
        __gm__ bfloat16_t* v782 = PTOAS__GLOBAL_TENSOR_DATA(v109);
        // pto: %171
        ;
        const int64_t v783 = 0;
        // pto: %171
        ;
        const int64_t v784 = 16384;
        // pto: %171
        ;
        pto::Shape<1, 1, 1, 1, 4096> v785 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %171
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v786 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %171
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v787 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v782 + (v783 + v781 * v784), v785, v786);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v779, v787);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %5
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v788 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %5
        ;
        uint64_t v789 = (uint64_t) v58;
        TASSIGN(v788, v789);
        // pto: %173
        ;
        __gm__ bfloat16_t* v790 = PTOAS__GLOBAL_TENSOR_DATA(v109);
        // pto: %173
        ;
        const int64_t v791 = 4096;
        // pto: %173
        ;
        const int64_t v792 = 16384;
        // pto: %173
        ;
        pto::Shape<1, 1, 1, 1, 4096> v793 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %173
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v794 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %173
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v795 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v790 + (v791 + v781 * v792), v793, v794);
        TLOAD(v788, v795);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %6
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v796 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %6
        ;
        uint64_t v797 = (uint64_t) v59;
        TASSIGN(v796, v797);
        // pto: %175
        ;
        __gm__ bfloat16_t* v798 = PTOAS__GLOBAL_TENSOR_DATA(v109);
        // pto: %175
        ;
        const int64_t v799 = 8192;
        // pto: %175
        ;
        const int64_t v800 = 16384;
        // pto: %175
        ;
        pto::Shape<1, 1, 1, 1, 4096> v801 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %175
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v802 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %175
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v803 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v798 + (v799 + v781 * v800), v801, v802);
        TLOAD(v796, v803);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %7
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v804 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %7
        ;
        uint64_t v805 = (uint64_t) v60;
        TASSIGN(v804, v805);
        // pto: %177
        ;
        __gm__ bfloat16_t* v806 = PTOAS__GLOBAL_TENSOR_DATA(v109);
        // pto: %177
        ;
        const int64_t v807 = 12288;
        // pto: %177
        ;
        const int64_t v808 = 16384;
        // pto: %177
        ;
        pto::Shape<1, 1, 1, 1, 4096> v809 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %177
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v810 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %177
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v811 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v806 + (v807 + v781 * v808), v809, v810);
        TLOAD(v804, v811);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        // pto: %scalar_x0_inline776__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v812 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %scalar_x0_inline776__ssa_v0
        ;
        uint64_t v813 = (uint64_t) v36;
        TASSIGN(v812, v813);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        RoundMode v814 = RoundMode::CAST_ROUND;
        SaturationMode v815 = SaturationMode::ON;
        TCVT(v812, v737, v814, v815);
        // pto: %scalar_y0_inline643__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v816 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %scalar_y0_inline643__ssa_v0
        ;
        uint64_t v817 = (uint64_t) v36;
        TASSIGN(v816, v817);
        pipe_barrier(PIPE_V);
        TMULS(v816, v812, v730);
        // pto: %scalar_x1_inline907__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v818 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %scalar_x1_inline907__ssa_v0
        ;
        uint64_t v819 = (uint64_t) v54;
        TASSIGN(v818, v819);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v820 = RoundMode::CAST_ROUND;
        SaturationMode v821 = SaturationMode::ON;
        TCVT(v818, v746, v820, v821);
        // pto: %scalar_y1_inline779__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v822 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %scalar_y1_inline779__ssa_v0
        ;
        uint64_t v823 = (uint64_t) v54;
        TASSIGN(v822, v823);
        pipe_barrier(PIPE_V);
        TMULS(v822, v818, v732);
        // pto: %scalar_sum01_inline642__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v824 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %scalar_sum01_inline642__ssa_v0
        ;
        uint64_t v825 = (uint64_t) v36;
        TASSIGN(v824, v825);
        pipe_barrier(PIPE_V);
        TADD(v824, v816, v822);
        // pto: %scalar_x2_inline649__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v826 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %scalar_x2_inline649__ssa_v0
        ;
        uint64_t v827 = (uint64_t) v54;
        TASSIGN(v826, v827);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        RoundMode v828 = RoundMode::CAST_ROUND;
        SaturationMode v829 = SaturationMode::ON;
        TCVT(v826, v754, v828, v829);
        // pto: %scalar_y2_inline763__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v830 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %scalar_y2_inline763__ssa_v0
        ;
        uint64_t v831 = (uint64_t) v54;
        TASSIGN(v830, v831);
        pipe_barrier(PIPE_V);
        TMULS(v830, v826, v734);
        // pto: %scalar_x3_inline762__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v832 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %scalar_x3_inline762__ssa_v0
        ;
        uint64_t v833 = (uint64_t) v55;
        TASSIGN(v832, v833);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        RoundMode v834 = RoundMode::CAST_ROUND;
        SaturationMode v835 = SaturationMode::ON;
        TCVT(v832, v762, v834, v835);
        // pto: %scalar_y3_inline736__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v836 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %scalar_y3_inline736__ssa_v0
        ;
        uint64_t v837 = (uint64_t) v55;
        TASSIGN(v836, v837);
        pipe_barrier(PIPE_V);
        TMULS(v836, v832, v736);
        // pto: %t__tmp_v281
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v838 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %t__tmp_v281
        ;
        uint64_t v839 = (uint64_t) v54;
        TASSIGN(v838, v839);
        pipe_barrier(PIPE_V);
        TADD(v838, v830, v836);
        // pto: %scalar_result_inline803__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v840 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %scalar_result_inline803__ssa_v0
        ;
        uint64_t v841 = (uint64_t) v36;
        TASSIGN(v840, v841);
        pipe_barrier(PIPE_V);
        TADD(v840, v824, v838);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        // pto: %scalar_bf16_inline877__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v842 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %scalar_bf16_inline877__ssa_v0
        ;
        uint64_t v843 = (uint64_t) v36;
        TASSIGN(v842, v843);
        pipe_barrier(PIPE_V);
        RoundMode v844 = RoundMode::CAST_RINT;
        SaturationMode v845 = SaturationMode::ON;
        TCVT(v842, v840, v844, v845);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        // pto: %mixed_view_inline813__ssa_v0_pview
        ;
        __gm__ bfloat16_t* v846 = PTOAS__GLOBAL_TENSOR_DATA(v170);
        // pto: %mixed_view_inline813__ssa_v0_pview
        ;
        const int64_t v847 = 0;
        // pto: %mixed_view_inline813__ssa_v0_pview
        ;
        const int64_t v848 = 4096;
        // pto: %mixed_view_inline813__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, 4096> v849 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %mixed_view_inline813__ssa_v0_pview
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v850 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %mixed_view_inline813__ssa_v0_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v851 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v846 + (v847 + v739 * v848), v849, v850);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v851, v842);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        // pto: %8
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v852 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %8
        ;
        uint64_t v853 = (uint64_t) v61;
        TASSIGN(v852, v853);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        RoundMode v854 = RoundMode::CAST_ROUND;
        SaturationMode v855 = SaturationMode::ON;
        TCVT(v852, v779, v854, v855);
        // pto: %9
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v856 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %9
        ;
        uint64_t v857 = (uint64_t) v61;
        TASSIGN(v856, v857);
        pipe_barrier(PIPE_V);
        TMULS(v856, v852, v772);
        // pto: %10
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v858 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %10
        ;
        uint64_t v859 = (uint64_t) v35;
        TASSIGN(v858, v859);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        RoundMode v860 = RoundMode::CAST_ROUND;
        SaturationMode v861 = SaturationMode::ON;
        TCVT(v858, v788, v860, v861);
        // pto: %11
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v862 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %11
        ;
        uint64_t v863 = (uint64_t) v35;
        TASSIGN(v862, v863);
        pipe_barrier(PIPE_V);
        TMULS(v862, v858, v774);
        // pto: %12
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v864 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %12
        ;
        uint64_t v865 = (uint64_t) v61;
        TASSIGN(v864, v865);
        pipe_barrier(PIPE_V);
        TADD(v864, v856, v862);
        // pto: %13
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v866 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %13
        ;
        uint64_t v867 = (uint64_t) v35;
        TASSIGN(v866, v867);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        RoundMode v868 = RoundMode::CAST_ROUND;
        SaturationMode v869 = SaturationMode::ON;
        TCVT(v866, v796, v868, v869);
        // pto: %14
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v870 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %14
        ;
        uint64_t v871 = (uint64_t) v35;
        TASSIGN(v870, v871);
        pipe_barrier(PIPE_V);
        TMULS(v870, v866, v776);
        // pto: %15
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v872 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %15
        ;
        uint64_t v873 = (uint64_t) v58;
        TASSIGN(v872, v873);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        RoundMode v874 = RoundMode::CAST_ROUND;
        SaturationMode v875 = SaturationMode::ON;
        TCVT(v872, v804, v874, v875);
        // pto: %16
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v876 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %16
        ;
        uint64_t v877 = (uint64_t) v58;
        TASSIGN(v876, v877);
        pipe_barrier(PIPE_V);
        TMULS(v876, v872, v778);
        // pto: %17
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v878 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %17
        ;
        uint64_t v879 = (uint64_t) v35;
        TASSIGN(v878, v879);
        pipe_barrier(PIPE_V);
        TADD(v878, v870, v876);
        // pto: %18
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v880 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %18
        ;
        uint64_t v881 = (uint64_t) v61;
        TASSIGN(v880, v881);
        pipe_barrier(PIPE_V);
        TADD(v880, v864, v878);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %19
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v882 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %19
        ;
        uint64_t v883 = (uint64_t) v61;
        TASSIGN(v882, v883);
        pipe_barrier(PIPE_V);
        RoundMode v884 = RoundMode::CAST_RINT;
        SaturationMode v885 = SaturationMode::ON;
        TCVT(v882, v880, v884, v885);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        // pto: %181
        ;
        __gm__ bfloat16_t* v886 = PTOAS__GLOBAL_TENSOR_DATA(v170);
        // pto: %181
        ;
        const int64_t v887 = 0;
        // pto: %181
        ;
        const int64_t v888 = 4096;
        // pto: %181
        ;
        pto::Shape<1, 1, 1, 1, 4096> v889 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %181
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v890 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %181
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v891 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v886 + (v887 + v781 * v888), v889, v890);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v891, v882);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
      };
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_S, PIPE_V, EVENT_ID1);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      // pto: %182, %183
      ;
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      if ((int64_t) ((uint64_t) v566 - (uint64_t) v723) == v63) {
        // pto: %184
        ;
        int64_t v892 = (int64_t) ((uint64_t) v564 + (uint64_t) v723);
        int64_t v893 = (int64_t) ((uint64_t) v722 * (uint64_t) v68);
        // pto: %185
        ;
        float v894 = v668.GetValue(v893);
        // pto: %188
        ;
        float v895 = v894 + v79;
        // pto: %191, %189
        ;
        float v896 = v668.GetValue((int64_t) ((uint64_t) v893 + (uint64_t) v63));
        // pto: %192
        ;
        float v897 = v896 + v79;
        // pto: %195, %193
        ;
        float v898 = v668.GetValue((int64_t) ((uint64_t) v893 + (uint64_t) v74));
        // pto: %196
        ;
        float v899 = v898 + v79;
        // pto: %199, %197
        ;
        float v900 = v668.GetValue((int64_t) ((uint64_t) v893 + (uint64_t) v66));
        // pto: %200
        ;
        float v901 = v900 + v79;
        // pto: %20
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v902 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %20
        ;
        uint64_t v903 = (uint64_t) v54;
        TASSIGN(v902, v903);
        // pto: %201
        ;
        int64_t v904 = v892 < v73 ? v73 : v892;
        // pto: %202
        ;
        __gm__ bfloat16_t* v905 = PTOAS__GLOBAL_TENSOR_DATA(v109);
        // pto: %202
        ;
        const int64_t v906 = 0;
        // pto: %202
        ;
        const int64_t v907 = 16384;
        // pto: %202
        ;
        pto::Shape<1, 1, 1, 1, 4096> v908 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %202
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v909 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %202
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v910 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v905 + (v906 + v904 * v907), v908, v909);
        TLOAD(v902, v910);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %21
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v911 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %21
        ;
        uint64_t v912 = (uint64_t) v36;
        TASSIGN(v911, v912);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v913 = RoundMode::CAST_ROUND;
        SaturationMode v914 = SaturationMode::ON;
        TCVT(v911, v902, v913, v914);
        // pto: %22
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v915 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %22
        ;
        uint64_t v916 = (uint64_t) v36;
        TASSIGN(v915, v916);
        pipe_barrier(PIPE_V);
        TMULS(v915, v911, v895);
        // pto: %23
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v917 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %23
        ;
        uint64_t v918 = (uint64_t) v55;
        TASSIGN(v917, v918);
        // pto: %204
        ;
        __gm__ bfloat16_t* v919 = PTOAS__GLOBAL_TENSOR_DATA(v109);
        // pto: %204
        ;
        const int64_t v920 = 4096;
        // pto: %204
        ;
        const int64_t v921 = 16384;
        // pto: %204
        ;
        pto::Shape<1, 1, 1, 1, 4096> v922 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %204
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v923 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %204
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v924 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v919 + (v920 + v904 * v921), v922, v923);
        TLOAD(v917, v924);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %24
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v925 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %24
        ;
        uint64_t v926 = (uint64_t) v54;
        TASSIGN(v925, v926);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v927 = RoundMode::CAST_ROUND;
        SaturationMode v928 = SaturationMode::ON;
        TCVT(v925, v917, v927, v928);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %25
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v929 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %25
        ;
        uint64_t v930 = (uint64_t) v54;
        TASSIGN(v929, v930);
        pipe_barrier(PIPE_V);
        TMULS(v929, v925, v897);
        // pto: %26
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v931 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %26
        ;
        uint64_t v932 = (uint64_t) v36;
        TASSIGN(v931, v932);
        pipe_barrier(PIPE_V);
        TADD(v931, v915, v929);
        // pto: %27
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v933 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %27
        ;
        uint64_t v934 = (uint64_t) v55;
        TASSIGN(v933, v934);
        // pto: %206
        ;
        __gm__ bfloat16_t* v935 = PTOAS__GLOBAL_TENSOR_DATA(v109);
        // pto: %206
        ;
        const int64_t v936 = 8192;
        // pto: %206
        ;
        const int64_t v937 = 16384;
        // pto: %206
        ;
        pto::Shape<1, 1, 1, 1, 4096> v938 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %206
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v939 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %206
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v940 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v935 + (v936 + v904 * v937), v938, v939);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v933, v940);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %28
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v941 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %28
        ;
        uint64_t v942 = (uint64_t) v54;
        TASSIGN(v941, v942);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v943 = RoundMode::CAST_ROUND;
        SaturationMode v944 = SaturationMode::ON;
        TCVT(v941, v933, v943, v944);
        // pto: %29
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v945 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %29
        ;
        uint64_t v946 = (uint64_t) v54;
        TASSIGN(v945, v946);
        pipe_barrier(PIPE_V);
        TMULS(v945, v941, v899);
        // pto: %30
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v947 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %30
        ;
        uint64_t v948 = (uint64_t) v61;
        TASSIGN(v947, v948);
        // pto: %208
        ;
        __gm__ bfloat16_t* v949 = PTOAS__GLOBAL_TENSOR_DATA(v109);
        // pto: %208
        ;
        const int64_t v950 = 12288;
        // pto: %208
        ;
        const int64_t v951 = 16384;
        // pto: %208
        ;
        pto::Shape<1, 1, 1, 1, 4096> v952 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %208
        ;
        pto::Stride<16384, 16384, 16384, 16384, 1> v953 = pto::Stride<16384, 16384, 16384, 16384, 1>();
        // pto: %208
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND> v954 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<16384, 16384, 16384, 16384, 1>, pto::Layout::ND>(v949 + (v950 + v904 * v951), v952, v953);
        TLOAD(v947, v954);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %31
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v955 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %31
        ;
        uint64_t v956 = (uint64_t) v55;
        TASSIGN(v955, v956);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v957 = RoundMode::CAST_ROUND;
        SaturationMode v958 = SaturationMode::ON;
        TCVT(v955, v947, v957, v958);
        // pto: %32
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v959 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %32
        ;
        uint64_t v960 = (uint64_t) v55;
        TASSIGN(v959, v960);
        pipe_barrier(PIPE_V);
        TMULS(v959, v955, v901);
        // pto: %33
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v961 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %33
        ;
        uint64_t v962 = (uint64_t) v54;
        TASSIGN(v961, v962);
        pipe_barrier(PIPE_V);
        TADD(v961, v945, v959);
        // pto: %34
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v963 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %34
        ;
        uint64_t v964 = (uint64_t) v36;
        TASSIGN(v963, v964);
        pipe_barrier(PIPE_V);
        TADD(v963, v931, v961);
        // pto: %35
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v965 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v63, v69);
        // pto: %35
        ;
        uint64_t v966 = (uint64_t) v36;
        TASSIGN(v965, v966);
        pipe_barrier(PIPE_V);
        RoundMode v967 = RoundMode::CAST_RINT;
        SaturationMode v968 = SaturationMode::ON;
        TCVT(v965, v963, v967, v968);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        // pto: %211
        ;
        __gm__ bfloat16_t* v969 = PTOAS__GLOBAL_TENSOR_DATA(v170);
        // pto: %211
        ;
        const int64_t v970 = 0;
        // pto: %211
        ;
        const int64_t v971 = 4096;
        // pto: %211
        ;
        pto::Shape<1, 1, 1, 1, 4096> v972 = pto::Shape<1, 1, 1, 1, 4096>();
        // pto: %211
        ;
        pto::Stride<4096, 4096, 4096, 4096, 1> v973 = pto::Stride<4096, 4096, 4096, 4096, 1>();
        // pto: %211
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND> v974 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 4096>, pto::Stride<4096, 4096, 4096, 4096, 1>, pto::Layout::ND>(v969 + (v970 + v904 * v971), v972, v973);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v974, v965);
      };
      set_flag(PIPE_S, PIPE_V, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}