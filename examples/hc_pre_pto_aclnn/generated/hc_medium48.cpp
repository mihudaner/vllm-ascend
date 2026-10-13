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

AICORE void hc_medium48_aic(__gm__ float* v1, __gm__ float* v2, __gm__ bfloat16_t* v3, __gm__ float* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ bfloat16_t* v12, __gm__ float* v13, int64_t v14, int32_t v15, int64_t v16, int64_t v17, int64_t v18, int64_t v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int64_t v28, int64_t v29, int64_t v30, int32_t v31, int32_t v32) {
  using T = float;

  #if defined(__DAV_CUBE__)
  const int64_t v33 = 896;
  const int32_t v34 = 0;
  // pto: %c393216_i64
  const int64_t v35 = 393216;
  // pto: %c0_i64
  const int64_t v36 = 0;
  // pto: %c24576_i64
  const int64_t v37 = 24576;
  // pto: %c16384_i64
  const int64_t v38 = 16384;
  // pto: %c32_index
  const int64_t v39 = 32;
  // pto: %c1_index
  const int64_t v40 = 1;
  // pto: %c16384_index
  const int64_t v41 = 16384;
  // pto: %c24_index
  const int64_t v42 = 24;
  // pto: %c16_index
  const int64_t v43 = 16;
  // pto: %c48_index
  const int64_t v44 = 48;
  // pto: %c1024_index
  const int64_t v45 = 1024;
  // pto: %c0_index
  const int64_t v46 = 0;
  // pto: %c256_index
  const int64_t v47 = 256;
  // pto: %c128_index
  const int64_t v48 = 128;
  // pto: %cn128_index
  const int64_t v49 = -128;
  // pto: %21
  int64_t v50 = (int64_t) ((uint64_t) v29 * (uint64_t) v43);
  // pto: %mixes_partials_inline456__ssa_v0_view
  const int64_t v51 = 1;
  // pto: %mixes_partials_inline456__ssa_v0_view
  const int64_t v52 = 1;
  // pto: %mixes_partials_inline456__ssa_v0_view
  const int64_t v53 = 1;
  // pto: %mixes_partials_inline456__ssa_v0_view
  int64_t v54 = v50 * v39;
  // pto: %mixes_partials_inline456__ssa_v0_view
  int64_t v55 = v53 * v54;
  // pto: %mixes_partials_inline456__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v56 = pto::Shape<1, 1, 1, -1, -1>(v51, v52, v53, v50, v39);
  // pto: %mixes_partials_inline456__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v57 = pto::Stride<-1, -1, -1, -1, -1>(v52 * v55, v55, v54, v39, v40);
  // pto: %mixes_partials_inline456__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v58 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v56, v57);
  // pto: %hc_fn__ssa_v0_view
  const int64_t v59 = 1;
  // pto: %hc_fn__ssa_v0_view
  const int64_t v60 = 1;
  // pto: %hc_fn__ssa_v0_view
  const int64_t v61 = 1;
  // pto: %hc_fn__ssa_v0_view
  int64_t v62 = v42 * v41;
  // pto: %hc_fn__ssa_v0_view
  int64_t v63 = v61 * v62;
  // pto: %hc_fn__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v64 = pto::Shape<1, 1, 1, -1, -1>(v59, v60, v61, v42, v41);
  // pto: %hc_fn__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v65 = pto::Stride<-1, -1, -1, -1, -1>(v60 * v63, v63, v62, v41, v40);
  // pto: %hc_fn__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v66 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v4, v64, v65);
  const int32_t v67 = 0;
  auto v68 = TPipe<0, Direction::DIR_V2C, 196608, 2, 2, false>(v13, v67, v34);
  // pto: %worker_inline463__ssa_v0
  // pto: %linear_workers_inline445__ssa_v0_idx
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  for (int64_t v69 = (int64_t) v31; v69 < v14; v69 += (int64_t) v15) {
    // pto: %5
    ;
    int64_t v70 = v69 % v43;
    // pto: %6
    ;
    int64_t v71 = (int64_t) ((uint64_t) v70 * (uint64_t) v45);
    // pto: %7, %8, %3, %4
    ;
    int64_t v72 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v70 * (uint64_t) v16) + (uint64_t) ((int64_t) (uint64_t) (v69 / v43) * (uint64_t) v44));
    // pto: %gathered_inline469__tile
    ;
    const int32_t v73 = 48;
    // pto: %gathered_inline469__tile
    ;
    const int32_t v74 = 1024;
    // pto: %gathered_inline469__tile
    ;
    Tile<TileType::Mat, float, 48, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v75 = Tile<TileType::Mat, float, 48, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v73, v74);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_MTE2);
    TPOP<TPipe<0, Direction::DIR_V2C, 196608, 2, 2, false>, Tile<TileType::Mat, float, 48, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v68, v75);
    // pto: %weight_inline434__tile
    ;
    Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v76 = Tile<TileType::Mat, float, 32, 1024, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v42, v45);
    // pto: %weight_inline434__tile
    ;
    uint64_t v77 = (uint64_t) v35;
    TASSIGN(v76, v77);
    // pto: %9
    ;
    int64_t v78 = v71 < v46 ? v46 : v71;
    // pto: %hc_fn__ssa_v0_pview
    ;
    const int64_t v79 = 0;
    // pto: %hc_fn__ssa_v0_pview
    ;
    __gm__ float* v80 = PTOAS__GLOBAL_TENSOR_DATA(v66);
    // pto: %hc_fn__ssa_v0_pview
    ;
    const int64_t v81 = 1;
    // pto: %hc_fn__ssa_v0_pview
    ;
    const int64_t v82 = 1;
    // pto: %hc_fn__ssa_v0_pview
    ;
    const int64_t v83 = 1;
    // pto: %hc_fn__ssa_v0_pview
    ;
    int64_t v84 = v42 * v41;
    // pto: %hc_fn__ssa_v0_pview
    ;
    int64_t v85 = v83 * v84;
    // pto: %hc_fn__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, 24, -1> v86 = pto::Shape<1, 1, 1, 24, -1>(v81, v82, v83, v42, v45);
    // pto: %hc_fn__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v87 = pto::Stride<-1, -1, -1, -1, -1>(v82 * v85, v85, v84, v41, v40);
    // pto: %hc_fn__ssa_v0_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v88 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v80 + (v79 + v46 * v41 + v78 * v40), v86, v87);
    TLOAD(v76, v88);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    // pto: %weight_inline434__tile_t
    ;
    Tile<TileType::Mat, float, 1024, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v89 = Tile<TileType::Mat, float, 1024, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v45, v42);
    // pto: %weight_inline434__tile_t
    ;
    uint64_t v90 = (uint64_t) v35;
    TASSIGN(v89, v90);
    // pto: %product_inline487__tile_l0_init_storage
    ;
    Tile<TileType::Acc, float, 48, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v91 = Tile<TileType::Acc, float, 48, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v44, v39);
    // pto: %product_inline487__tile_l0_init_storage
    ;
    uint64_t v92 = (uint64_t) v36;
    TASSIGN(v91, v92);
    v91.SetValidShape(v44, v42);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int64_t v93 = v46; v93 < v45; v93 += v47) {
      // pto: %product_inline487__tile_l0_a
      ;
      Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v94 = Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v44, v48);
      // pto: %product_inline487__tile_l0_a
      ;
      uint64_t v95 = (uint64_t) v36;
      TASSIGN(v94, v95);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      uint16_t v96 = static_cast<uint16_t>(v46);
      uint16_t v97 = static_cast<uint16_t>(v93);
      TEXTRACT(v94, v75, v96, v97);
      // pto: %10
      ;
      int64_t v98 = (int64_t) ((uint64_t) v45 - (uint64_t) v93);
      // pto: %11
      ;
      int64_t v99 = v98 < v48 ? v98 : v48;
      // pto: %product_inline487__tile_l0_b
      ;
      Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v100 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v99, v42);
      // pto: %product_inline487__tile_l0_b
      ;
      uint64_t v101 = (uint64_t) v36;
      TASSIGN(v100, v101);
      uint16_t v102 = static_cast<uint16_t>(v93);
      uint16_t v103 = static_cast<uint16_t>(v46);
      TEXTRACT(v100, v89, v102, v103);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      // pto: %0
      ;
      Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v104 = Tile<TileType::Left, float, 48, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v44, v48);
      // pto: %0
      ;
      uint64_t v105 = (uint64_t) v37;
      TASSIGN(v104, v105);
      // pto: %12
      ;
      int64_t v106 = (int64_t) ((uint64_t) v93 + (uint64_t) v48);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      uint16_t v107 = static_cast<uint16_t>(v46);
      uint16_t v108 = static_cast<uint16_t>(v106);
      TEXTRACT(v104, v75, v107, v108);
      int64_t v109 = (int64_t) ((uint64_t) v33 - (uint64_t) v93);
      // pto: %15
      ;
      int64_t v110 = v109 < v46 ? v46 : v109;
      // pto: %16
      ;
      int64_t v111 = v110 < v48 ? v110 : v48;
      // pto: %1
      ;
      Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal> v112 = Tile<TileType::Right, float, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Normal>(v111, v42);
      // pto: %1
      ;
      uint64_t v113 = (uint64_t) v38;
      TASSIGN(v112, v113);
      uint16_t v114 = static_cast<uint16_t>(v106);
      uint16_t v115 = static_cast<uint16_t>(v46);
      TEXTRACT(v112, v89, v114, v115);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      // pto: %18
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v93 == v46) {
        pipe_barrier(PIPE_M);
        TMATMUL(v91, v94, v100);
      } else {
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v91, v91, v94, v100);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      // pto: %19
      ;
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      if (v93 == v49) {
        pipe_barrier(PIPE_M);
        TMATMUL(v91, v104, v112);
      } else {
        pipe_barrier(PIPE_M);
        TMATMUL_ACC(v91, v91, v104, v112);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    };
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TFREE<TPipe<0, Direction::DIR_V2C, 196608, 2, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v68);
    // pto: %20
    ;
    int64_t v116 = v72 < v46 ? v46 : v72;
    // pto: %mixes_partials_inline456__iter_v1_pview
    ;
    __gm__ float* v117 = PTOAS__GLOBAL_TENSOR_DATA(v58);
    // pto: %mixes_partials_inline456__iter_v1_pview
    ;
    const int64_t v118 = 0;
    // pto: %mixes_partials_inline456__iter_v1_pview
    ;
    const int64_t v119 = 32;
    // pto: %mixes_partials_inline456__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 48, 24> v120 = pto::Shape<1, 1, 1, 48, 24>();
    // pto: %mixes_partials_inline456__iter_v1_pview
    ;
    pto::Stride<1536, 1536, 1536, 32, 1> v121 = pto::Stride<1536, 1536, 1536, 32, 1>();
    // pto: %mixes_partials_inline456__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 48, 24>, pto::Stride<1536, 1536, 1536, 32, 1>, pto::Layout::ND> v122 = GlobalTensor<float, pto::Shape<1, 1, 1, 48, 24>, pto::Stride<1536, 1536, 1536, 32, 1>, pto::Layout::ND>(v117 + (v118 + v116 * v119), v120, v121);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v122, v91);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
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

AICORE void hc_medium48_aiv(__gm__ float* v1, __gm__ float* v2, __gm__ bfloat16_t* v3, __gm__ float* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, __gm__ float* v9, __gm__ float* v10, __gm__ float* v11, __gm__ bfloat16_t* v12, __gm__ float* v13, int64_t v14, int32_t v15, int64_t v16, int64_t v17, int64_t v18, int64_t v19, int64_t v20, int64_t v21, int64_t v22, int64_t v23, int64_t v24, int64_t v25, int64_t v26, int64_t v27, int64_t v28, int64_t v29, int64_t v30, int32_t v31, int32_t v32, int32_t v33) {
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  // pto: %c0_i64
  const int64_t v34 = 0;
  // pto: %c49152_i64
  const int64_t v35 = 49152;
  // pto: %c98304_i64
  const int64_t v36 = 98304;
  // pto: %c96_i64
  const int64_t v37 = 96;
  // pto: %c50176_i64
  const int64_t v38 = 50176;
  // pto: %c50688_i64
  const int64_t v39 = 50688;
  // pto: %c50240_i64
  const int64_t v40 = 50240;
  const int64_t v41 = 49184;
  // pto: %c49664_i64
  const int64_t v42 = 49664;
  // pto: %c49920_i64
  const int64_t v43 = 49920;
  // pto: %c50432_i64
  const int64_t v44 = 50432;
  // pto: %c49280_i64
  const int64_t v45 = 49280;
  // pto: %c49408_i64
  const int64_t v46 = 49408;
  // pto: %c49536_i64
  const int64_t v47 = 49536;
  // pto: %c49792_i64
  const int64_t v48 = 49792;
  // pto: %c50048_i64
  const int64_t v49 = 50048;
  // pto: %c50944_i64
  const int64_t v50 = 50944;
  // pto: %c50976_i64
  const int64_t v51 = 50976;
  // pto: %c51008_i64
  const int64_t v52 = 51008;
  // pto: %c51040_i64
  const int64_t v53 = 51040;
  // pto: %c50208_i64
  const int64_t v54 = 50208;
  // pto: %c16384_i64
  const int64_t v55 = 16384;
  // pto: %c65536_i64
  const int64_t v56 = 65536;
  // pto: %c81920_i64
  const int64_t v57 = 81920;
  // pto: %c90112_i64
  const int64_t v58 = 90112;
  // pto: %c114688_i64
  const int64_t v59 = 114688;
  // pto: %c131072_i64
  const int64_t v60 = 131072;
  // pto: %c139264_i64
  const int64_t v61 = 139264;
  // pto: %c8192_i64
  const int64_t v62 = 8192;
  // pto: %c32_index
  const int64_t v63 = 32;
  // pto: %c1_index
  const int64_t v64 = 1;
  // pto: %c16384_index
  const int64_t v65 = 16384;
  // pto: %c24_index
  const int64_t v66 = 24;
  // pto: %c3_index
  const int64_t v67 = 3;
  const int64_t v68 = 16;
  // pto: %c4096_index
  const int64_t v69 = 4096;
  // pto: %c0_i32
  const int32_t v70 = 0;
  // pto: %c48_index
  const int64_t v71 = 48;
  // pto: %c1024_index
  const int64_t v72 = 1024;
  // pto: %c0_index
  const int64_t v73 = 0;
  // pto: %cst_40
  const float v74 = 0.0f;
  // pto: %c2_index
  const int64_t v75 = 2;
  // pto: %c512_index
  const int64_t v76 = 512;
  // pto: %c256_index
  const int64_t v77 = 256;
  // pto: %c8_index
  const int64_t v78 = 8;
  // pto: %cst_45
  const float v79 = 6.10351563E-5f;
  // pto: %cst_46
  const float v80 = 9.99999997E-7f;
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
  // pto: %c64_index
  const int64_t v92 = 64;
  // pto: %c4_index
  const int64_t v93 = 4;
  // pto: %c8192_index
  const int64_t v94 = 8192;
  // pto: %c12288_index
  const int64_t v95 = 12288;
  // pto: %250
  int64_t v96 = (int64_t) ((uint64_t) v29 * (uint64_t) v68);
  // pto: %rms_partials_inline488__ssa_v0_view
  const int64_t v97 = 1;
  // pto: %rms_partials_inline488__ssa_v0_view
  const int64_t v98 = 1;
  // pto: %rms_partials_inline488__ssa_v0_view
  const int64_t v99 = 1;
  // pto: %rms_partials_inline488__ssa_v0_view
  int64_t v100 = v96 * v64;
  // pto: %rms_partials_inline488__ssa_v0_view
  int64_t v101 = v99 * v100;
  // pto: %rms_partials_inline488__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v102 = pto::Shape<1, 1, 1, -1, -1>(v97, v98, v99, v96, v64);
  // pto: %rms_partials_inline488__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v103 = pto::Stride<-1, -1, -1, -1, -1>(v98 * v101, v101, v100, v64, v96);
  // pto: %rms_partials_inline488__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v104 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v2, v102, v103);
  // pto: %x_flat_inline429__ssa_v0_view
  const int64_t v105 = 1;
  // pto: %x_flat_inline429__ssa_v0_view
  const int64_t v106 = 1;
  // pto: %x_flat_inline429__ssa_v0_view
  const int64_t v107 = 1;
  // pto: %x_flat_inline429__ssa_v0_view
  int64_t v108 = (int64_t) v30;
  // pto: %x_flat_inline429__ssa_v0_view
  int64_t v109 = v108 * v65;
  // pto: %x_flat_inline429__ssa_v0_view
  int64_t v110 = v107 * v109;
  // pto: %x_flat_inline429__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v111 = pto::Shape<1, 1, 1, -1, -1>(v105, v106, v107, v108, v65);
  // pto: %x_flat_inline429__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v112 = pto::Stride<-1, -1, -1, -1, -1>(v106 * v110, v110, v109, v65, v64);
  // pto: %x_flat_inline429__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v113 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v111, v112);
  // pto: %251
  int64_t v114 = (int64_t) ((uint64_t) v29 * (uint64_t) v63);
  // pto: %parts_2d_inline418__ssa_v0_view
  const int64_t v115 = 1;
  // pto: %parts_2d_inline418__ssa_v0_view
  const int64_t v116 = 1;
  // pto: %parts_2d_inline418__ssa_v0_view
  const int64_t v117 = 1;
  // pto: %parts_2d_inline418__ssa_v0_view
  int64_t v118 = v68 * v114;
  // pto: %parts_2d_inline418__ssa_v0_view
  int64_t v119 = v117 * v118;
  // pto: %parts_2d_inline418__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v120 = pto::Shape<1, 1, 1, -1, -1>(v115, v116, v117, v68, v114);
  // pto: %parts_2d_inline418__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v121 = pto::Stride<-1, -1, -1, -1, -1>(v116 * v119, v119, v118, v114, v64);
  // pto: %parts_2d_inline418__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v122 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v6, v120, v121);
  // pto: %rms_2d_inline566__ssa_v0_view
  const int64_t v123 = 1;
  // pto: %rms_2d_inline566__ssa_v0_view
  const int64_t v124 = 1;
  // pto: %rms_2d_inline566__ssa_v0_view
  const int64_t v125 = 1;
  // pto: %rms_2d_inline566__ssa_v0_view
  int64_t v126 = (int64_t) v29;
  // pto: %rms_2d_inline566__ssa_v0_view
  int64_t v127 = v68 * v126;
  // pto: %rms_2d_inline566__ssa_v0_view
  int64_t v128 = v125 * v127;
  // pto: %rms_2d_inline566__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v129 = pto::Shape<1, 1, 1, -1, -1>(v123, v124, v125, v68, (int64_t) v29);
  // pto: %rms_2d_inline566__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v130 = pto::Stride<-1, -1, -1, -1, -1>(v124 * v128, v128, v127, v126, v64);
  // pto: %rms_2d_inline566__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v131 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v7, v129, v130);
  // pto: %hc_base_2d_inline594__ssa_v0_view
  const int64_t v132 = 1;
  // pto: %hc_base_2d_inline594__ssa_v0_view
  const int64_t v133 = 1;
  // pto: %hc_base_2d_inline594__ssa_v0_view
  const int64_t v134 = 1;
  // pto: %hc_base_2d_inline594__ssa_v0_view
  int64_t v135 = v64 * v66;
  // pto: %hc_base_2d_inline594__ssa_v0_view
  int64_t v136 = v134 * v135;
  // pto: %hc_base_2d_inline594__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v137 = pto::Shape<1, 1, 1, -1, -1>(v132, v133, v134, v64, v66);
  // pto: %hc_base_2d_inline594__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v138 = pto::Stride<-1, -1, -1, -1, -1>(v133 * v136, v136, v135, v66, v64);
  // pto: %hc_base_2d_inline594__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v139 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v8, v137, v138);
  // pto: %comb_inline515__ssa_v0_view
  const int64_t v140 = 1;
  // pto: %comb_inline515__ssa_v0_view
  const int64_t v141 = 1;
  // pto: %comb_inline515__ssa_v0_view
  const int64_t v142 = 1;
  // pto: %comb_inline515__ssa_v0_view
  int64_t v143 = (int64_t) v30;
  // pto: %comb_inline515__ssa_v0_view
  int64_t v144 = v143 * v68;
  // pto: %comb_inline515__ssa_v0_view
  int64_t v145 = v142 * v144;
  // pto: %comb_inline515__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v146 = pto::Shape<1, 1, 1, -1, -1>(v140, v141, v142, v143, v68);
  // pto: %comb_inline515__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v147 = pto::Stride<-1, -1, -1, -1, -1>(v141 * v145, v145, v144, v68, v64);
  // pto: %comb_inline515__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v148 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v9, v146, v147);
  // pto: %hc_scale_2d_inline482__ssa_v0_view
  const int64_t v149 = 1;
  // pto: %hc_scale_2d_inline482__ssa_v0_view
  const int64_t v150 = 1;
  // pto: %hc_scale_2d_inline482__ssa_v0_view
  const int64_t v151 = 1;
  // pto: %hc_scale_2d_inline482__ssa_v0_view
  int64_t v152 = v64 * v67;
  // pto: %hc_scale_2d_inline482__ssa_v0_view
  int64_t v153 = v151 * v152;
  // pto: %hc_scale_2d_inline482__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v154 = pto::Shape<1, 1, 1, -1, -1>(v149, v150, v151, v64, v67);
  // pto: %hc_scale_2d_inline482__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v155 = pto::Stride<-1, -1, -1, -1, -1>(v150 * v153, v153, v152, v67, v64);
  // pto: %hc_scale_2d_inline482__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v156 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v10, v154, v155);
  // pto: %252
  int64_t v157 = (int64_t) ((uint64_t) v30 * (uint64_t) v93);
  // pto: %post_flat_inline392__ssa_v0_view
  const int64_t v158 = 1;
  // pto: %post_flat_inline392__ssa_v0_view
  const int64_t v159 = 1;
  // pto: %post_flat_inline392__ssa_v0_view
  const int64_t v160 = 1;
  // pto: %post_flat_inline392__ssa_v0_view
  int64_t v161 = v64 * v157;
  // pto: %post_flat_inline392__ssa_v0_view
  int64_t v162 = v160 * v161;
  // pto: %post_flat_inline392__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v163 = pto::Shape<1, 1, 1, -1, -1>(v158, v159, v160, v64, v157);
  // pto: %post_flat_inline392__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v164 = pto::Stride<-1, -1, -1, -1, -1>(v159 * v162, v162, v161, v157, v64);
  // pto: %post_flat_inline392__ssa_v0_view
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v165 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v11, v163, v164);
  // pto: %mixed_view_inline454__ssa_v0_view
  const int64_t v166 = 1;
  // pto: %mixed_view_inline454__ssa_v0_view
  const int64_t v167 = 1;
  // pto: %mixed_view_inline454__ssa_v0_view
  const int64_t v168 = 1;
  // pto: %mixed_view_inline454__ssa_v0_view
  int64_t v169 = (int64_t) v30;
  // pto: %mixed_view_inline454__ssa_v0_view
  int64_t v170 = v169 * v69;
  // pto: %mixed_view_inline454__ssa_v0_view
  int64_t v171 = v168 * v170;
  // pto: %mixed_view_inline454__ssa_v0_view
  pto::Shape<1, 1, 1, -1, -1> v172 = pto::Shape<1, 1, 1, -1, -1>(v166, v167, v168, v169, v69);
  // pto: %mixed_view_inline454__ssa_v0_view
  pto::Stride<-1, -1, -1, -1, -1> v173 = pto::Stride<-1, -1, -1, -1, -1>(v167 * v171, v171, v170, v69, v64);
  // pto: %mixed_view_inline454__ssa_v0_view
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v174 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v12, v172, v173);
  const int32_t v175 = 0;
  auto v176 = TPipe<0, Direction::DIR_V2C, 196608, 2, 2, false>(v13, v175, v70);
  // pto: %worker_inline463__ssa_v0
  int64_t v177 = (int64_t) v31;
  // pto: %linear_workers_inline445__ssa_v0_idx
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
  set_flag(PIPE_S, PIPE_V, EVENT_ID0);
  set_flag(PIPE_S, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  for (int64_t v178 = v177; v178 < v14; v178 += (int64_t) v15) {
    // pto: %39, %40
    ;
    int64_t v179 = (int64_t) ((uint64_t) (v178 / v68) * (uint64_t) v71);
    // pto: %41
    ;
    int64_t v180 = v178 % v68;
    // pto: %42
    ;
    int64_t v181 = (int64_t) ((uint64_t) v180 * (uint64_t) v72);
    // pto: %lane_inline438__ssa_v0, %45
    ;
    int64_t v182 = (int64_t) ((uint64_t) ((int64_t) v33) * (uint64_t) v66);
    // pto: %46
    ;
    int64_t v183 = (int64_t) ((uint64_t) v179 + (uint64_t) v182);
    // pto: %47
    ;
    int64_t v184 = (int64_t) ((uint64_t) v17 - (uint64_t) v183);
    // pto: %48
    ;
    int64_t v185 = v184 < v66 ? v184 : v66;
    // pto: %49
    ;
    int64_t v186 = v185 < v73 ? v73 : v185;
    // pto: %source_inline374__tile
    ;
    Tile<TileType::Vec, bfloat16_t, 24, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v187 = Tile<TileType::Vec, bfloat16_t, 24, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v186, v72);
    // pto: %source_inline374__tile
    ;
    uint64_t v188 = (uint64_t) v34;
    TASSIGN(v187, v188);
    // pto: %50
    ;
    int64_t v189 = v183 < v73 ? v73 : v183;
    // pto: %51
    ;
    int64_t v190 = v181 < v73 ? v73 : v181;
    // pto: %x_flat_inline429__ssa_v0_pview
    ;
    const int64_t v191 = 0;
    // pto: %x_flat_inline429__ssa_v0_pview
    ;
    __gm__ bfloat16_t* v192 = PTOAS__GLOBAL_TENSOR_DATA(v113);
    // pto: %x_flat_inline429__ssa_v0_pview
    ;
    const int64_t v193 = 1;
    // pto: %x_flat_inline429__ssa_v0_pview
    ;
    const int64_t v194 = 1;
    // pto: %x_flat_inline429__ssa_v0_pview
    ;
    const int64_t v195 = 1;
    // pto: %x_flat_inline429__ssa_v0_pview
    ;
    int64_t v196 = v186 * v65;
    // pto: %x_flat_inline429__ssa_v0_pview
    ;
    int64_t v197 = v195 * v196;
    // pto: %x_flat_inline429__ssa_v0_pview
    ;
    pto::Shape<1, 1, 1, -1, -1> v198 = pto::Shape<1, 1, 1, -1, -1>(v193, v194, v195, v186, v72);
    // pto: %x_flat_inline429__ssa_v0_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v199 = pto::Stride<-1, -1, -1, -1, -1>(v194 * v197, v197, v196, v65, v64);
    // pto: %x_flat_inline429__ssa_v0_pview
    ;
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v200 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v192 + (v191 + v189 * v65 + v190 * v64), v198, v199);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v187, v200);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // pto: %fp32_inline372__tile
    ;
    Tile<TileType::Vec, float, 24, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v201 = Tile<TileType::Vec, float, 24, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v186, v72);
    // pto: %fp32_inline372__tile
    ;
    uint64_t v202 = (uint64_t) v35;
    TASSIGN(v201, v202);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    RoundMode v203 = RoundMode::CAST_ROUND;
    SaturationMode v204 = SaturationMode::ON;
    TCVT(v201, v187, v203, v204);
    v201.SetValidShape(v186, v72);
    // pto: %clean_inline371__tile
    ;
    Tile<TileType::Vec, float, 24, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v205 = Tile<TileType::Vec, float, 24, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v66, v72);
    // pto: %clean_inline371__tile
    ;
    uint64_t v206 = (uint64_t) v35;
    TASSIGN(v205, v206);
    pipe_barrier(PIPE_V);
    TFILLPAD<pto::TFillPadMode::InPlace>(v205, v201);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    v205.SetValidShape(v66, v72);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TPUSH<TPipe<0, Direction::DIR_V2C, 196608, 2, 2, false>, Tile<TileType::Vec, float, 24, 1024, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v176, v205);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    // pto: %local_sum_inline370__tile
    ;
    Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v207 = Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v66);
    // pto: %local_sum_inline370__tile
    ;
    uint64_t v208 = (uint64_t) v34;
    TASSIGN(v207, v208);
    TEXPANDS(v207, v74);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    for (int64_t v209 = v73; v209 < v75; v209 += v64) {
      // pto: %xr_inline368__tile
      ;
      Tile<TileType::Vec, bfloat16_t, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v210 = Tile<TileType::Vec, bfloat16_t, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v186, v76);
      // pto: %xr_inline368__tile
      ;
      uint64_t v211 = (uint64_t) v36;
      TASSIGN(v210, v211);
      // pto: %54, %52
      ;
      int64_t v212 = (int64_t) ((uint64_t) v181 + (uint64_t) ((int64_t) (uint64_t) v209 * (uint64_t) v76));
      // pto: %55
      ;
      int64_t v213 = v212 < v73 ? v73 : v212;
      // pto: %56
      ;
      const int64_t v214 = 0;
      // pto: %56
      ;
      __gm__ bfloat16_t* v215 = PTOAS__GLOBAL_TENSOR_DATA(v113);
      // pto: %56
      ;
      const int64_t v216 = 1;
      // pto: %56
      ;
      const int64_t v217 = 1;
      // pto: %56
      ;
      const int64_t v218 = 1;
      // pto: %56
      ;
      int64_t v219 = v186 * v65;
      // pto: %56
      ;
      int64_t v220 = v218 * v219;
      // pto: %56
      ;
      pto::Shape<1, 1, 1, -1, -1> v221 = pto::Shape<1, 1, 1, -1, -1>(v216, v217, v218, v186, v76);
      // pto: %56
      ;
      pto::Stride<-1, -1, -1, -1, -1> v222 = pto::Stride<-1, -1, -1, -1, -1>(v217 * v220, v220, v219, v65, v64);
      // pto: %56
      ;
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v223 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v215 + (v214 + v189 * v65 + v213 * v64), v221, v222);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v210, v223);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      // pto: %xr32_inline565__tile
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v224 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v186, v76);
      // pto: %xr32_inline565__tile
      ;
      uint64_t v225 = (uint64_t) v35;
      TASSIGN(v224, v225);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      RoundMode v226 = RoundMode::CAST_ROUND;
      SaturationMode v227 = SaturationMode::ON;
      TCVT(v224, v210, v226, v227);
      v224.SetValidShape(v186, v76);
      // pto: %xr_clean_inline367__tile
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v228 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v66, v76);
      // pto: %xr_clean_inline367__tile
      ;
      uint64_t v229 = (uint64_t) v35;
      TASSIGN(v228, v229);
      pipe_barrier(PIPE_V);
      TFILLPAD<pto::TFillPadMode::InPlace>(v228, v224);
      v228.SetValidShape(v66, v76);
      // pto: %1
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v230 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v66, v76);
      // pto: %1
      ;
      uint64_t v231 = (uint64_t) v35;
      TASSIGN(v230, v231);
      pipe_barrier(PIPE_V);
      TMUL(v230, v228, v228);
      // pto: %tmp_tile
      ;
      Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v232 = Tile<TileType::Vec, float, 24, 512, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v76);
      // pto: %tmp_tile
      ;
      uint64_t v233 = (uint64_t) v36;
      TASSIGN(v232, v233);
      // pto: %2
      ;
      Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v234 = Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v64);
      // pto: %2
      ;
      uint64_t v235 = (uint64_t) v37;
      TASSIGN(v234, v235);
      pipe_barrier(PIPE_V);
      TROWSUM(v234, v230, v232);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      // pto: %rms_sums_inline405__tile
      ;
      Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v236 = Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v66);
      // pto: %rms_sums_inline405__tile
      ;
      uint64_t v237 = (uint64_t) v37;
      TASSIGN(v236, v237);
      // pto: %3
      ;
      Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v238 = Tile<TileType::Vec, float, 1, 24, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v66);
      // pto: %3
      ;
      uint64_t v239 = (uint64_t) v34;
      TASSIGN(v238, v239);
      pipe_barrier(PIPE_V);
      TADD(v238, v207, v236);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    // pto: %43, %44, %58
    ;
    int64_t v240 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v180 * (uint64_t) v16) + (uint64_t) v179) + (uint64_t) v182);
    // pto: %4
    ;
    Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v241 = Tile<TileType::Vec, float, 24, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v66, v64);
    // pto: %4
    ;
    uint64_t v242 = (uint64_t) v34;
    TASSIGN(v241, v242);
    // pto: %59
    ;
    int64_t v243 = v240 < v73 ? v73 : v240;
    // pto: %rms_partials_inline488__iter_v1_pview
    ;
    const int64_t v244 = 0;
    // pto: %rms_partials_inline488__iter_v1_pview
    ;
    __gm__ float* v245 = PTOAS__GLOBAL_TENSOR_DATA(v104);
    // pto: %rms_partials_inline488__iter_v1_pview
    ;
    const int64_t v246 = 1;
    // pto: %rms_partials_inline488__iter_v1_pview
    ;
    const int64_t v247 = 1;
    // pto: %rms_partials_inline488__iter_v1_pview
    ;
    const int64_t v248 = 1;
    // pto: %rms_partials_inline488__iter_v1_pview
    ;
    int64_t v249 = v66 * v64;
    // pto: %rms_partials_inline488__iter_v1_pview
    ;
    int64_t v250 = v248 * v249;
    // pto: %rms_partials_inline488__iter_v1_pview
    ;
    pto::Shape<1, 1, 1, 24, 1> v251 = pto::Shape<1, 1, 1, 24, 1>(v246, v247, v248, v66, v64);
    // pto: %rms_partials_inline488__iter_v1_pview
    ;
    pto::Stride<-1, -1, -1, -1, -1> v252 = pto::Stride<-1, -1, -1, -1, -1>(v247 * v250, v250, v249, v64, v96);
    // pto: %rms_partials_inline488__iter_v1_pview
    ;
    GlobalTensor<float, pto::Shape<1, 1, 1, 24, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN> v253 = GlobalTensor<float, pto::Shape<1, 1, 1, 24, 1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::DN>(v245 + (v244 + v243 * v64 + v73 * v96), v251, v252);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    TSTORE(v253, v241);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  pipe_barrier(PIPE_ALL);
  dsb(DSB_DDR);
  SYNCALL<SyncCoreType::Mix>();
  dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE);
  // pto: %60, %61, %tail_lane_inline530__ssa_v0
  int64_t v254 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v177 * (uint64_t) v75) + (uint64_t) ((int64_t) v33));
  // pto: %62
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  if (v254 < v18) {
    // pto: %64
    ;
    int64_t v255 = v254 < v20 ? v254 : v20;
    // pto: %63, %65
    ;
    int64_t v256 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v254 * (uint64_t) v19) + (uint64_t) v255);
    // pto: %67
    ;
    int64_t v257 = (int64_t) ((uint64_t) v20 - (uint64_t) v254);
    // pto: %68
    ;
    int64_t v258 = v257 < v73 ? v73 : v257;
    // pto: %69
    ;
    int64_t v259 = v258 < v64 ? v258 : v64;
    // pto: %66, %70
    ;
    int64_t v260 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v256 + (uint64_t) v19) + (uint64_t) v259);
    for (int64_t v261 = v256; v261 < v260; v261 += v75) {
      // pto: %scale2_inline389__tile
      ;
      float v262 = (v5)[v75];
      // pto: %71
      ;
      int64_t v263 = (int64_t) ((uint64_t) v260 - (uint64_t) v261);
      // pto: %72
      ;
      int64_t v264 = v263 < v75 ? v263 : v75;
      // pto: %73
      ;
      int64_t v265 = (int64_t) ((uint64_t) v264 * (uint64_t) v63);
      // pto: %c_parts_load_inline379__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v266 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v265);
      // pto: %c_parts_load_inline379__ssa_v0
      ;
      uint64_t v267 = (uint64_t) v35;
      TASSIGN(v266, v267);
      // pto: %74
      ;
      int64_t v268 = (int64_t) ((uint64_t) v261 * (uint64_t) v63);
      // pto: %75
      ;
      int64_t v269 = v268 < v73 ? v73 : v268;
      // pto: %parts_2d_inline418__ssa_v0_pview
      ;
      const int64_t v270 = 0;
      // pto: %parts_2d_inline418__ssa_v0_pview
      ;
      __gm__ float* v271 = PTOAS__GLOBAL_TENSOR_DATA(v122);
      // pto: %parts_2d_inline418__ssa_v0_pview
      ;
      const int64_t v272 = 1;
      // pto: %parts_2d_inline418__ssa_v0_pview
      ;
      const int64_t v273 = 1;
      // pto: %parts_2d_inline418__ssa_v0_pview
      ;
      const int64_t v274 = 1;
      // pto: %parts_2d_inline418__ssa_v0_pview
      ;
      int64_t v275 = v68 * v114;
      // pto: %parts_2d_inline418__ssa_v0_pview
      ;
      int64_t v276 = v274 * v275;
      // pto: %parts_2d_inline418__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 16, -1> v277 = pto::Shape<1, 1, 1, 16, -1>(v272, v273, v274, v68, v265);
      // pto: %parts_2d_inline418__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v278 = pto::Stride<-1, -1, -1, -1, -1>(v273 * v276, v276, v275, v114, v64);
      // pto: %parts_2d_inline418__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v279 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v271 + (v270 + v73 * v114 + v269 * v64), v277, v278);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
      TLOAD(v266, v279);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      // pto: %t__tmp_v95
      ;
      Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v280 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v68, v77);
      // pto: %t__tmp_v95
      ;
      uint64_t v281 = (uint64_t) v35;
      TASSIGN(v280, v281);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      TFILLPAD<pto::TFillPadMode::InPlace>(v280, v266);
      v280.SetValidShape(v68, v77);
      // pto: %t__tmp_v96
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v282 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v77);
      // pto: %t__tmp_v96
      ;
      uint64_t v283 = (uint64_t) v35;
      TASSIGN(v282, v283);
      pipe_barrier(PIPE_V);
      TCOLSUM(v282, v280);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
      // pto: %c_total_inline358__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v284 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v63);
      // pto: %c_total_inline358__ssa_v0
      ;
      uint64_t v285 = (uint64_t) v35;
      TASSIGN(v284, v285);
      // pto: %c_sq_load_inline419__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v286 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v264);
      // pto: %c_sq_load_inline419__ssa_v0
      ;
      uint64_t v287 = (uint64_t) v38;
      TASSIGN(v286, v287);
      // pto: %77
      ;
      int64_t v288 = v261 < v73 ? v73 : v261;
      // pto: %rms_2d_inline566__ssa_v0_pview
      ;
      int64_t v289 = (int64_t) v29;
      // pto: %rms_2d_inline566__ssa_v0_pview
      ;
      const int64_t v290 = 0;
      // pto: %rms_2d_inline566__ssa_v0_pview
      ;
      __gm__ float* v291 = PTOAS__GLOBAL_TENSOR_DATA(v131);
      // pto: %rms_2d_inline566__ssa_v0_pview
      ;
      const int64_t v292 = 1;
      // pto: %rms_2d_inline566__ssa_v0_pview
      ;
      const int64_t v293 = 1;
      // pto: %rms_2d_inline566__ssa_v0_pview
      ;
      const int64_t v294 = 1;
      // pto: %rms_2d_inline566__ssa_v0_pview
      ;
      int64_t v295 = v68 * v289;
      // pto: %rms_2d_inline566__ssa_v0_pview
      ;
      int64_t v296 = v294 * v295;
      // pto: %rms_2d_inline566__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 16, -1> v297 = pto::Shape<1, 1, 1, 16, -1>(v292, v293, v294, v68, v264);
      // pto: %rms_2d_inline566__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v298 = pto::Stride<-1, -1, -1, -1, -1>(v293 * v296, v296, v295, v289, v64);
      // pto: %rms_2d_inline566__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v299 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v291 + (v290 + v73 * v289 + v288 * v64), v297, v298);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
      TLOAD(v286, v299);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      // pto: %t__tmp_v97
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v300 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v68, v78);
      // pto: %t__tmp_v97
      ;
      uint64_t v301 = (uint64_t) v38;
      TASSIGN(v300, v301);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      TFILLPAD<pto::TFillPadMode::InPlace>(v300, v286);
      v300.SetValidShape(v68, v78);
      // pto: %c_rms_tmp_inline492__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v302 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v78);
      // pto: %c_rms_tmp_inline492__ssa_v0
      ;
      uint64_t v303 = (uint64_t) v39;
      TASSIGN(v302, v303);
      // pto: %t__tmp_v98
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v304 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v78);
      // pto: %t__tmp_v98
      ;
      uint64_t v305 = (uint64_t) v38;
      TASSIGN(v304, v305);
      pipe_barrier(PIPE_V);
      TCOLSUM(v304, v300);
      // pto: %t__tmp_v99
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v306 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v78);
      // pto: %t__tmp_v99
      ;
      uint64_t v307 = (uint64_t) v38;
      TASSIGN(v306, v307);
      pipe_barrier(PIPE_V);
      TMULS(v306, v304, v79);
      // pto: %t__tmp_v100
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v308 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v78);
      // pto: %t__tmp_v100
      ;
      uint64_t v309 = (uint64_t) v38;
      TASSIGN(v308, v309);
      pipe_barrier(PIPE_V);
      TADDS(v308, v306, v80);
      // pto: %t__tmp_v101
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v310 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v78);
      // pto: %t__tmp_v101
      ;
      uint64_t v311 = (uint64_t) v40;
      TASSIGN(v310, v311);
      pipe_barrier(PIPE_V);
      TRSQRT(v310, v308, v302);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      // pto: %c_inv_inline356__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v312 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v64);
      // pto: %c_inv_inline356__ssa_v0
      ;
      uint64_t v313 = (uint64_t) v40;
      TASSIGN(v312, v313);
      // pto: %comb_inv_inline412__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v314 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v64);
      // pto: %comb_inv_inline412__ssa_v0
      ;
      uint64_t v315 = (uint64_t) v40;
      TASSIGN(v314, v315);
      // pto: %t__tmp_v102
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v316 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v68);
      // pto: %t__tmp_v102
      ;
      uint64_t v317 = (uint64_t) v41;
      TASSIGN(v316, v317);
      // pto: %slice_view
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 16, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v318;
      // pto: %slice_view
      ;
      uint64_t v319 = (uint64_t) v41;
      TASSIGN(v318, v319);
      // pto: %raw_inline385__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v320 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v68);
      // pto: %raw_inline385__ssa_v0
      ;
      uint64_t v321 = (uint64_t) v35;
      TASSIGN(v320, v321);
      TADDS(v320, v318, v74);
      // pto: %base_inline521__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v322 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v68);
      // pto: %base_inline521__ssa_v0
      ;
      uint64_t v323 = (uint64_t) v38;
      TASSIGN(v322, v323);
      // pto: %hc_base_2d_inline594__ssa_v0_pview
      ;
      __gm__ float* v324 = PTOAS__GLOBAL_TENSOR_DATA(v139);
      // pto: %hc_base_2d_inline594__ssa_v0_pview
      ;
      unsigned v325 = 8;
      // pto: %hc_base_2d_inline594__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 16> v326 = pto::Shape<1, 1, 1, 1, 16>();
      // pto: %hc_base_2d_inline594__ssa_v0_pview
      ;
      pto::Stride<24, 24, 24, 24, 1> v327 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %hc_base_2d_inline594__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v328 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v324 + v325, v326, v327);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v322, v328);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      // pto: %t__tmp_v103
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v329 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v68);
      // pto: %t__tmp_v103
      ;
      uint64_t v330 = (uint64_t) v42;
      TASSIGN(v329, v330);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v329, v320, v314);
      // pto: %t__tmp_v104
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v331 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v68);
      // pto: %t__tmp_v104
      ;
      uint64_t v332 = (uint64_t) v42;
      TASSIGN(v331, v332);
      pipe_barrier(PIPE_V);
      TMULS(v331, v329, v262);
      // pto: %t__tmp_v105
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v333 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v68);
      // pto: %t__tmp_v105
      ;
      uint64_t v334 = (uint64_t) v35;
      TASSIGN(v333, v334);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      TCOLEXPAND(v333, v322);
      // pto: %logits_inline409__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v335 = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v68);
      // pto: %logits_inline409__ssa_v0
      ;
      uint64_t v336 = (uint64_t) v43;
      TASSIGN(v335, v336);
      pipe_barrier(PIPE_V);
      TADD(v335, v331, v333);
      // pto: %t__tmp_v106
      ;
      Tile<TileType::Vec, float, 2, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v337 = Tile<TileType::Vec, float, 2, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v75, v68);
      // pto: %t__tmp_v106
      ;
      uint64_t v338 = (uint64_t) v43;
      TASSIGN(v337, v338);
      // pto: %regular_logits_inline568__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v339 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %regular_logits_inline568__ssa_v0
      ;
      uint64_t v340 = (uint64_t) v43;
      TASSIGN(v339, v340);
      // pto: %ids_inline425__ci_tmp_v0
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v341 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %ids_inline425__ci_tmp_v0
      ;
      uint64_t v342 = (uint64_t) v35;
      TASSIGN(v341, v342);
      // pto: %ids_inline425__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v343 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %ids_inline425__ssa_v1
      ;
      uint64_t v344 = (uint64_t) v44;
      TASSIGN(v343, v344);
      // pto: %ci_tmp_view
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v345;
      TRESHAPE(v345, v341);
      // pto: %ci_dst_view
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v346;
      TRESHAPE(v346, v343);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v346, v70, v345);
      // pto: %tmp_inline524__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v347 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %tmp_inline524__ssa_v1
      ;
      uint64_t v348 = (uint64_t) v35;
      TASSIGN(v347, v348);
      // pto: %idx_float_inline443__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v349 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %idx_float_inline443__ssa_v0
      ;
      uint64_t v350 = (uint64_t) v45;
      TASSIGN(v349, v350);
      pipe_barrier(PIPE_V);
      RoundMode v351 = RoundMode::CAST_ROUND;
      SaturationMode v352 = SaturationMode::ON;
      TCVT(v349, v343, v351, v352);
      // pto: %t__tmp_v107
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v353 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v107
      ;
      uint64_t v354 = (uint64_t) v46;
      TASSIGN(v353, v354);
      pipe_barrier(PIPE_V);
      TMULS(v353, v349, v82);
      // pto: %t__tmp_v108
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v355 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v108
      ;
      uint64_t v356 = (uint64_t) v46;
      TASSIGN(v355, v356);
      pipe_barrier(PIPE_V);
      RoundMode v357 = RoundMode::CAST_FLOOR;
      SaturationMode v358 = SaturationMode::ON;
      TCVT(v355, v353, v357, v358);
      // pto: %ri1_low_inline570__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v359 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %ri1_low_inline570__ssa_v0
      ;
      uint64_t v360 = (uint64_t) v46;
      TASSIGN(v359, v360);
      pipe_barrier(PIPE_V);
      RoundMode v361 = RoundMode::CAST_ROUND;
      SaturationMode v362 = SaturationMode::ON;
      TCVT(v359, v355, v361, v362);
      // pto: %t__tmp_v109
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v363 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v109
      ;
      uint64_t v364 = (uint64_t) v47;
      TASSIGN(v363, v364);
      TMULS(v363, v349, v83);
      // pto: %t__tmp_v110
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v365 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v110
      ;
      uint64_t v366 = (uint64_t) v47;
      TASSIGN(v365, v366);
      pipe_barrier(PIPE_V);
      RoundMode v367 = RoundMode::CAST_FLOOR;
      SaturationMode v368 = SaturationMode::ON;
      TCVT(v365, v363, v367, v368);
      // pto: %ri1_high_inline571__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v369 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %ri1_high_inline571__ssa_v0
      ;
      uint64_t v370 = (uint64_t) v47;
      TASSIGN(v369, v370);
      pipe_barrier(PIPE_V);
      RoundMode v371 = RoundMode::CAST_ROUND;
      SaturationMode v372 = SaturationMode::ON;
      TCVT(v369, v365, v371, v372);
      // pto: %t__tmp_v111
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v373 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v111
      ;
      uint64_t v374 = (uint64_t) v47;
      TASSIGN(v373, v374);
      pipe_barrier(PIPE_V);
      TMULS(v373, v369, v84);
      // pto: %ri1_bit_inline572__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v375 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %ri1_bit_inline572__ssa_v0
      ;
      uint64_t v376 = (uint64_t) v46;
      TASSIGN(v375, v376);
      pipe_barrier(PIPE_V);
      TSUB(v375, v359, v373);
      // pto: %t__tmp_v112
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v377 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v112
      ;
      uint64_t v378 = (uint64_t) v46;
      TASSIGN(v377, v378);
      pipe_barrier(PIPE_V);
      TMULS(v377, v375, v84);
      // pto: %t__tmp_v113
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v379 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v113
      ;
      uint64_t v380 = (uint64_t) v46;
      TASSIGN(v379, v380);
      pipe_barrier(PIPE_V);
      TSUB(v379, v349, v377);
      // pto: %t__tmp_v114
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v381 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v114
      ;
      uint64_t v382 = (uint64_t) v46;
      TASSIGN(v381, v382);
      pipe_barrier(PIPE_V);
      TADDS(v381, v379, v82);
      // pto: %ri1_inline537__ssa_v1
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v383 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %ri1_inline537__ssa_v1
      ;
      uint64_t v384 = (uint64_t) v46;
      TASSIGN(v383, v384);
      pipe_barrier(PIPE_V);
      RoundMode v385 = RoundMode::CAST_ROUND;
      SaturationMode v386 = SaturationMode::ON;
      TCVT(v383, v381, v385, v386);
      // pto: %t__tmp_v115
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v387 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v115
      ;
      uint64_t v388 = (uint64_t) v47;
      TASSIGN(v387, v388);
      TMULS(v387, v349, v83);
      // pto: %t__tmp_v116
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v389 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v116
      ;
      uint64_t v390 = (uint64_t) v47;
      TASSIGN(v389, v390);
      pipe_barrier(PIPE_V);
      RoundMode v391 = RoundMode::CAST_FLOOR;
      SaturationMode v392 = SaturationMode::ON;
      TCVT(v389, v387, v391, v392);
      // pto: %ri2_low_inline575__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v393 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %ri2_low_inline575__ssa_v0
      ;
      uint64_t v394 = (uint64_t) v47;
      TASSIGN(v393, v394);
      pipe_barrier(PIPE_V);
      RoundMode v395 = RoundMode::CAST_ROUND;
      SaturationMode v396 = SaturationMode::ON;
      TCVT(v393, v389, v395, v396);
      // pto: %t__tmp_v117
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v397 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v117
      ;
      uint64_t v398 = (uint64_t) v42;
      TASSIGN(v397, v398);
      TMULS(v397, v349, v85);
      // pto: %t__tmp_v118
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v399 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v118
      ;
      uint64_t v400 = (uint64_t) v42;
      TASSIGN(v399, v400);
      pipe_barrier(PIPE_V);
      RoundMode v401 = RoundMode::CAST_FLOOR;
      SaturationMode v402 = SaturationMode::ON;
      TCVT(v399, v397, v401, v402);
      // pto: %ri2_high_inline390__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v403 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %ri2_high_inline390__ssa_v0
      ;
      uint64_t v404 = (uint64_t) v42;
      TASSIGN(v403, v404);
      pipe_barrier(PIPE_V);
      RoundMode v405 = RoundMode::CAST_ROUND;
      SaturationMode v406 = SaturationMode::ON;
      TCVT(v403, v399, v405, v406);
      // pto: %t__tmp_v119
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v407 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v119
      ;
      uint64_t v408 = (uint64_t) v42;
      TASSIGN(v407, v408);
      pipe_barrier(PIPE_V);
      TMULS(v407, v403, v84);
      // pto: %ri2_bit_inline441__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v409 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %ri2_bit_inline441__ssa_v0
      ;
      uint64_t v410 = (uint64_t) v47;
      TASSIGN(v409, v410);
      pipe_barrier(PIPE_V);
      TSUB(v409, v393, v407);
      // pto: %t__tmp_v120
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v411 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v120
      ;
      uint64_t v412 = (uint64_t) v47;
      TASSIGN(v411, v412);
      pipe_barrier(PIPE_V);
      TMULS(v411, v409, v86);
      // pto: %t__tmp_v121
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v413 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v121
      ;
      uint64_t v414 = (uint64_t) v47;
      TASSIGN(v413, v414);
      pipe_barrier(PIPE_V);
      TSUB(v413, v349, v411);
      // pto: %t__tmp_v122
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v415 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v122
      ;
      uint64_t v416 = (uint64_t) v47;
      TASSIGN(v415, v416);
      pipe_barrier(PIPE_V);
      TADDS(v415, v413, v84);
      // pto: %ri2_inline576__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v417 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %ri2_inline576__ssa_v0
      ;
      uint64_t v418 = (uint64_t) v47;
      TASSIGN(v417, v418);
      pipe_barrier(PIPE_V);
      RoundMode v419 = RoundMode::CAST_ROUND;
      SaturationMode v420 = SaturationMode::ON;
      TCVT(v417, v415, v419, v420);
      // pto: %t__tmp_v123
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v421 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v123
      ;
      uint64_t v422 = (uint64_t) v42;
      TASSIGN(v421, v422);
      TMULS(v421, v349, v85);
      // pto: %t__tmp_v124
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v423 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v124
      ;
      uint64_t v424 = (uint64_t) v42;
      TASSIGN(v423, v424);
      pipe_barrier(PIPE_V);
      RoundMode v425 = RoundMode::CAST_FLOOR;
      SaturationMode v426 = SaturationMode::ON;
      TCVT(v423, v421, v425, v426);
      // pto: %ci1_low_inline577__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v427 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %ci1_low_inline577__ssa_v0
      ;
      uint64_t v428 = (uint64_t) v42;
      TASSIGN(v427, v428);
      pipe_barrier(PIPE_V);
      RoundMode v429 = RoundMode::CAST_ROUND;
      SaturationMode v430 = SaturationMode::ON;
      TCVT(v427, v423, v429, v430);
      // pto: %t__tmp_v125
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v431 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v125
      ;
      uint64_t v432 = (uint64_t) v48;
      TASSIGN(v431, v432);
      TMULS(v431, v349, v87);
      // pto: %t__tmp_v126
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v433 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v126
      ;
      uint64_t v434 = (uint64_t) v48;
      TASSIGN(v433, v434);
      pipe_barrier(PIPE_V);
      RoundMode v435 = RoundMode::CAST_FLOOR;
      SaturationMode v436 = SaturationMode::ON;
      TCVT(v433, v431, v435, v436);
      // pto: %ci1_high_inline581__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v437 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %ci1_high_inline581__ssa_v0
      ;
      uint64_t v438 = (uint64_t) v48;
      TASSIGN(v437, v438);
      pipe_barrier(PIPE_V);
      RoundMode v439 = RoundMode::CAST_ROUND;
      SaturationMode v440 = SaturationMode::ON;
      TCVT(v437, v433, v439, v440);
      // pto: %t__tmp_v127
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v441 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v127
      ;
      uint64_t v442 = (uint64_t) v48;
      TASSIGN(v441, v442);
      pipe_barrier(PIPE_V);
      TMULS(v441, v437, v84);
      // pto: %ci1_bit_inline582__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v443 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %ci1_bit_inline582__ssa_v0
      ;
      uint64_t v444 = (uint64_t) v42;
      TASSIGN(v443, v444);
      pipe_barrier(PIPE_V);
      TSUB(v443, v427, v441);
      // pto: %t__tmp_v128
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v445 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v128
      ;
      uint64_t v446 = (uint64_t) v42;
      TASSIGN(v445, v446);
      pipe_barrier(PIPE_V);
      TMULS(v445, v443, v88);
      // pto: %t__tmp_v129
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v447 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v129
      ;
      uint64_t v448 = (uint64_t) v42;
      TASSIGN(v447, v448);
      pipe_barrier(PIPE_V);
      TSUB(v447, v349, v445);
      // pto: %t__tmp_v130
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v449 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v130
      ;
      uint64_t v450 = (uint64_t) v42;
      TASSIGN(v449, v450);
      pipe_barrier(PIPE_V);
      TADDS(v449, v447, v86);
      // pto: %ci1_inline584__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v451 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %ci1_inline584__ssa_v0
      ;
      uint64_t v452 = (uint64_t) v42;
      TASSIGN(v451, v452);
      pipe_barrier(PIPE_V);
      RoundMode v453 = RoundMode::CAST_ROUND;
      SaturationMode v454 = SaturationMode::ON;
      TCVT(v451, v449, v453, v454);
      // pto: %t__tmp_v131
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v455 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v131
      ;
      uint64_t v456 = (uint64_t) v48;
      TASSIGN(v455, v456);
      TMULS(v455, v349, v87);
      // pto: %t__tmp_v132
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v457 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v132
      ;
      uint64_t v458 = (uint64_t) v48;
      TASSIGN(v457, v458);
      pipe_barrier(PIPE_V);
      RoundMode v459 = RoundMode::CAST_FLOOR;
      SaturationMode v460 = SaturationMode::ON;
      TCVT(v457, v455, v459, v460);
      // pto: %ci2_low_inline574__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v461 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %ci2_low_inline574__ssa_v0
      ;
      uint64_t v462 = (uint64_t) v48;
      TASSIGN(v461, v462);
      pipe_barrier(PIPE_V);
      RoundMode v463 = RoundMode::CAST_ROUND;
      SaturationMode v464 = SaturationMode::ON;
      TCVT(v461, v457, v463, v464);
      // pto: %t__tmp_v133
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v465 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v133
      ;
      uint64_t v466 = (uint64_t) v44;
      TASSIGN(v465, v466);
      TMULS(v465, v349, v89);
      // pto: %t__tmp_v134
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v467 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v134
      ;
      uint64_t v468 = (uint64_t) v44;
      TASSIGN(v467, v468);
      pipe_barrier(PIPE_V);
      RoundMode v469 = RoundMode::CAST_FLOOR;
      SaturationMode v470 = SaturationMode::ON;
      TCVT(v467, v465, v469, v470);
      // pto: %ci2_high_inline435__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v471 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %ci2_high_inline435__ssa_v0
      ;
      uint64_t v472 = (uint64_t) v44;
      TASSIGN(v471, v472);
      pipe_barrier(PIPE_V);
      RoundMode v473 = RoundMode::CAST_ROUND;
      SaturationMode v474 = SaturationMode::ON;
      TCVT(v471, v467, v473, v474);
      // pto: %t__tmp_v135
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v475 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v135
      ;
      uint64_t v476 = (uint64_t) v44;
      TASSIGN(v475, v476);
      pipe_barrier(PIPE_V);
      TMULS(v475, v471, v84);
      // pto: %ci2_bit_inline380__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v477 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %ci2_bit_inline380__ssa_v0
      ;
      uint64_t v478 = (uint64_t) v48;
      TASSIGN(v477, v478);
      pipe_barrier(PIPE_V);
      TSUB(v477, v461, v475);
      // pto: %t__tmp_v136
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v479 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v136
      ;
      uint64_t v480 = (uint64_t) v48;
      TASSIGN(v479, v480);
      pipe_barrier(PIPE_V);
      TMULS(v479, v477, v90);
      // pto: %t__tmp_v137
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v481 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v137
      ;
      uint64_t v482 = (uint64_t) v45;
      TASSIGN(v481, v482);
      pipe_barrier(PIPE_V);
      TSUB(v481, v349, v479);
      // pto: %t__tmp_v138
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v483 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v138
      ;
      uint64_t v484 = (uint64_t) v45;
      TASSIGN(v483, v484);
      pipe_barrier(PIPE_V);
      TADDS(v483, v481, v88);
      // pto: %ci2_inline585__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v485 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %ci2_inline585__ssa_v0
      ;
      uint64_t v486 = (uint64_t) v45;
      TASSIGN(v485, v486);
      pipe_barrier(PIPE_V);
      RoundMode v487 = RoundMode::CAST_ROUND;
      SaturationMode v488 = SaturationMode::ON;
      TCVT(v485, v483, v487, v488);
      // pto: %t__tmp_v139
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v489 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v139
      ;
      uint64_t v490 = (uint64_t) v48;
      TASSIGN(v489, v490);
      TGATHER(v489, v339, v383, v347);
      // pto: %max2_inline442__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v491 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %max2_inline442__ssa_v1
      ;
      uint64_t v492 = (uint64_t) v48;
      TASSIGN(v491, v492);
      pipe_barrier(PIPE_V);
      TMAX(v491, v339, v489);
      // pto: %t__tmp_v140
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v493 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v140
      ;
      uint64_t v494 = (uint64_t) v44;
      TASSIGN(v493, v494);
      pipe_barrier(PIPE_V);
      TGATHER(v493, v491, v417, v347);
      // pto: %max4_inline504__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v495 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %max4_inline504__ssa_v1
      ;
      uint64_t v496 = (uint64_t) v48;
      TASSIGN(v495, v496);
      pipe_barrier(PIPE_V);
      TMAX(v495, v491, v493);
      // pto: %t__tmp_v141
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v497 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v141
      ;
      uint64_t v498 = (uint64_t) v48;
      TASSIGN(v497, v498);
      pipe_barrier(PIPE_V);
      TSUB(v497, v339, v495);
      // pto: %exps_inline614__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v499 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %exps_inline614__ssa_v1
      ;
      uint64_t v500 = (uint64_t) v48;
      TASSIGN(v499, v500);
      pipe_barrier(PIPE_V);
      TEXP(v499, v497);
      // pto: %t__tmp_v142
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v501 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v142
      ;
      uint64_t v502 = (uint64_t) v43;
      TASSIGN(v501, v502);
      pipe_barrier(PIPE_V);
      TGATHER(v501, v499, v383, v347);
      // pto: %sum2_inline549__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v503 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %sum2_inline549__ssa_v1
      ;
      uint64_t v504 = (uint64_t) v43;
      TASSIGN(v503, v504);
      pipe_barrier(PIPE_V);
      TADD(v503, v499, v501);
      // pto: %t__tmp_v143
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v505 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v143
      ;
      uint64_t v506 = (uint64_t) v49;
      TASSIGN(v505, v506);
      pipe_barrier(PIPE_V);
      TGATHER(v505, v503, v417, v347);
      // pto: %sum4_inline437__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v507 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %sum4_inline437__ssa_v1
      ;
      uint64_t v508 = (uint64_t) v43;
      TASSIGN(v507, v508);
      pipe_barrier(PIPE_V);
      TADD(v507, v503, v505);
      // pto: %t__tmp_v144
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v509 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v144
      ;
      uint64_t v510 = (uint64_t) v48;
      TASSIGN(v509, v510);
      pipe_barrier(PIPE_V);
      TDIV(v509, v499, v507);
      // pto: %probs_inline552__ssa_v1
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v511 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %probs_inline552__ssa_v1
      ;
      uint64_t v512 = (uint64_t) v48;
      TASSIGN(v511, v512);
      pipe_barrier(PIPE_V);
      TADDS(v511, v509, v80);
      // pto: %t__tmp_v145
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v513 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v145
      ;
      uint64_t v514 = (uint64_t) v43;
      TASSIGN(v513, v514);
      pipe_barrier(PIPE_V);
      TGATHER(v513, v511, v451, v347);
      // pto: %col2_inline414__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v515 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %col2_inline414__ssa_v4
      ;
      uint64_t v516 = (uint64_t) v43;
      TASSIGN(v515, v516);
      pipe_barrier(PIPE_V);
      TADD(v515, v511, v513);
      // pto: %t__tmp_v146
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v517 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v146
      ;
      uint64_t v518 = (uint64_t) v49;
      TASSIGN(v517, v518);
      pipe_barrier(PIPE_V);
      TGATHER(v517, v515, v485, v347);
      // pto: %t__tmp_v147
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v519 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v147
      ;
      uint64_t v520 = (uint64_t) v43;
      TASSIGN(v519, v520);
      pipe_barrier(PIPE_V);
      TADD(v519, v515, v517);
      // pto: %col4_inline553__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v521 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %col4_inline553__ssa_v4
      ;
      uint64_t v522 = (uint64_t) v43;
      TASSIGN(v521, v522);
      pipe_barrier(PIPE_V);
      TADDS(v521, v519, v80);
      // pto: %cur_inline555__ssa_v4
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v523 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %cur_inline555__ssa_v4
      ;
      uint64_t v524 = (uint64_t) v48;
      TASSIGN(v523, v524);
      pipe_barrier(PIPE_V);
      TDIV(v523, v511, v521);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      for (int64_t v525 = v73; v525 < v91; v525 += v64) {
        // pto: %t__tmp_v148
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v526 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
        // pto: %t__tmp_v148
        ;
        uint64_t v527 = (uint64_t) v43;
        TASSIGN(v526, v527);
        pipe_barrier(PIPE_V);
        TGATHER(v526, v523, v383, v347);
        // pto: %row2_inline558__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v528 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
        // pto: %row2_inline558__ssa_v1
        ;
        uint64_t v529 = (uint64_t) v43;
        TASSIGN(v528, v529);
        pipe_barrier(PIPE_V);
        TADD(v528, v523, v526);
        // pto: %t__tmp_v149
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v530 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
        // pto: %t__tmp_v149
        ;
        uint64_t v531 = (uint64_t) v49;
        TASSIGN(v530, v531);
        pipe_barrier(PIPE_V);
        TGATHER(v530, v528, v417, v347);
        // pto: %t__tmp_v150
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v532 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
        // pto: %t__tmp_v150
        ;
        uint64_t v533 = (uint64_t) v43;
        TASSIGN(v532, v533);
        pipe_barrier(PIPE_V);
        TADD(v532, v528, v530);
        // pto: %row4_inline561__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v534 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
        // pto: %row4_inline561__ssa_v1
        ;
        uint64_t v535 = (uint64_t) v43;
        TASSIGN(v534, v535);
        pipe_barrier(PIPE_V);
        TADDS(v534, v532, v80);
        // pto: %norm_inline562__ssa_v1
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v536 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
        // pto: %norm_inline562__ssa_v1
        ;
        uint64_t v537 = (uint64_t) v43;
        TASSIGN(v536, v537);
        pipe_barrier(PIPE_V);
        TDIV(v536, v523, v534);
        // pto: %t__tmp_v151
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v538 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
        // pto: %t__tmp_v151
        ;
        uint64_t v539 = (uint64_t) v49;
        TASSIGN(v538, v539);
        pipe_barrier(PIPE_V);
        TGATHER(v538, v536, v451, v347);
        // pto: %col2_inline414__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v540 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
        // pto: %col2_inline414__ssa_v7
        ;
        uint64_t v541 = (uint64_t) v49;
        TASSIGN(v540, v541);
        pipe_barrier(PIPE_V);
        TADD(v540, v536, v538);
        // pto: %t__tmp_v152
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v542 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
        // pto: %t__tmp_v152
        ;
        uint64_t v543 = (uint64_t) v38;
        TASSIGN(v542, v543);
        pipe_barrier(PIPE_V);
        TGATHER(v542, v540, v485, v347);
        // pto: %t__tmp_v153
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v544 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
        // pto: %t__tmp_v153
        ;
        uint64_t v545 = (uint64_t) v49;
        TASSIGN(v544, v545);
        pipe_barrier(PIPE_V);
        TADD(v544, v540, v542);
        // pto: %col4_inline553__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v546 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
        // pto: %col4_inline553__ssa_v7
        ;
        uint64_t v547 = (uint64_t) v49;
        TASSIGN(v546, v547);
        pipe_barrier(PIPE_V);
        TADDS(v546, v544, v80);
        // pto: %cur_inline555__ssa_v7
        ;
        Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v548 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
        // pto: %cur_inline555__ssa_v7
        ;
        uint64_t v549 = (uint64_t) v48;
        TASSIGN(v548, v549);
        pipe_barrier(PIPE_V);
        TDIV(v548, v536, v546);
      };
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      // pto: %t__tmp_v154
      ;
      Tile<TileType::Vec, float, 2, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v550 = Tile<TileType::Vec, float, 2, 16, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v75, v68);
      // pto: %t__tmp_v154
      ;
      uint64_t v551 = (uint64_t) v48;
      TASSIGN(v550, v551);
      v550.SetValidShape(v264, v68);
      // pto: %comb_inline515__ssa_v0_pview
      ;
      const int64_t v552 = 0;
      // pto: %comb_inline515__ssa_v0_pview
      ;
      __gm__ float* v553 = PTOAS__GLOBAL_TENSOR_DATA(v148);
      // pto: %comb_inline515__ssa_v0_pview
      ;
      const int64_t v554 = 1;
      // pto: %comb_inline515__ssa_v0_pview
      ;
      const int64_t v555 = 1;
      // pto: %comb_inline515__ssa_v0_pview
      ;
      const int64_t v556 = 1;
      // pto: %comb_inline515__ssa_v0_pview
      ;
      int64_t v557 = v264 * v68;
      // pto: %comb_inline515__ssa_v0_pview
      ;
      int64_t v558 = v556 * v557;
      // pto: %comb_inline515__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, -1, 16> v559 = pto::Shape<1, 1, 1, -1, 16>(v554, v555, v556, v264, v68);
      // pto: %comb_inline515__ssa_v0_pview
      ;
      pto::Stride<-1, -1, -1, -1, -1> v560 = pto::Stride<-1, -1, -1, -1, -1>(v555 * v558, v558, v557, v68, v64);
      // pto: %comb_inline515__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v561 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, 16>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v553 + (v552 + v288 * v68 + v73 * v64), v559, v560);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
      TSTORE(v561, v550);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    };
  } else {
    // pto: %80
    ;
    int64_t v562 = (int64_t) ((uint64_t) v254 - (uint64_t) v18);
    // pto: %82
    ;
    int64_t v563 = v562 < v22 ? v562 : v22;
    // pto: %81, %83, %84
    ;
    int64_t v564 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v562 * (uint64_t) v21) + (uint64_t) v563) * (uint64_t) v75);
    // pto: %85
    ;
    int64_t v565 = (int64_t) ((uint64_t) v22 - (uint64_t) v562);
    // pto: %86
    ;
    int64_t v566 = v565 < v73 ? v73 : v565;
    // pto: %87
    ;
    int64_t v567 = v566 < v64 ? v566 : v64;
    // pto: %90, %88, %89
    ;
    int64_t v568 = (int64_t) ((uint64_t) v564 + (uint64_t) ((int64_t) (uint64_t) ((int64_t) (uint64_t) v21 + (uint64_t) v567) * (uint64_t) v75));
    // pto: %91
    ;
    // pto: %92
    ;
    // pto: %d_begin_inline608__phi_v2
    ;
    int64_t v569;
    // pto: %d_end_inline610__phi_v2
    ;
    int64_t v570;
    // pto: %work_begin_inline453__phi_v3
    ;
    int64_t v571;
    // pto: %work_end_inline534__phi_v3
    ;
    int64_t v572;
    if (v24 < v25) {
      // pto: %93
      ;
      // pto: %d_worker_inline497__phi_v2
      ;
      int64_t v573;
      // pto: %d_workers_inline613__phi_v2
      ;
      int64_t v574;
      // pto: %pair_id_inline612__phi_v2
      ;
      int64_t v575;
      if (v562 < v26) {
        // pto: %94
        ;
        int64_t v576 = (int64_t) ((uint64_t) v27 + (uint64_t) v64);
        // pto: %95
        ;
        // pto: %97
        ;
        v573 = v562 % v576;
        v574 = v576;
        v575 = v562 / v576;
      } else {
        // pto: %99
        ;
        int64_t v577 = (int64_t) ((uint64_t) v562 - (uint64_t) v26);
        // pto: %101, %100
        ;
        // pto: %102
        ;
        v573 = v577 % v27;
        v574 = v27;
        v575 = (int64_t) ((uint64_t) v28 + (uint64_t) (v577 / v27));
      };
      // pto: %103
      ;
      uint64_t v578 = (uint64_t) v575;
      // pto: %103
      ;
      int64_t v579 = (int64_t) (v578 * (uint64_t) v75);
      // pto: %104
      ;
      int64_t v580 = v23 < v579 ? v23 : v579;
      // pto: %105
      ;
      int64_t v581 = (int64_t) ((uint64_t) v580 + (uint64_t) v75);
      // pto: %106
      ;
      // pto: %107
      ;
      int64_t v582 = v78 / v574;
      // pto: %108
      ;
      int64_t v583 = v78 % v574;
      // pto: %109
      ;
      uint64_t v584 = (uint64_t) v573;
      // pto: %110
      ;
      bool v585 = v573 < v583;
      // pto: %110
      ;
      int64_t v586 = v585 ? v573 : v583;
      // pto: %109, %111
      ;
      int64_t v587 = (int64_t) ((uint64_t) ((int64_t) v584 * (uint64_t) v582) + (uint64_t) v586);
      // pto: %113
      ;
      uint64_t v588 = (uint64_t) v573;
      // pto: %113
      ;
      int64_t v589 = (int64_t) ((uint64_t) v583 - v588);
      // pto: %114
      ;
      int64_t v590 = v589 < v73 ? v73 : v589;
      // pto: %115
      ;
      int64_t v591 = v590 < v64 ? v590 : v64;
      // pto: %112, %116
      ;
      v569 = v587;
      v570 = (int64_t) ((uint64_t) ((int64_t) (uint64_t) v587 + (uint64_t) v582) + (uint64_t) v591);
      v571 = v580;
      v572 = v23 < v581 ? v23 : v581;
    } else {
      v569 = v73;
      v570 = v78;
      v571 = v564;
      v572 = v23 < v568 ? v23 : v568;
    };
    // pto: %117
    ;
    bool v592 = v25 <= v562;
    // pto: %work_begin_inline453__phi_v5
    ;
    int64_t v593 = v592 ? v73 : v571;
    // pto: %work_end_inline534__phi_v5
    ;
    int64_t v594 = v592 ? v73 : v572;
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
    for (int64_t v595 = v593; v595 < v594; v595 += v75) {
      // pto: %118
      ;
      int64_t v596 = (int64_t) ((uint64_t) v594 - (uint64_t) v595);
      // pto: %119
      ;
      int64_t v597 = v596 < v75 ? v596 : v75;
      // pto: %120
      ;
      int64_t v598 = (int64_t) ((uint64_t) v597 * (uint64_t) v63);
      // pto: %m_parts_load_inline511__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v599 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v598);
      // pto: %m_parts_load_inline511__ssa_v0
      ;
      uint64_t v600 = (uint64_t) v35;
      TASSIGN(v599, v600);
      // pto: %121
      ;
      int64_t v601 = (int64_t) ((uint64_t) v595 * (uint64_t) v63);
      // pto: %122
      ;
      int64_t v602 = v601 < v73 ? v73 : v601;
      // pto: %124
      ;
      const int64_t v603 = 0;
      // pto: %124
      ;
      __gm__ float* v604 = PTOAS__GLOBAL_TENSOR_DATA(v122);
      // pto: %124
      ;
      const int64_t v605 = 1;
      // pto: %124
      ;
      const int64_t v606 = 1;
      // pto: %124
      ;
      const int64_t v607 = 1;
      // pto: %124
      ;
      int64_t v608 = v68 * v114;
      // pto: %124
      ;
      int64_t v609 = v607 * v608;
      // pto: %124
      ;
      pto::Shape<1, 1, 1, 16, -1> v610 = pto::Shape<1, 1, 1, 16, -1>(v605, v606, v607, v68, v598);
      // pto: %124
      ;
      pto::Stride<-1, -1, -1, -1, -1> v611 = pto::Stride<-1, -1, -1, -1, -1>(v606 * v609, v609, v608, v114, v64);
      // pto: %124
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v612 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v604 + (v603 + v73 * v114 + v602 * v64), v610, v611);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      TLOAD(v599, v612);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      // pto: %t__tmp_v155
      ;
      Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v613 = Tile<TileType::Vec, float, 16, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v68, v77);
      // pto: %t__tmp_v155
      ;
      uint64_t v614 = (uint64_t) v35;
      TASSIGN(v613, v614);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      TFILLPAD<pto::TFillPadMode::InPlace>(v613, v599);
      v613.SetValidShape(v68, v77);
      // pto: %t__tmp_v156
      ;
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v615 = Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v77);
      // pto: %t__tmp_v156
      ;
      uint64_t v616 = (uint64_t) v35;
      TASSIGN(v615, v616);
      pipe_barrier(PIPE_V);
      TCOLSUM(v615, v613);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      // pto: %m_total_inline618__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v617 = Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v63);
      // pto: %m_total_inline618__ssa_v0
      ;
      uint64_t v618 = (uint64_t) v35;
      TASSIGN(v617, v618);
      // pto: %m_sq_load_inline398__ssa_v0
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v619 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v68, v597);
      // pto: %m_sq_load_inline398__ssa_v0
      ;
      uint64_t v620 = (uint64_t) v38;
      TASSIGN(v619, v620);
      // pto: %125
      ;
      int64_t v621 = v595 < v73 ? v73 : v595;
      // pto: %126
      ;
      int64_t v622 = (int64_t) v29;
      // pto: %126
      ;
      const int64_t v623 = 0;
      // pto: %126
      ;
      __gm__ float* v624 = PTOAS__GLOBAL_TENSOR_DATA(v131);
      // pto: %126
      ;
      const int64_t v625 = 1;
      // pto: %126
      ;
      const int64_t v626 = 1;
      // pto: %126
      ;
      const int64_t v627 = 1;
      // pto: %126
      ;
      int64_t v628 = v68 * v622;
      // pto: %126
      ;
      int64_t v629 = v627 * v628;
      // pto: %126
      ;
      pto::Shape<1, 1, 1, 16, -1> v630 = pto::Shape<1, 1, 1, 16, -1>(v625, v626, v627, v68, v597);
      // pto: %126
      ;
      pto::Stride<-1, -1, -1, -1, -1> v631 = pto::Stride<-1, -1, -1, -1, -1>(v626 * v629, v629, v628, v622, v64);
      // pto: %126
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v632 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v624 + (v623 + v73 * v622 + v621 * v64), v630, v631);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      TLOAD(v619, v632);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      // pto: %t__tmp_v157
      ;
      Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null> v633 = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero, CompactMode::Null>(v68, v78);
      // pto: %t__tmp_v157
      ;
      uint64_t v634 = (uint64_t) v38;
      TASSIGN(v633, v634);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      TFILLPAD<pto::TFillPadMode::InPlace>(v633, v619);
      v633.SetValidShape(v68, v78);
      // pto: %m_rms_tmp_inline457__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v635 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v78);
      // pto: %m_rms_tmp_inline457__ssa_v0
      ;
      uint64_t v636 = (uint64_t) v39;
      TASSIGN(v635, v636);
      // pto: %t__tmp_v158
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v637 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v78);
      // pto: %t__tmp_v158
      ;
      uint64_t v638 = (uint64_t) v38;
      TASSIGN(v637, v638);
      pipe_barrier(PIPE_V);
      TCOLSUM(v637, v633);
      // pto: %t__tmp_v159
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v639 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v78);
      // pto: %t__tmp_v159
      ;
      uint64_t v640 = (uint64_t) v38;
      TASSIGN(v639, v640);
      pipe_barrier(PIPE_V);
      TMULS(v639, v637, v79);
      // pto: %t__tmp_v160
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v641 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v78);
      // pto: %t__tmp_v160
      ;
      uint64_t v642 = (uint64_t) v38;
      TASSIGN(v641, v642);
      pipe_barrier(PIPE_V);
      TADDS(v641, v639, v80);
      // pto: %t__tmp_v161
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v643 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v78);
      // pto: %t__tmp_v161
      ;
      uint64_t v644 = (uint64_t) v50;
      TASSIGN(v643, v644);
      pipe_barrier(PIPE_V);
      TRSQRT(v643, v641, v635);
      // pto: %m_inv_inline619__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v645 = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v64);
      // pto: %m_inv_inline619__ssa_v0
      ;
      uint64_t v646 = (uint64_t) v50;
      TASSIGN(v645, v646);
      // pto: %gate_base_inline484__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v647 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v78);
      // pto: %gate_base_inline484__ssa_v0
      ;
      uint64_t v648 = (uint64_t) v51;
      TASSIGN(v647, v648);
      // pto: %127
      ;
      __gm__ float* v649 = PTOAS__GLOBAL_TENSOR_DATA(v139);
      // pto: %127
      ;
      pto::Shape<1, 1, 1, 1, 8> v650 = pto::Shape<1, 1, 1, 1, 8>();
      // pto: %127
      ;
      pto::Stride<24, 24, 24, 24, 1> v651 = pto::Stride<24, 24, 24, 24, 1>();
      // pto: %127
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND> v652 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<24, 24, 24, 24, 1>, pto::Layout::ND>(v649, v650, v651);
      TLOAD(v647, v652);
      // pto: %gate_scales_inline620__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v653 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v67);
      // pto: %gate_scales_inline620__ssa_v0
      ;
      uint64_t v654 = (uint64_t) v52;
      TASSIGN(v653, v654);
      // pto: %hc_scale_2d_inline482__ssa_v0_pview
      ;
      __gm__ float* v655 = PTOAS__GLOBAL_TENSOR_DATA(v156);
      // pto: %hc_scale_2d_inline482__ssa_v0_pview
      ;
      pto::Shape<1, 1, 1, 1, 3> v656 = pto::Shape<1, 1, 1, 1, 3>();
      // pto: %hc_scale_2d_inline482__ssa_v0_pview
      ;
      pto::Stride<3, 3, 3, 3, 1> v657 = pto::Stride<3, 3, 3, 3, 1>();
      // pto: %hc_scale_2d_inline482__ssa_v0_pview
      ;
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND> v658 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 3>, pto::Stride<3, 3, 3, 3, 1>, pto::Layout::ND>(v655, v656, v657);
      TLOAD(v653, v658);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      // pto: %t__ci_tmp_v1
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v659 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__ci_tmp_v1
      ;
      uint64_t v660 = (uint64_t) v38;
      TASSIGN(v659, v660);
      // pto: %t__tmp_v162
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v661 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v78);
      // pto: %t__tmp_v162
      ;
      uint64_t v662 = (uint64_t) v53;
      TASSIGN(v661, v662);
      // pto: %128
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v663;
      TRESHAPE(v663, v659);
      // pto: %129
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v664;
      TRESHAPE(v664, v661);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v664, v70, v663);
      // pto: %gate_ids_inline621__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v665 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v78);
      // pto: %gate_ids_inline621__ssa_v0
      ;
      uint64_t v666 = (uint64_t) v38;
      TASSIGN(v665, v666);
      pipe_barrier(PIPE_V);
      RoundMode v667 = RoundMode::CAST_ROUND;
      SaturationMode v668 = SaturationMode::ON;
      TCVT(v665, v661, v667, v668);
      // pto: %t__tmp_v163
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v669 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v78);
      // pto: %t__tmp_v163
      ;
      uint64_t v670 = (uint64_t) v38;
      TASSIGN(v669, v670);
      pipe_barrier(PIPE_V);
      TMULS(v669, v665, v85);
      // pto: %gate_scale_ids_inline483__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v671 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v78);
      // pto: %gate_scale_ids_inline483__ssa_v0
      ;
      uint64_t v672 = (uint64_t) v38;
      TASSIGN(v671, v672);
      pipe_barrier(PIPE_V);
      RoundMode v673 = RoundMode::CAST_FLOOR;
      SaturationMode v674 = SaturationMode::ON;
      TCVT(v671, v669, v673, v674);
      // pto: %gate_tmp_inline458__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v675 = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v78);
      // pto: %gate_tmp_inline458__ssa_v0
      ;
      uint64_t v676 = (uint64_t) v54;
      TASSIGN(v675, v676);
      // pto: %gate_scale_vec_inline485__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v677 = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v78);
      // pto: %gate_scale_vec_inline485__ssa_v0
      ;
      uint64_t v678 = (uint64_t) v40;
      TASSIGN(v677, v678);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
      TGATHER(v677, v653, v671, v675);
      // pto: %t__tmp_v164
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v679 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %t__tmp_v164
      ;
      uint64_t v680 = (uint64_t) v35;
      TASSIGN(v679, v680);
      // pto: %130
      ;
      Tile<TileType::Vec, float, 8, 32, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v681;
      // pto: %130
      ;
      uint64_t v682 = (uint64_t) v35;
      TASSIGN(v681, v682);
      // pto: %gate_raw_inline464__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v683 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %gate_raw_inline464__ssa_v0
      ;
      uint64_t v684 = (uint64_t) v35;
      TASSIGN(v683, v684);
      TADDS(v683, v681, v74);
      // pto: %gate_norm_inline607__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v685 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %gate_norm_inline607__ssa_v0
      ;
      uint64_t v686 = (uint64_t) v35;
      TASSIGN(v685, v686);
      pipe_barrier(PIPE_V);
      TROWEXPANDMUL(v685, v683, v645);
      // pto: %gate_scaled_inline623__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v687 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %gate_scaled_inline623__ssa_v0
      ;
      uint64_t v688 = (uint64_t) v35;
      TASSIGN(v687, v688);
      pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(v687, v685, v677);
      // pto: %t__tmp_v165
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v689 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %t__tmp_v165
      ;
      uint64_t v690 = (uint64_t) v46;
      TASSIGN(v689, v690);
      TCOLEXPAND(v689, v647);
      // pto: %gate_logits_inline431__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v691 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %gate_logits_inline431__ssa_v0
      ;
      uint64_t v692 = (uint64_t) v35;
      TASSIGN(v691, v692);
      pipe_barrier(PIPE_V);
      TADD(v691, v687, v689);
      // pto: %t__tmp_v166
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v693 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %t__tmp_v166
      ;
      uint64_t v694 = (uint64_t) v35;
      TASSIGN(v693, v694);
      pipe_barrier(PIPE_V);
      TNEG(v693, v691);
      // pto: %t__tmp_v167
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v695 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %t__tmp_v167
      ;
      uint64_t v696 = (uint64_t) v35;
      TASSIGN(v695, v696);
      pipe_barrier(PIPE_V);
      TEXP(v695, v693);
      // pto: %t__tmp_v168
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v697 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %t__tmp_v168
      ;
      uint64_t v698 = (uint64_t) v35;
      TASSIGN(v697, v698);
      pipe_barrier(PIPE_V);
      TADDS(v697, v695, v82);
      // pto: %gate_sigmoid_inline427__ssa_v0
      ;
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v699 = Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v78, v78);
      // pto: %gate_sigmoid_inline427__ssa_v0
      ;
      uint64_t v700 = (uint64_t) v55;
      TASSIGN(v699, v700);
      pipe_barrier(PIPE_V);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
      TRECIP(v699, v697);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      // pto: %t__ci_tmp_v2
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v701 = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v81);
      // pto: %t__ci_tmp_v2
      ;
      uint64_t v702 = (uint64_t) v35;
      TASSIGN(v701, v702);
      // pto: %t__tmp_v169
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v703 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v169
      ;
      uint64_t v704 = (uint64_t) v43;
      TASSIGN(v703, v704);
      // pto: %131
      ;
      Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v705;
      TRESHAPE(v705, v701);
      // pto: %132
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v706;
      TRESHAPE(v706, v703);
      pipe_barrier(PIPE_V);
      TCI<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, 1, 192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, int32_t, 0>(v706, v70, v705);
      // pto: %post_ids_float_inline478__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v707 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %post_ids_float_inline478__ssa_v0
      ;
      uint64_t v708 = (uint64_t) v35;
      TASSIGN(v707, v708);
      pipe_barrier(PIPE_V);
      RoundMode v709 = RoundMode::CAST_ROUND;
      SaturationMode v710 = SaturationMode::ON;
      TCVT(v707, v703, v709, v710);
      // pto: %t__tmp_v170
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v711 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v170
      ;
      uint64_t v712 = (uint64_t) v45;
      TASSIGN(v711, v712);
      pipe_barrier(PIPE_V);
      TMULS(v711, v707, v85);
      // pto: %t__tmp_v171
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v713 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v171
      ;
      uint64_t v714 = (uint64_t) v45;
      TASSIGN(v713, v714);
      pipe_barrier(PIPE_V);
      RoundMode v715 = RoundMode::CAST_FLOOR;
      SaturationMode v716 = SaturationMode::ON;
      TCVT(v713, v711, v715, v716);
      // pto: %post_rows_inline625__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v717 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %post_rows_inline625__ssa_v0
      ;
      uint64_t v718 = (uint64_t) v45;
      TASSIGN(v717, v718);
      pipe_barrier(PIPE_V);
      RoundMode v719 = RoundMode::CAST_ROUND;
      SaturationMode v720 = SaturationMode::ON;
      TCVT(v717, v713, v719, v720);
      // pto: %t__tmp_v172
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v721 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v172
      ;
      uint64_t v722 = (uint64_t) v45;
      TASSIGN(v721, v722);
      pipe_barrier(PIPE_V);
      TMULS(v721, v717, v86);
      // pto: %t__tmp_v173
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v723 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v173
      ;
      uint64_t v724 = (uint64_t) v35;
      TASSIGN(v723, v724);
      pipe_barrier(PIPE_V);
      TADD(v723, v707, v721);
      // pto: %t__tmp_v174
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v725 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v174
      ;
      uint64_t v726 = (uint64_t) v35;
      TASSIGN(v725, v726);
      pipe_barrier(PIPE_V);
      TADDS(v725, v723, v86);
      // pto: %post_ids_inline514__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v727 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %post_ids_inline514__ssa_v0
      ;
      uint64_t v728 = (uint64_t) v35;
      TASSIGN(v727, v728);
      pipe_barrier(PIPE_V);
      RoundMode v729 = RoundMode::CAST_ROUND;
      SaturationMode v730 = SaturationMode::ON;
      TCVT(v727, v725, v729, v730);
      // pto: %post_tmp_inline628__ssa_v0
      ;
      Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v731 = Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %post_tmp_inline628__ssa_v0
      ;
      uint64_t v732 = (uint64_t) v45;
      TASSIGN(v731, v732);
      // pto: %t__tmp_v175
      ;
      Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v733 = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v92);
      // pto: %t__tmp_v175
      ;
      uint64_t v734 = (uint64_t) v55;
      TASSIGN(v733, v734);
      // pto: %post_values_inline629__ssa_v0
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v735 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %post_values_inline629__ssa_v0
      ;
      uint64_t v736 = (uint64_t) v46;
      TASSIGN(v735, v736);
      pipe_barrier(PIPE_V);
      TGATHER(v735, v733, v727, v731);
      // pto: %t__tmp_v176
      ;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v737 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v63);
      // pto: %t__tmp_v176
      ;
      uint64_t v738 = (uint64_t) v35;
      TASSIGN(v737, v738);
      pipe_barrier(PIPE_V);
      TMULS(v737, v735, v84);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      // pto: %133
      ;
      int64_t v739 = (int64_t) ((uint64_t) v597 * (uint64_t) v93);
      v737.SetValidShape(v64, v739);
      // pto: %134
      ;
      bool v740 = v569 == v73;
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      if (v740) {
        // pto: %136
        ;
        int64_t v741 = (int64_t) ((uint64_t) v595 * (uint64_t) v93);
        // pto: %137
        ;
        int64_t v742 = v741 < v73 ? v73 : v741;
        // pto: %post_flat_inline392__ssa_v0_pview
        ;
        const int64_t v743 = 0;
        // pto: %post_flat_inline392__ssa_v0_pview
        ;
        __gm__ float* v744 = PTOAS__GLOBAL_TENSOR_DATA(v165);
        // pto: %post_flat_inline392__ssa_v0_pview
        ;
        const int64_t v745 = 1;
        // pto: %post_flat_inline392__ssa_v0_pview
        ;
        const int64_t v746 = 1;
        // pto: %post_flat_inline392__ssa_v0_pview
        ;
        const int64_t v747 = 1;
        // pto: %post_flat_inline392__ssa_v0_pview
        ;
        int64_t v748 = v64 * v157;
        // pto: %post_flat_inline392__ssa_v0_pview
        ;
        int64_t v749 = v747 * v748;
        // pto: %post_flat_inline392__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, -1> v750 = pto::Shape<1, 1, 1, 1, -1>(v745, v746, v747, v64, v739);
        // pto: %post_flat_inline392__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v751 = pto::Stride<-1, -1, -1, -1, -1>(v746 * v749, v749, v748, v157, v64);
        // pto: %post_flat_inline392__ssa_v0_pview
        ;
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v752 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v744 + (v743 + v73 * v157 + v742 * v64), v750, v751);
        TSTORE(v752, v737);
      };
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
      // pto: %138
      ;
      uint64_t v753 = (uint64_t) v569;
      // pto: %138
      ;
      int64_t v754 = (int64_t) (v753 * (uint64_t) v76);
      // pto: %139
      ;
      uint64_t v755 = (uint64_t) v570;
      // pto: %139
      ;
      uint64_t v756 = (uint64_t) v569;
      // pto: %139, %140
      ;
      int64_t v757 = (int64_t) ((uint64_t) ((int64_t) v755 - v756) * (uint64_t) v76);
      // pto: %141
      ;
      int64_t v758 = v597 / v75;
      // pto: %142
      ;
      int64_t v759 = (int64_t) ((uint64_t) v758 * (uint64_t) v75);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      for (int64_t v760 = v73; v760 < v759; v760 += v75) {
        // pto: %143
        ;
        int64_t v761 = (int64_t) ((uint64_t) v595 + (uint64_t) v760);
        // pto: %144
        ;
        int64_t v762 = (int64_t) ((uint64_t) v760 + (uint64_t) v64);
        // pto: %145
        ;
        int64_t v763 = (int64_t) ((uint64_t) v595 + (uint64_t) v762);
        // pto: %flat_offset_mul
        ;
        int64_t v764 = (int64_t) ((uint64_t) v760 * (uint64_t) v78);
        // pto: %t__tmp_v177
        ;
        float v765 = v699.GetValue(v764);
        // pto: %146
        ;
        float v766 = v765 + v80;
        // pto: %148, %t__tmp_v178
        ;
        float v767 = v699.GetValue((int64_t) ((uint64_t) v764 + (uint64_t) v64));
        // pto: %149
        ;
        float v768 = v767 + v80;
        // pto: %151, %t__tmp_v179
        ;
        float v769 = v699.GetValue((int64_t) ((uint64_t) v764 + (uint64_t) v75));
        // pto: %152
        ;
        float v770 = v769 + v80;
        // pto: %154, %t__tmp_v180
        ;
        float v771 = v699.GetValue((int64_t) ((uint64_t) v764 + (uint64_t) v67));
        // pto: %155
        ;
        float v772 = v771 + v80;
        // pto: %t__tmp_v181
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v773 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %t__tmp_v181
        ;
        uint64_t v774 = (uint64_t) v56;
        TASSIGN(v773, v774);
        // pto: %156
        ;
        int64_t v775 = v761 < v73 ? v73 : v761;
        // pto: %157
        ;
        int64_t v776 = v754 < v73 ? v73 : v754;
        // pto: %158
        ;
        const int64_t v777 = 0;
        // pto: %158
        ;
        __gm__ bfloat16_t* v778 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %158
        ;
        const int64_t v779 = 1;
        // pto: %158
        ;
        const int64_t v780 = 1;
        // pto: %158
        ;
        const int64_t v781 = 1;
        // pto: %158
        ;
        int64_t v782 = v64 * v65;
        // pto: %158
        ;
        int64_t v783 = v781 * v782;
        // pto: %158
        ;
        pto::Shape<1, 1, 1, 1, -1> v784 = pto::Shape<1, 1, 1, 1, -1>(v779, v780, v781, v64, v757);
        // pto: %158
        ;
        pto::Stride<-1, -1, -1, -1, -1> v785 = pto::Stride<-1, -1, -1, -1, -1>(v780 * v783, v783, v782, v65, v64);
        // pto: %158
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v786 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v778 + (v777 + v775 * v65 + v776 * v64), v784, v785);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        TLOAD(v773, v786);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %t__tmp_v182
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v787 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %t__tmp_v182
        ;
        uint64_t v788 = (uint64_t) v57;
        TASSIGN(v787, v788);
        // pto: %160
        ;
        int64_t v789 = (int64_t) ((uint64_t) v754 + (uint64_t) v69);
        // pto: %161
        ;
        int64_t v790 = v789 < v73 ? v73 : v789;
        // pto: %162
        ;
        const int64_t v791 = 0;
        // pto: %162
        ;
        __gm__ bfloat16_t* v792 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %162
        ;
        const int64_t v793 = 1;
        // pto: %162
        ;
        const int64_t v794 = 1;
        // pto: %162
        ;
        const int64_t v795 = 1;
        // pto: %162
        ;
        int64_t v796 = v64 * v65;
        // pto: %162
        ;
        int64_t v797 = v795 * v796;
        // pto: %162
        ;
        pto::Shape<1, 1, 1, 1, -1> v798 = pto::Shape<1, 1, 1, 1, -1>(v793, v794, v795, v64, v757);
        // pto: %162
        ;
        pto::Stride<-1, -1, -1, -1, -1> v799 = pto::Stride<-1, -1, -1, -1, -1>(v794 * v797, v797, v796, v65, v64);
        // pto: %162
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v800 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v792 + (v791 + v775 * v65 + v790 * v64), v798, v799);
        TLOAD(v787, v800);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        // pto: %t__tmp_v183
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v801 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %t__tmp_v183
        ;
        uint64_t v802 = (uint64_t) v58;
        TASSIGN(v801, v802);
        // pto: %164
        ;
        int64_t v803 = (int64_t) ((uint64_t) v754 + (uint64_t) v94);
        // pto: %165
        ;
        int64_t v804 = v803 < v73 ? v73 : v803;
        // pto: %166
        ;
        const int64_t v805 = 0;
        // pto: %166
        ;
        __gm__ bfloat16_t* v806 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %166
        ;
        const int64_t v807 = 1;
        // pto: %166
        ;
        const int64_t v808 = 1;
        // pto: %166
        ;
        const int64_t v809 = 1;
        // pto: %166
        ;
        int64_t v810 = v64 * v65;
        // pto: %166
        ;
        int64_t v811 = v809 * v810;
        // pto: %166
        ;
        pto::Shape<1, 1, 1, 1, -1> v812 = pto::Shape<1, 1, 1, 1, -1>(v807, v808, v809, v64, v757);
        // pto: %166
        ;
        pto::Stride<-1, -1, -1, -1, -1> v813 = pto::Stride<-1, -1, -1, -1, -1>(v808 * v811, v811, v810, v65, v64);
        // pto: %166
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v814 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v806 + (v805 + v775 * v65 + v804 * v64), v812, v813);
        TLOAD(v801, v814);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        // pto: %t__tmp_v184
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v815 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %t__tmp_v184
        ;
        uint64_t v816 = (uint64_t) v34;
        TASSIGN(v815, v816);
        // pto: %168
        ;
        int64_t v817 = (int64_t) ((uint64_t) v754 + (uint64_t) v95);
        // pto: %169
        ;
        int64_t v818 = v817 < v73 ? v73 : v817;
        // pto: %170
        ;
        const int64_t v819 = 0;
        // pto: %170
        ;
        __gm__ bfloat16_t* v820 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %170
        ;
        const int64_t v821 = 1;
        // pto: %170
        ;
        const int64_t v822 = 1;
        // pto: %170
        ;
        const int64_t v823 = 1;
        // pto: %170
        ;
        int64_t v824 = v64 * v65;
        // pto: %170
        ;
        int64_t v825 = v823 * v824;
        // pto: %170
        ;
        pto::Shape<1, 1, 1, 1, -1> v826 = pto::Shape<1, 1, 1, 1, -1>(v821, v822, v823, v64, v757);
        // pto: %170
        ;
        pto::Stride<-1, -1, -1, -1, -1> v827 = pto::Stride<-1, -1, -1, -1, -1>(v822 * v825, v825, v824, v65, v64);
        // pto: %170
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v828 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v820 + (v819 + v775 * v65 + v818 * v64), v826, v827);
        TLOAD(v815, v828);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // pto: %173
        ;
        int64_t v829 = (int64_t) ((uint64_t) v762 * (uint64_t) v78);
        // pto: %171
        ;
        float v830 = v699.GetValue(v829);
        // pto: %175
        ;
        float v831 = v830 + v80;
        // pto: %179, %176
        ;
        float v832 = v699.GetValue((int64_t) ((uint64_t) v829 + (uint64_t) v64));
        // pto: %180
        ;
        float v833 = v832 + v80;
        // pto: %184, %181
        ;
        float v834 = v699.GetValue((int64_t) ((uint64_t) v829 + (uint64_t) v75));
        // pto: %185
        ;
        float v835 = v834 + v80;
        // pto: %189, %186
        ;
        float v836 = v699.GetValue((int64_t) ((uint64_t) v829 + (uint64_t) v67));
        // pto: %190
        ;
        float v837 = v836 + v80;
        // pto: %5
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v838 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %5
        ;
        uint64_t v839 = (uint64_t) v59;
        TASSIGN(v838, v839);
        // pto: %191
        ;
        int64_t v840 = v763 < v73 ? v73 : v763;
        // pto: %193
        ;
        const int64_t v841 = 0;
        // pto: %193
        ;
        __gm__ bfloat16_t* v842 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %193
        ;
        const int64_t v843 = 1;
        // pto: %193
        ;
        const int64_t v844 = 1;
        // pto: %193
        ;
        const int64_t v845 = 1;
        // pto: %193
        ;
        int64_t v846 = v64 * v65;
        // pto: %193
        ;
        int64_t v847 = v845 * v846;
        // pto: %193
        ;
        pto::Shape<1, 1, 1, 1, -1> v848 = pto::Shape<1, 1, 1, 1, -1>(v843, v844, v845, v64, v757);
        // pto: %193
        ;
        pto::Stride<-1, -1, -1, -1, -1> v849 = pto::Stride<-1, -1, -1, -1, -1>(v844 * v847, v847, v846, v65, v64);
        // pto: %193
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v850 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v842 + (v841 + v840 * v65 + v776 * v64), v848, v849);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v838, v850);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        // pto: %6
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v851 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %6
        ;
        uint64_t v852 = (uint64_t) v60;
        TASSIGN(v851, v852);
        // pto: %197
        ;
        const int64_t v853 = 0;
        // pto: %197
        ;
        __gm__ bfloat16_t* v854 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %197
        ;
        const int64_t v855 = 1;
        // pto: %197
        ;
        const int64_t v856 = 1;
        // pto: %197
        ;
        const int64_t v857 = 1;
        // pto: %197
        ;
        int64_t v858 = v64 * v65;
        // pto: %197
        ;
        int64_t v859 = v857 * v858;
        // pto: %197
        ;
        pto::Shape<1, 1, 1, 1, -1> v860 = pto::Shape<1, 1, 1, 1, -1>(v855, v856, v857, v64, v757);
        // pto: %197
        ;
        pto::Stride<-1, -1, -1, -1, -1> v861 = pto::Stride<-1, -1, -1, -1, -1>(v856 * v859, v859, v858, v65, v64);
        // pto: %197
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v862 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v854 + (v853 + v840 * v65 + v790 * v64), v860, v861);
        TLOAD(v851, v862);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        // pto: %7
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v863 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %7
        ;
        uint64_t v864 = (uint64_t) v61;
        TASSIGN(v863, v864);
        // pto: %201
        ;
        const int64_t v865 = 0;
        // pto: %201
        ;
        __gm__ bfloat16_t* v866 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %201
        ;
        const int64_t v867 = 1;
        // pto: %201
        ;
        const int64_t v868 = 1;
        // pto: %201
        ;
        const int64_t v869 = 1;
        // pto: %201
        ;
        int64_t v870 = v64 * v65;
        // pto: %201
        ;
        int64_t v871 = v869 * v870;
        // pto: %201
        ;
        pto::Shape<1, 1, 1, 1, -1> v872 = pto::Shape<1, 1, 1, 1, -1>(v867, v868, v869, v64, v757);
        // pto: %201
        ;
        pto::Stride<-1, -1, -1, -1, -1> v873 = pto::Stride<-1, -1, -1, -1, -1>(v868 * v871, v871, v870, v65, v64);
        // pto: %201
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v874 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v866 + (v865 + v840 * v65 + v804 * v64), v872, v873);
        TLOAD(v863, v874);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        // pto: %8
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v875 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %8
        ;
        uint64_t v876 = (uint64_t) v62;
        TASSIGN(v875, v876);
        // pto: %205
        ;
        const int64_t v877 = 0;
        // pto: %205
        ;
        __gm__ bfloat16_t* v878 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %205
        ;
        const int64_t v879 = 1;
        // pto: %205
        ;
        const int64_t v880 = 1;
        // pto: %205
        ;
        const int64_t v881 = 1;
        // pto: %205
        ;
        int64_t v882 = v64 * v65;
        // pto: %205
        ;
        int64_t v883 = v881 * v882;
        // pto: %205
        ;
        pto::Shape<1, 1, 1, 1, -1> v884 = pto::Shape<1, 1, 1, 1, -1>(v879, v880, v881, v64, v757);
        // pto: %205
        ;
        pto::Stride<-1, -1, -1, -1, -1> v885 = pto::Stride<-1, -1, -1, -1, -1>(v880 * v883, v883, v882, v65, v64);
        // pto: %205
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v886 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v878 + (v877 + v840 * v65 + v818 * v64), v884, v885);
        TLOAD(v875, v886);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        // pto: %part_scalar_x0_inline348__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v887 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %part_scalar_x0_inline348__ssa_v0
        ;
        uint64_t v888 = (uint64_t) v35;
        TASSIGN(v887, v888);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        RoundMode v889 = RoundMode::CAST_ROUND;
        SaturationMode v890 = SaturationMode::ON;
        TCVT(v887, v773, v889, v890);
        // pto: %part_scalar_y0_inline347__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v891 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %part_scalar_y0_inline347__ssa_v0
        ;
        uint64_t v892 = (uint64_t) v35;
        TASSIGN(v891, v892);
        pipe_barrier(PIPE_V);
        TMULS(v891, v887, v766);
        // pto: %part_scalar_x1_inline624__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v893 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %part_scalar_x1_inline624__ssa_v0
        ;
        uint64_t v894 = (uint64_t) v56;
        TASSIGN(v893, v894);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        RoundMode v895 = RoundMode::CAST_ROUND;
        SaturationMode v896 = SaturationMode::ON;
        TCVT(v893, v787, v895, v896);
        // pto: %part_scalar_y1_inline346__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v897 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %part_scalar_y1_inline346__ssa_v0
        ;
        uint64_t v898 = (uint64_t) v56;
        TASSIGN(v897, v898);
        pipe_barrier(PIPE_V);
        TMULS(v897, v893, v768);
        // pto: %part_scalar_sum01_inline597__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v899 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %part_scalar_sum01_inline597__ssa_v0
        ;
        uint64_t v900 = (uint64_t) v35;
        TASSIGN(v899, v900);
        pipe_barrier(PIPE_V);
        TADD(v899, v891, v897);
        // pto: %part_scalar_x2_inline345__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v901 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %part_scalar_x2_inline345__ssa_v0
        ;
        uint64_t v902 = (uint64_t) v56;
        TASSIGN(v901, v902);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        RoundMode v903 = RoundMode::CAST_ROUND;
        SaturationMode v904 = SaturationMode::ON;
        TCVT(v901, v801, v903, v904);
        // pto: %part_scalar_y2_inline349__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v905 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %part_scalar_y2_inline349__ssa_v0
        ;
        uint64_t v906 = (uint64_t) v56;
        TASSIGN(v905, v906);
        pipe_barrier(PIPE_V);
        TMULS(v905, v901, v770);
        // pto: %part_scalar_x3_inline344__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v907 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %part_scalar_x3_inline344__ssa_v0
        ;
        uint64_t v908 = (uint64_t) v57;
        TASSIGN(v907, v908);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        RoundMode v909 = RoundMode::CAST_ROUND;
        SaturationMode v910 = SaturationMode::ON;
        TCVT(v907, v815, v909, v910);
        // pto: %part_scalar_y3_inline343__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v911 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %part_scalar_y3_inline343__ssa_v0
        ;
        uint64_t v912 = (uint64_t) v57;
        TASSIGN(v911, v912);
        pipe_barrier(PIPE_V);
        TMULS(v911, v907, v772);
        // pto: %t__tmp_v185
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v913 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %t__tmp_v185
        ;
        uint64_t v914 = (uint64_t) v56;
        TASSIGN(v913, v914);
        pipe_barrier(PIPE_V);
        TADD(v913, v905, v911);
        // pto: %part_scalar_result_inline522__ssa_v0
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v915 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %part_scalar_result_inline522__ssa_v0
        ;
        uint64_t v916 = (uint64_t) v35;
        TASSIGN(v915, v916);
        pipe_barrier(PIPE_V);
        TADD(v915, v899, v913);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        // pto: %part_scalar_bf16_inline342__ssa_v0
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v917 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %part_scalar_bf16_inline342__ssa_v0
        ;
        uint64_t v918 = (uint64_t) v35;
        TASSIGN(v917, v918);
        pipe_barrier(PIPE_V);
        RoundMode v919 = RoundMode::CAST_RINT;
        SaturationMode v920 = SaturationMode::ON;
        TCVT(v917, v915, v919, v920);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        v917.SetValidShape(v64, v757);
        // pto: %mixed_view_inline454__ssa_v0_pview
        ;
        const int64_t v921 = 0;
        // pto: %mixed_view_inline454__ssa_v0_pview
        ;
        __gm__ bfloat16_t* v922 = PTOAS__GLOBAL_TENSOR_DATA(v174);
        // pto: %mixed_view_inline454__ssa_v0_pview
        ;
        const int64_t v923 = 1;
        // pto: %mixed_view_inline454__ssa_v0_pview
        ;
        const int64_t v924 = 1;
        // pto: %mixed_view_inline454__ssa_v0_pview
        ;
        const int64_t v925 = 1;
        // pto: %mixed_view_inline454__ssa_v0_pview
        ;
        int64_t v926 = v64 * v69;
        // pto: %mixed_view_inline454__ssa_v0_pview
        ;
        int64_t v927 = v925 * v926;
        // pto: %mixed_view_inline454__ssa_v0_pview
        ;
        pto::Shape<1, 1, 1, 1, -1> v928 = pto::Shape<1, 1, 1, 1, -1>(v923, v924, v925, v64, v757);
        // pto: %mixed_view_inline454__ssa_v0_pview
        ;
        pto::Stride<-1, -1, -1, -1, -1> v929 = pto::Stride<-1, -1, -1, -1, -1>(v924 * v927, v927, v926, v69, v64);
        // pto: %mixed_view_inline454__ssa_v0_pview
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v930 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v922 + (v921 + v775 * v69 + v776 * v64), v928, v929);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v930, v917);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        // pto: %9
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v931 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %9
        ;
        uint64_t v932 = (uint64_t) v36;
        TASSIGN(v931, v932);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        RoundMode v933 = RoundMode::CAST_ROUND;
        SaturationMode v934 = SaturationMode::ON;
        TCVT(v931, v838, v933, v934);
        // pto: %10
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v935 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %10
        ;
        uint64_t v936 = (uint64_t) v36;
        TASSIGN(v935, v936);
        pipe_barrier(PIPE_V);
        TMULS(v935, v931, v831);
        // pto: %11
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v937 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %11
        ;
        uint64_t v938 = (uint64_t) v59;
        TASSIGN(v937, v938);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        RoundMode v939 = RoundMode::CAST_ROUND;
        SaturationMode v940 = SaturationMode::ON;
        TCVT(v937, v851, v939, v940);
        // pto: %12
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v941 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %12
        ;
        uint64_t v942 = (uint64_t) v59;
        TASSIGN(v941, v942);
        pipe_barrier(PIPE_V);
        TMULS(v941, v937, v833);
        // pto: %13
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v943 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %13
        ;
        uint64_t v944 = (uint64_t) v36;
        TASSIGN(v943, v944);
        pipe_barrier(PIPE_V);
        TADD(v943, v935, v941);
        // pto: %14
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v945 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %14
        ;
        uint64_t v946 = (uint64_t) v59;
        TASSIGN(v945, v946);
        pipe_barrier(PIPE_V);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
        RoundMode v947 = RoundMode::CAST_ROUND;
        SaturationMode v948 = SaturationMode::ON;
        TCVT(v945, v863, v947, v948);
        // pto: %15
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v949 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %15
        ;
        uint64_t v950 = (uint64_t) v59;
        TASSIGN(v949, v950);
        pipe_barrier(PIPE_V);
        TMULS(v949, v945, v835);
        // pto: %16
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v951 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %16
        ;
        uint64_t v952 = (uint64_t) v60;
        TASSIGN(v951, v952);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID7);
        RoundMode v953 = RoundMode::CAST_ROUND;
        SaturationMode v954 = SaturationMode::ON;
        TCVT(v951, v875, v953, v954);
        // pto: %17
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v955 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %17
        ;
        uint64_t v956 = (uint64_t) v60;
        TASSIGN(v955, v956);
        pipe_barrier(PIPE_V);
        TMULS(v955, v951, v837);
        // pto: %18
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v957 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %18
        ;
        uint64_t v958 = (uint64_t) v59;
        TASSIGN(v957, v958);
        pipe_barrier(PIPE_V);
        TADD(v957, v949, v955);
        // pto: %19
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v959 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %19
        ;
        uint64_t v960 = (uint64_t) v36;
        TASSIGN(v959, v960);
        pipe_barrier(PIPE_V);
        TADD(v959, v943, v957);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %20
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v961 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %20
        ;
        uint64_t v962 = (uint64_t) v36;
        TASSIGN(v961, v962);
        pipe_barrier(PIPE_V);
        RoundMode v963 = RoundMode::CAST_RINT;
        SaturationMode v964 = SaturationMode::ON;
        TCVT(v961, v959, v963, v964);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        v961.SetValidShape(v64, v757);
        // pto: %211
        ;
        const int64_t v965 = 0;
        // pto: %211
        ;
        __gm__ bfloat16_t* v966 = PTOAS__GLOBAL_TENSOR_DATA(v174);
        // pto: %211
        ;
        const int64_t v967 = 1;
        // pto: %211
        ;
        const int64_t v968 = 1;
        // pto: %211
        ;
        const int64_t v969 = 1;
        // pto: %211
        ;
        int64_t v970 = v64 * v69;
        // pto: %211
        ;
        int64_t v971 = v969 * v970;
        // pto: %211
        ;
        pto::Shape<1, 1, 1, 1, -1> v972 = pto::Shape<1, 1, 1, 1, -1>(v967, v968, v969, v64, v757);
        // pto: %211
        ;
        pto::Stride<-1, -1, -1, -1, -1> v973 = pto::Stride<-1, -1, -1, -1, -1>(v968 * v971, v971, v970, v69, v64);
        // pto: %211
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v974 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v966 + (v965 + v840 * v69 + v776 * v64), v972, v973);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID7);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v974, v961);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      };
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_S, PIPE_V, EVENT_ID1);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
      // pto: %212, %213
      ;
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID7);
      if ((int64_t) ((uint64_t) v597 - (uint64_t) v759) == v64) {
        // pto: %214
        ;
        int64_t v975 = (int64_t) ((uint64_t) v595 + (uint64_t) v759);
        int64_t v976 = (int64_t) ((uint64_t) v758 * (uint64_t) v68);
        // pto: %215
        ;
        float v977 = v699.GetValue(v976);
        // pto: %218
        ;
        float v978 = v977 + v80;
        // pto: %221, %219
        ;
        float v979 = v699.GetValue((int64_t) ((uint64_t) v976 + (uint64_t) v64));
        // pto: %222
        ;
        float v980 = v979 + v80;
        // pto: %225, %223
        ;
        float v981 = v699.GetValue((int64_t) ((uint64_t) v976 + (uint64_t) v75));
        // pto: %226
        ;
        float v982 = v981 + v80;
        // pto: %229, %227
        ;
        float v983 = v699.GetValue((int64_t) ((uint64_t) v976 + (uint64_t) v67));
        // pto: %230
        ;
        float v984 = v983 + v80;
        // pto: %22
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v985 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %22
        ;
        uint64_t v986 = (uint64_t) v56;
        TASSIGN(v985, v986);
        // pto: %231
        ;
        int64_t v987 = v975 < v73 ? v73 : v975;
        // pto: %232
        ;
        int64_t v988 = v754 < v73 ? v73 : v754;
        // pto: %233
        ;
        const int64_t v989 = 0;
        // pto: %233
        ;
        __gm__ bfloat16_t* v990 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %233
        ;
        const int64_t v991 = 1;
        // pto: %233
        ;
        const int64_t v992 = 1;
        // pto: %233
        ;
        const int64_t v993 = 1;
        // pto: %233
        ;
        int64_t v994 = v64 * v65;
        // pto: %233
        ;
        int64_t v995 = v993 * v994;
        // pto: %233
        ;
        pto::Shape<1, 1, 1, 1, -1> v996 = pto::Shape<1, 1, 1, 1, -1>(v991, v992, v993, v64, v757);
        // pto: %233
        ;
        pto::Stride<-1, -1, -1, -1, -1> v997 = pto::Stride<-1, -1, -1, -1, -1>(v992 * v995, v995, v994, v65, v64);
        // pto: %233
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v998 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v990 + (v989 + v987 * v65 + v988 * v64), v996, v997);
        TLOAD(v985, v998);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %23
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v999 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %23
        ;
        uint64_t v1000 = (uint64_t) v35;
        TASSIGN(v999, v1000);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v1001 = RoundMode::CAST_ROUND;
        SaturationMode v1002 = SaturationMode::ON;
        TCVT(v999, v985, v1001, v1002);
        // pto: %24
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1003 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %24
        ;
        uint64_t v1004 = (uint64_t) v35;
        TASSIGN(v1003, v1004);
        pipe_barrier(PIPE_V);
        TMULS(v1003, v999, v978);
        // pto: %25
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1005 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %25
        ;
        uint64_t v1006 = (uint64_t) v57;
        TASSIGN(v1005, v1006);
        // pto: %235
        ;
        int64_t v1007 = (int64_t) ((uint64_t) v754 + (uint64_t) v69);
        // pto: %236
        ;
        int64_t v1008 = v1007 < v73 ? v73 : v1007;
        // pto: %237
        ;
        const int64_t v1009 = 0;
        // pto: %237
        ;
        __gm__ bfloat16_t* v1010 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %237
        ;
        const int64_t v1011 = 1;
        // pto: %237
        ;
        const int64_t v1012 = 1;
        // pto: %237
        ;
        const int64_t v1013 = 1;
        // pto: %237
        ;
        int64_t v1014 = v64 * v65;
        // pto: %237
        ;
        int64_t v1015 = v1013 * v1014;
        // pto: %237
        ;
        pto::Shape<1, 1, 1, 1, -1> v1016 = pto::Shape<1, 1, 1, 1, -1>(v1011, v1012, v1013, v64, v757);
        // pto: %237
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1017 = pto::Stride<-1, -1, -1, -1, -1>(v1012 * v1015, v1015, v1014, v65, v64);
        // pto: %237
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1018 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1010 + (v1009 + v987 * v65 + v1008 * v64), v1016, v1017);
        TLOAD(v1005, v1018);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %26
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1019 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %26
        ;
        uint64_t v1020 = (uint64_t) v56;
        TASSIGN(v1019, v1020);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v1021 = RoundMode::CAST_ROUND;
        SaturationMode v1022 = SaturationMode::ON;
        TCVT(v1019, v1005, v1021, v1022);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        // pto: %27
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1023 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %27
        ;
        uint64_t v1024 = (uint64_t) v56;
        TASSIGN(v1023, v1024);
        pipe_barrier(PIPE_V);
        TMULS(v1023, v1019, v980);
        // pto: %28
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1025 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %28
        ;
        uint64_t v1026 = (uint64_t) v35;
        TASSIGN(v1025, v1026);
        pipe_barrier(PIPE_V);
        TADD(v1025, v1003, v1023);
        // pto: %29
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1027 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %29
        ;
        uint64_t v1028 = (uint64_t) v57;
        TASSIGN(v1027, v1028);
        // pto: %239
        ;
        int64_t v1029 = (int64_t) ((uint64_t) v754 + (uint64_t) v94);
        // pto: %240
        ;
        int64_t v1030 = v1029 < v73 ? v73 : v1029;
        // pto: %241
        ;
        const int64_t v1031 = 0;
        // pto: %241
        ;
        __gm__ bfloat16_t* v1032 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %241
        ;
        const int64_t v1033 = 1;
        // pto: %241
        ;
        const int64_t v1034 = 1;
        // pto: %241
        ;
        const int64_t v1035 = 1;
        // pto: %241
        ;
        int64_t v1036 = v64 * v65;
        // pto: %241
        ;
        int64_t v1037 = v1035 * v1036;
        // pto: %241
        ;
        pto::Shape<1, 1, 1, 1, -1> v1038 = pto::Shape<1, 1, 1, 1, -1>(v1033, v1034, v1035, v64, v757);
        // pto: %241
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1039 = pto::Stride<-1, -1, -1, -1, -1>(v1034 * v1037, v1037, v1036, v65, v64);
        // pto: %241
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1040 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1032 + (v1031 + v987 * v65 + v1030 * v64), v1038, v1039);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v1027, v1040);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %30
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1041 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %30
        ;
        uint64_t v1042 = (uint64_t) v56;
        TASSIGN(v1041, v1042);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        RoundMode v1043 = RoundMode::CAST_ROUND;
        SaturationMode v1044 = SaturationMode::ON;
        TCVT(v1041, v1027, v1043, v1044);
        // pto: %31
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1045 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %31
        ;
        uint64_t v1046 = (uint64_t) v56;
        TASSIGN(v1045, v1046);
        pipe_barrier(PIPE_V);
        TMULS(v1045, v1041, v982);
        // pto: %32
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1047 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %32
        ;
        uint64_t v1048 = (uint64_t) v36;
        TASSIGN(v1047, v1048);
        // pto: %243
        ;
        int64_t v1049 = (int64_t) ((uint64_t) v754 + (uint64_t) v95);
        // pto: %244
        ;
        int64_t v1050 = v1049 < v73 ? v73 : v1049;
        // pto: %245
        ;
        const int64_t v1051 = 0;
        // pto: %245
        ;
        __gm__ bfloat16_t* v1052 = PTOAS__GLOBAL_TENSOR_DATA(v113);
        // pto: %245
        ;
        const int64_t v1053 = 1;
        // pto: %245
        ;
        const int64_t v1054 = 1;
        // pto: %245
        ;
        const int64_t v1055 = 1;
        // pto: %245
        ;
        int64_t v1056 = v64 * v65;
        // pto: %245
        ;
        int64_t v1057 = v1055 * v1056;
        // pto: %245
        ;
        pto::Shape<1, 1, 1, 1, -1> v1058 = pto::Shape<1, 1, 1, 1, -1>(v1053, v1054, v1055, v64, v757);
        // pto: %245
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1059 = pto::Stride<-1, -1, -1, -1, -1>(v1054 * v1057, v1057, v1056, v65, v64);
        // pto: %245
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1060 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1052 + (v1051 + v987 * v65 + v1050 * v64), v1058, v1059);
        TLOAD(v1047, v1060);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // pto: %33
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1061 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %33
        ;
        uint64_t v1062 = (uint64_t) v57;
        TASSIGN(v1061, v1062);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        RoundMode v1063 = RoundMode::CAST_ROUND;
        SaturationMode v1064 = SaturationMode::ON;
        TCVT(v1061, v1047, v1063, v1064);
        // pto: %34
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1065 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %34
        ;
        uint64_t v1066 = (uint64_t) v57;
        TASSIGN(v1065, v1066);
        pipe_barrier(PIPE_V);
        TMULS(v1065, v1061, v984);
        // pto: %35
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1067 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %35
        ;
        uint64_t v1068 = (uint64_t) v56;
        TASSIGN(v1067, v1068);
        pipe_barrier(PIPE_V);
        TADD(v1067, v1045, v1065);
        // pto: %36
        ;
        Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1069 = Tile<TileType::Vec, float, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %36
        ;
        uint64_t v1070 = (uint64_t) v35;
        TASSIGN(v1069, v1070);
        pipe_barrier(PIPE_V);
        TADD(v1069, v1025, v1067);
        // pto: %37
        ;
        Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v1071 = Tile<TileType::Vec, bfloat16_t, 1, 4096, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v64, v757);
        // pto: %37
        ;
        uint64_t v1072 = (uint64_t) v35;
        TASSIGN(v1071, v1072);
        pipe_barrier(PIPE_V);
        RoundMode v1073 = RoundMode::CAST_RINT;
        SaturationMode v1074 = SaturationMode::ON;
        TCVT(v1071, v1069, v1073, v1074);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        v1071.SetValidShape(v64, v757);
        // pto: %249
        ;
        const int64_t v1075 = 0;
        // pto: %249
        ;
        __gm__ bfloat16_t* v1076 = PTOAS__GLOBAL_TENSOR_DATA(v174);
        // pto: %249
        ;
        const int64_t v1077 = 1;
        // pto: %249
        ;
        const int64_t v1078 = 1;
        // pto: %249
        ;
        const int64_t v1079 = 1;
        // pto: %249
        ;
        int64_t v1080 = v64 * v69;
        // pto: %249
        ;
        int64_t v1081 = v1079 * v1080;
        // pto: %249
        ;
        pto::Shape<1, 1, 1, 1, -1> v1082 = pto::Shape<1, 1, 1, 1, -1>(v1077, v1078, v1079, v64, v757);
        // pto: %249
        ;
        pto::Stride<-1, -1, -1, -1, -1> v1083 = pto::Stride<-1, -1, -1, -1, -1>(v1078 * v1081, v1081, v1080, v69, v64);
        // pto: %249
        ;
        GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v1084 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1076 + (v1075 + v987 * v69 + v988 * v64), v1082, v1083);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(v1084, v1071);
      };
      set_flag(PIPE_S, PIPE_V, EVENT_ID0);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
    };
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID6);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}