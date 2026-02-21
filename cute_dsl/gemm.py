import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils

# ---------------------------------------------------------------------------
# Blackwell Device Kernel (TMA Load + MMA Pipeline + SIMT Epilogue)
# ---------------------------------------------------------------------------
@cute.kernel
def gemm_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    tma_tensor_a: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    tma_tensor_b: cute.Tensor,
    mC_mnl: cute.Tensor,
    cluster_layout_vmnk: cute.Layout,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    alpha: cute.Float32,
    beta: cute.Float32,
):
    # Setup base coordinates
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    bidx, bidy, bidz = cute.arch.block_idx()
    mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
    is_leader_cta = (mma_tile_coord_v == 0)
    
    # Resolve cluster shapes and Multicast masks
    cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    block_in_cluster_coord = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
    
    cta_coord = (bidx, bidy, bidz)
    mma_tile_coord_mnl = (
        cta_coord[0] // cute.size(tiled_mma.thr_id.shape),
        cta_coord[1],
        cta_coord[2],
    )
    num_ab_stage = 3
    use_2cta_instrs = True
    mma_tiler = (256, 128, cute.size(tiled_mma.shape_mnk, mode=[2]) * 4)
    cta_tile_shape_mnk = (
        mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
        mma_tiler[1],
        mma_tiler[2],
    )
    epi_tile = cta_tile_shape_mnk[:2]

    # 1. Pipeline and Storage configuration
    num_acc_stage = 1
    @cute.struct
    class SharedStorage:
        ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]
        acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, num_acc_stage * 2]
        tmem_dealloc_mbar_ptr: cutlass.Int64
        tmem_holding_buf: cutlass.Int32

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    num_mcast_ctas_a = cute.size(cluster_layout_vmnk.shape[2])
    num_mcast_ctas_b = cute.size(cluster_layout_vmnk.shape[1])
    is_a_mcast = num_mcast_ctas_a > 1
    is_b_mcast = num_mcast_ctas_b > 1
    a_copy_size = cute.size_in_bytes(
        cutlass.Float16, cute.slice_(a_smem_layout_staged, (None, None, None, 0))
    )
    b_copy_size = cute.size_in_bytes(
        cutlass.Float16, cute.slice_(b_smem_layout_staged, (None, None, None, 0))
    )
    num_tma_load_bytes = (a_copy_size + b_copy_size) * cute.size(tiled_mma.thr_id.shape)

    num_tma_producer = num_mcast_ctas_a + num_mcast_ctas_b - 1
    ab_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    ab_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
    
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
        num_stages=num_ab_stage,
        producer_group=ab_producer_group,
        consumer_group=ab_consumer_group,
        tx_count=num_tma_load_bytes,
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    ).make_participants()

    # TMEM / pipeline parameters
    threads_per_cta = 128

    # Accumulator pipeline barrier for async UMMA completion -> TMEM consumption.
    acc_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    acc_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, threads_per_cta)
    acc_pipeline = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
        num_stages=num_acc_stage,
        producer_group=acc_producer_group,
        consumer_group=acc_consumer_group,
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    )
    acc_producer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, num_acc_stage
    )
    acc_consumer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, num_acc_stage
    )

    # TMEM Initialization
    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=threads_per_cta)
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
        is_two_cta=use_2cta_instrs,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
    )

    pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

    # Shared buffers layout mappings
    sA = smem.allocate_tensor(cutlass.Float16, a_smem_layout_staged.outer, byte_alignment=128, swizzle=a_smem_layout_staged.inner)
    sB = smem.allocate_tensor(cutlass.Float16, b_smem_layout_staged.outer, byte_alignment=128, swizzle=b_smem_layout_staged.inner)

    a_full_mcast_mask = cpasync.create_tma_multicast_mask(
        cluster_layout_vmnk, block_in_cluster_coord, mcast_mode=2
    )
    b_full_mcast_mask = cpasync.create_tma_multicast_mask(
        cluster_layout_vmnk, block_in_cluster_coord, mcast_mode=1
    )

    # Local tile partition on GMEM tensors.
    gA_mkl = cute.local_tile(
        tma_tensor_a, cute.slice_(mma_tiler, (None, 0, None)), (None, None, None)
    )
    gB_nkl = cute.local_tile(
        tma_tensor_b, cute.slice_(mma_tiler, (0, None, None)), (None, None, None)
    )
    gC_mnl = cute.local_tile(
        mC_mnl, cute.slice_(mma_tiler, (None, None, 0)), (None, None, None)
    )
    k_tile_cnt = cute.size(gA_mkl, mode=[3])
    thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
    tCgA = thr_mma.partition_A(gA_mkl)
    tCgB = thr_mma.partition_B(gB_nkl)
    tCgC = thr_mma.partition_C(gC_mnl)

    # 2. Partitions mappings for Blackwell Async Ops
    a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a, block_in_cluster_coord[2], a_cta_layout,
        cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3)
    )

    b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b, block_in_cluster_coord[1], b_cta_layout,
        cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3)
    )

    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
    tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

    pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

    # Allocate TMEM locally for Accumulators
    num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)
    tmem.allocate(num_tmem_alloc_cols)
    tmem.wait_for_alloc()

    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

    tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
    tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

    # 3. Pipelined TMA fetch mainloop -> tcgen05
    prefetch_k_tile_cnt = cutlass.min(num_ab_stage - 2, k_tile_cnt)
    if warp_idx == 0:
        for k_tile_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):
            producer_handle = ab_producer.acquire_and_advance()
            cute.copy(tma_atom_a, tAgA[(None, k_tile_idx)], tAsA[(None, producer_handle.index)], tma_bar_ptr=producer_handle.barrier, mcast_mask=a_full_mcast_mask)
            cute.copy(tma_atom_b, tBgB[(None, k_tile_idx)], tBsB[(None, producer_handle.index)], tma_bar_ptr=producer_handle.barrier, mcast_mask=b_full_mcast_mask)

        peek_ab_full_status = cutlass.Boolean(False)
        if is_leader_cta:
            peek_ab_full_status = ab_consumer.try_wait()
        peek_ab_empty_status = ab_producer.try_acquire()

        for k_tile_idx in cutlass.range(k_tile_cnt):
            if k_tile_idx < k_tile_cnt - prefetch_k_tile_cnt:
                producer_handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
                cute.copy(tma_atom_a, tAgA[(None, producer_handle.count)], tAsA[(None, producer_handle.index)], tma_bar_ptr=producer_handle.barrier, mcast_mask=a_full_mcast_mask)
                cute.copy(tma_atom_b, tBgB[(None, producer_handle.count)], tBsB[(None, producer_handle.index)], tma_bar_ptr=producer_handle.barrier, mcast_mask=b_full_mcast_mask)

            if is_leader_cta:
                consumer_handle = ab_consumer.wait_and_advance(peek_ab_full_status)
                num_kblks = cute.size(tCrA, mode=[2])
                for kblk_idx in cutlass.range(num_kblks, unroll_full=True):
                    kblk_crd = (None, None, kblk_idx, consumer_handle.index)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[kblk_crd], tCrB[kblk_crd], tCtAcc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                consumer_handle.release()

            if k_tile_idx + 1 < k_tile_cnt - prefetch_k_tile_cnt:
                peek_ab_empty_status = ab_producer.try_acquire()
            if k_tile_idx + 1 < k_tile_cnt and is_leader_cta:
                peek_ab_full_status = ab_consumer.try_wait()

        if is_leader_cta:
            acc_pipeline.producer_commit(acc_producer_state)

    tmem.relinquish_alloc_permit()
    acc_pipeline.consumer_wait(acc_consumer_state)

    # 4. Math Epilogue (C = alpha * A@B + beta * C) Downcasted back to native FP16 Memory Map
    
    copy_atom_t2r = sm100_utils.get_tmem_load_op(
        epi_tile + (mma_tiler[2],), utils.LayoutEnum.from_tensor(mC_mnl),
        cutlass.Float16, cutlass.Float32, epi_tile, use_2cta_instrs
    )
    tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])
    thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
    
    tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
    
    tCgC_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], epi_tile)
    tTR_gC = thr_copy_t2r.partition_D(tCgC_epi)
    tTR_rC = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, cutlass.Float16)
    
    simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float16)
    tTR_gC = tTR_gC[(None, None, None, None, None, *mma_tile_coord_mnl)]
    tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))
    tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0)].shape, cutlass.Float32)
    
    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
    for subtile_idx in cutlass.range(subtile_cnt):
        # Fetch TMEM state (FP32)
        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
        
        # Load C initial values (FP16) from GMEM
        cute.copy(simt_atom, tTR_gC[(None, None, None, subtile_idx)], tTR_rC)
        
        acc_vec = tTR_rAcc.load()
        c_vec = tTR_rC.load().to(cutlass.Float32) # scale accumulation needs to be done natively on FP32
        
        # Calculate Operation Scaling
        res_vec = alpha * acc_vec + beta * c_vec
        
        # Repackage final result & Ship securely to GMEM Memory
        tTR_rC.store(res_vec.to(cutlass.Float16))
        cute.copy(simt_atom, tTR_rC, tTR_gC[(None, None, None, subtile_idx)])

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)

    if warp_idx == 0:
        ab_producer.tail()

# ---------------------------------------------------------------------------
# Safe fallback kernel for 2D GEMM with explicit bounds checks.
# Used for small / non-tile-aligned shapes where the fast path has OOB risk.
# ---------------------------------------------------------------------------
@cute.kernel
def gemm_kernel_fallback_2d(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    M: cutlass.Int32,
    N: cutlass.Int32,
    K: cutlass.Int32,
    alpha: cute.Float32,
    beta: cute.Float32,
):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    m_idx = bidx * 16 + tidx
    n_idx = bidy * 16 + tidy

    if m_idx < M and n_idx < N:
        acc = cutlass.Float32(0.0)
        for kk in cutlass.range(K, unroll=1):
            acc += A[(m_idx, kk)].to(cutlass.Float32) * B[(kk, n_idx)].to(cutlass.Float32)
        c_prev = C[(m_idx, n_idx)].to(cutlass.Float32)
        out_val = alpha * acc + beta * c_prev
        C[(m_idx, n_idx)] = out_val.to(C.element_type)

# ---------------------------------------------------------------------------
# Target JIT Entry Point (Remains unchained)
# ---------------------------------------------------------------------------
@cute.jit
def solve(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    M: cutlass.Int32,
    N: cutlass.Int32,
    K: cutlass.Int32,
    alpha: cute.Float32,
    beta: cute.Float32,
):
    c_shape = C.layout.shape
    # Fast kernel currently has no edge predication; use it only when dimensions
    # are aligned to the CTA tile (128x128) and at least one full tile.
    use_fast_kernel = cutlass.const_expr(
        c_shape[0] >= 128
        and c_shape[1] >= 128
        and (c_shape[0] % 128 == 0)
        and (c_shape[1] % 128 == 0)
    )

    if cutlass.const_expr(not use_fast_kernel):
        gemm_kernel_fallback_2d(
            A, B, C, M, N, K, alpha, beta
        ).launch(
            grid=(cute.ceil_div(M, 16), cute.ceil_div(N, 16), 1),
            block=[16, 16, 1],
        )
        return

    # 2D interface expected by external runners:
    # A: (M, K), B: (K, N), C: (M, N)
    # Convert to internal 3D views with batch L=1:
    # A_reshaped: (M, K, 1), B_reshaped: (N, K, 1), C_reshaped: (M, N, 1)
    # Note: B is transposed as a view (K,N) -> (N,K) for the kernel's expected layout.
    a_shape = A.layout.shape
    a_stride = A.layout.stride
    b_shape = B.layout.shape
    b_stride = B.layout.stride
    c_stride = C.layout.stride

    A_reshaped = cute.make_tensor(
        A.iterator,
        cute.make_layout((a_shape[0], a_shape[1], 1), stride=(a_stride[0], a_stride[1], 0)),
    )
    B_reshaped = cute.make_tensor(
        B.iterator,
        cute.make_layout((b_shape[1], b_shape[0], 1), stride=(b_stride[1], b_stride[0], 0)),
    )
    C_reshaped = cute.make_tensor(
        C.iterator,
        cute.make_layout((c_shape[0], c_shape[1], 1), stride=(c_stride[0], c_stride[1], 0)),
    )

    # Keep M/N/K in the signature for caller compatibility.
    _ = M
    _ = N
    _ = K

    # SM100 architectural layout constants ensuring 2CTA optimal bounds.
    use_2cta_instrs = True
    mma_tiler_mn = (256, 128)
    cluster_shape_mn = (2, 1)

    # Fetch layouts and determine major modes to assign native tiled boundaries
    a_major_mode = utils.LayoutEnum.from_tensor(A_reshaped).mma_major_mode()
    b_major_mode = utils.LayoutEnum.from_tensor(B_reshaped).mma_major_mode()
    
    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        cutlass.Float16, a_major_mode, b_major_mode, cutlass.Float32, tcgen05.CtaGroup.TWO, mma_tiler_mn
    )

    mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
    mma_tiler = (mma_tiler_mn[0], mma_tiler_mn[1], mma_inst_shape_k * 4)
    cta_tile_shape_mnk = (
        mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
        mma_tiler[1],
        mma_tiler[2],
    )

    cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((*cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,))
    
    # 3-Stage Pipeline allocations parameters for Async loading overlaps
    num_ab_stage = 3
    a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, cutlass.Float16, num_ab_stage)
    b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler, cutlass.Float16, num_ab_stage)

    # Establish localized TMA atoms referencing 
    a_op = sm100_utils.cluster_shape_to_tma_atom_A(cluster_shape_mn, tiled_mma.thr_id)
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        a_op, A_reshaped, cute.slice_(a_smem_layout, (None, None, None, 0)), mma_tiler, tiled_mma, cluster_layout_vmnk.shape
    )

    b_op = sm100_utils.cluster_shape_to_tma_atom_B(cluster_shape_mn, tiled_mma.thr_id)
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        b_op, B_reshaped, cute.slice_(b_smem_layout, (None, None, None, 0)), mma_tiler, tiled_mma, cluster_layout_vmnk.shape
    )

    epi_tile = cta_tile_shape_mnk[:2]
    
    # Grid evaluation from output tensor layout (batch L preserved).
    grid = cute.round_up(
        (
            cute.ceil_div(C_reshaped.layout.shape[0], cta_tile_shape_mnk[0]),
            cute.ceil_div(C_reshaped.layout.shape[1], cta_tile_shape_mnk[1]),
            C_reshaped.layout.shape[2],
        ),
        (*cluster_shape_mn, 1),
    )

    # Kernel JIT launch evaluating inputs configuration
    gemm_kernel(
        tiled_mma, tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, C_reshaped,
        cluster_layout_vmnk, a_smem_layout, b_smem_layout,
        alpha, beta
    ).launch(
        grid=grid,
        block=[128, 1, 1],  # 128 threads required per 2CTA block path
        cluster=(*cluster_shape_mn, 1),
    )
