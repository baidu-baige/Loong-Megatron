"""Plan MoE token offloading from overloaded home experts to spare experts."""

from typing import Literal, Union

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Cache for static tensors in gen_assignment_sinkhorn.
#
# M, home_rank_of_e, topo_bonus, is_self_er are pure functions of the
# (R, E, ranks_per_node, topo_inter_cost, _inter_w) config — they do NOT
# depend on the token counts.  Caching them eliminates ~15 GPU kernel
# launches in the hot path (every training step).
#
# Key: (device_str, R, E, ranks_per_node, topo_inter_cost, inter_w)
# Value: dict of pre-computed contiguous tensors on that device.
# ---------------------------------------------------------------------------
_sinkhorn_static_cache: dict = {}


def _get_sinkhorn_statics(device, R: int, E: int, hpr: int,
                           ranks_per_node: int,
                           topo_inter_cost: float,
                           inter_w: float) -> dict:
    """Return (possibly cached) static tensors for gen_assignment_sinkhorn."""
    key = (str(device), R, E, ranks_per_node, float(topo_inter_cost), float(inter_w))
    if key in _sinkhorn_static_cache:
        return _sinkhorn_static_cache[key]

    rank_nodes     = torch.arange(R, device=device) // ranks_per_node        # [R]
    is_intra_rr    = (rank_nodes.unsqueeze(1) == rank_nodes.unsqueeze(0))    # [R, R]
    temperature    = float(topo_inter_cost)
    M = torch.where(is_intra_rr,
                    torch.tensor(-1.0 / temperature, device=device),
                    torch.tensor(-1.0, device=device)).exp_().contiguous()   # [R, R]

    home_rank_of_e = torch.arange(E, device=device) // hpr                   # [E]
    home_node_e    = home_rank_of_e // ranks_per_node                         # [E]
    tgt_node_r     = torch.arange(R, device=device) // ranks_per_node        # [R]
    is_intra_er    = (home_node_e.unsqueeze(1) == tgt_node_r.unsqueeze(0))   # [E, R]
    is_self_er     = (home_rank_of_e.unsqueeze(1) ==
                      torch.arange(R, device=device).unsqueeze(0))            # [E, R]

    topo_bonus = is_intra_er.float()
    if inter_w != 0.0:
        topo_bonus = topo_bonus.masked_fill(~is_intra_er, inter_w)
    topo_bonus = topo_bonus.masked_fill(is_self_er, 0.0)

    cache = {
        'M':              M,
        'home_rank_of_e': home_rank_of_e,
        'is_intra_er':    is_intra_er,
        'is_self_er':     is_self_er,
        'topo_bonus_f':   topo_bonus.float().contiguous(),
        'is_self_er_i8':  is_self_er.to(torch.int8).contiguous(),
    }
    # Cap cache size to avoid unbounded growth (unlikely in practice).
    if len(_sinkhorn_static_cache) >= 32:
        _sinkhorn_static_cache.clear()
    _sinkhorn_static_cache[key] = cache
    return cache


@triton.jit
def approx_bin_packing_kernel(
    count_spillover_remaining_ptr,
    capacity_spare_sorted_ptr,
    indices_spare_sort_ptr,
    count_tokens_from_expert_to_ep_rank_ptr,
    bucket_items_ptr,
    bucket_heads_ptr,
    bucket_tails_ptr,
    m_ptr,
    interval_size_ptr,
    num_experts: tl.constexpr,
    num_ep_ranks: tl.constexpr,
    num_buckets: tl.constexpr,
    max_items_per_bucket: tl.constexpr,
):
    """
    Triton kernel for approximate bin packing.

    Bucket structure:
    - bucket 0: values >= m
    - bucket i: values in range [m - i*interval_size, m - (i-1)*interval_size)
    - bucket num_buckets-1: smallest positive values
    """
    m = tl.load(m_ptr)
    interval_size = tl.load(interval_size_ptr)

    current_bucket = 0
    done = False

    for ep_rank_iter in range(num_ep_ranks):
        if not done:
            spare_capacity = tl.load(capacity_spare_sorted_ptr + ep_rank_iter)

            if spare_capacity <= 0:
                done = True

            if not done:
                ep_rank_idx = tl.load(indices_spare_sort_ptr + ep_rank_iter)

                expert_idx = -1
                found_item = False

                for search_bucket in range(current_bucket, num_buckets):
                    if not found_item:
                        head = tl.load(bucket_heads_ptr + search_bucket)
                        tail = tl.load(bucket_tails_ptr + search_bucket)

                        has_items = head < tail

                        if has_items:
                            bucket_offset = search_bucket * max_items_per_bucket + head
                            candidate_idx = tl.load(bucket_items_ptr + bucket_offset)
                            remaining = tl.load(count_spillover_remaining_ptr + candidate_idx)

                            tl.store(bucket_heads_ptr + search_bucket, head + 1)

                            if remaining > 0:
                                expert_idx = candidate_idx
                                found_item = True
                                current_bucket = search_bucket
                        else:
                            current_bucket = search_bucket + 1

                if found_item:
                    spillover_to_place = tl.load(count_spillover_remaining_ptr + expert_idx)
                    to_place = tl.minimum(spillover_to_place, spare_capacity)

                    assignment_offset = expert_idx * num_ep_ranks + ep_rank_idx
                    tl.store(count_tokens_from_expert_to_ep_rank_ptr + assignment_offset, to_place)

                    new_remaining = spillover_to_place - to_place
                    tl.store(count_spillover_remaining_ptr + expert_idx, new_remaining)

                    if new_remaining > 0:
                        new_bucket_idx_int = 0
                        if new_remaining < m:
                            bucket_calc = (m - new_remaining) // interval_size + 1
                            new_bucket_idx_int = tl.minimum(bucket_calc, num_buckets - 1)
                            new_bucket_idx_int = tl.maximum(new_bucket_idx_int, 0)

                        if new_bucket_idx_int < num_buckets:
                            tail = tl.load(bucket_tails_ptr + new_bucket_idx_int)
                            bucket_offset = new_bucket_idx_int * max_items_per_bucket + tail
                            tl.store(bucket_items_ptr + bucket_offset, expert_idx)
                            tl.store(bucket_tails_ptr + new_bucket_idx_int, tail + 1)
                else:
                    done = True


def approx_bin_packing_triton(
    count_spillover_per_expert,
    capacity_spare_per_ep_rank,
    avg_tokens_per_ep_rank,
    num_buckets=8,
):
    """
    Triton-accelerated approximate bin packing algorithm.

    Args:
        count_spillover_per_expert: Spillover tokens per expert (num_experts,), sorted descending,
            torch.Tensor on device.
        capacity_spare_per_ep_rank: Spare capacities per EP rank (num_ep_ranks,), sorted descending,
            torch.Tensor on device.
        avg_tokens_per_ep_rank: Average tokens per EP rank (scalar tensor on device)
        num_buckets: Number of buckets for approximate sorting

    Returns:
        tuple: (count_tokens_from_expert_to_ep_rank, count_spillover_remaining)
            - count_tokens_from_expert_to_ep_rank: shape (num_experts, num_ep_ranks)
            - count_spillover_remaining: shape (num_experts,)
    """
    device = count_spillover_per_expert.device
    num_experts = count_spillover_per_expert.shape[0]
    num_ep_ranks = capacity_spare_per_ep_rank.shape[0]

    count_spillover_per_expert = count_spillover_per_expert.to(torch.int32)
    capacity_spare_per_ep_rank = capacity_spare_per_ep_rank.to(torch.int32)

    count_tokens_from_expert_to_ep_rank = torch.zeros(num_experts, num_ep_ranks, dtype=torch.int32, device=device)
    count_spillover_remaining = count_spillover_per_expert.clone()

    m_tensor = avg_tokens_per_ep_rank.to(torch.int32).reshape(1)
    interval_size_tensor = (m_tensor // (num_buckets - 1) if num_buckets > 1 else m_tensor).reshape(1)

    # Compute bucket indices for each expert
    bucket_indices = torch.where(
        count_spillover_per_expert <= 0,
        torch.full_like(count_spillover_per_expert, num_buckets, dtype=torch.int32),
        torch.where(
            count_spillover_per_expert >= m_tensor,
            torch.zeros_like(count_spillover_per_expert, dtype=torch.int32),
            ((m_tensor - count_spillover_per_expert) // interval_size_tensor).to(torch.int32) + 1
        )
    ).clamp(0, num_buckets - 1).to(torch.int32)

    bucket_indices = torch.where(count_spillover_per_expert <= 0, num_buckets, bucket_indices).to(torch.int32)

    max_items_per_bucket = num_experts + num_ep_ranks
    total_buckets = num_buckets + 1

    bucket_items_size = num_buckets * max_items_per_bucket
    bucket_items_with_trash = torch.full((bucket_items_size + 1,), -1, dtype=torch.int32, device=device)
    trash_bin_index = bucket_items_size

    bucket_heads = torch.zeros(num_buckets, dtype=torch.int32, device=device)

    # Count items per bucket
    bucket_counts_all = torch.zeros(total_buckets, dtype=torch.int64, device=device)
    ones = torch.ones(num_experts, dtype=torch.int64, device=device)
    bucket_counts_all.scatter_add_(0, bucket_indices.to(torch.int64), ones)
    bucket_counts = bucket_counts_all[:num_buckets]
    bucket_tails = bucket_counts.to(torch.int32)

    # Compute positions within buckets
    bucket_offsets_all = torch.cat([torch.zeros(1, dtype=torch.int64, device=device),
                                     torch.cumsum(bucket_counts_all, dim=0)[:-1]])

    expert_indices = torch.arange(num_experts, device=device, dtype=torch.int64)
    bucket_start_for_each = bucket_offsets_all[bucket_indices]
    position_within_bucket = (expert_indices - bucket_start_for_each).to(torch.int32)

    valid_mask = bucket_indices < num_buckets
    clamped_bucket_indices = torch.clamp(bucket_indices, 0, num_buckets - 1)
    flat_indices = (
        clamped_bucket_indices.to(torch.int64) * max_items_per_bucket
        + position_within_bucket.to(torch.int64)
    )

    scatter_indices = torch.where(valid_mask, flat_indices, torch.full_like(flat_indices, trash_bin_index))
    scatter_values = torch.where(
        valid_mask,
        expert_indices.to(torch.int32),
        torch.full_like(expert_indices, -1, dtype=torch.int32),
    )
    bucket_items_with_trash.scatter_(0, scatter_indices, scatter_values)

    bucket_items = bucket_items_with_trash[:bucket_items_size].contiguous()

    capacity_spare_sorted = capacity_spare_per_ep_rank
    indices_spare_sort = torch.arange(num_ep_ranks, device=device, dtype=torch.int32)

    grid = (1,)
    approx_bin_packing_kernel[grid](
        count_spillover_remaining,
        capacity_spare_sorted,
        indices_spare_sort,
        count_tokens_from_expert_to_ep_rank,
        bucket_items,
        bucket_heads,
        bucket_tails,
        m_tensor,
        interval_size_tensor,
        num_experts=num_experts,
        num_ep_ranks=num_ep_ranks,
        num_buckets=num_buckets,
        max_items_per_bucket=max_items_per_bucket,
    )

    return count_tokens_from_expert_to_ep_rank, count_spillover_remaining


def one_shot_greedy_assignment(
    count_tokens_per_chunk: torch.Tensor,
    capacity_per_bucket: torch.Tensor
) -> torch.Tensor:
    """
    Perform one-shot greedy assignment between token chunks and buckets.

    This function implements a greedy algorithm that assigns token chunks to buckets
    in a single pass. It calculates the overlap between cumulative token chunks and
    cumulative bucket capacities to determine an assignment strategy.

    The algorithm works by:
    1. Computing cumulative sums for both token chunks and buckets
    2. Finding overlapping ranges between token chunk intervals and bucket intervals
    3. Calculating the overlap amount as the assignment

    Args:
        count_tokens_per_chunk: Token chunks to be assigned
            Shape: [num_token_chunks]
            Type: torch.Tensor (int)
            Description: Each element represents the number of tokens in a chunk
                       that needs to be assigned to buckets

        capacity_per_bucket: Bucket capacities available for assignment
            Shape: [num_buckets]
            Type: torch.Tensor (int)
            Description: Each element represents the capacity of a bucket that can
                       receive token chunks

    Returns:
        count_tokens_from_chunk_to_bucket: Assignment matrix showing token chunk to bucket assignments
            Shape: [num_token_chunks, num_buckets]
            Type: torch.Tensor (same dtype as inputs)
            Description: Element [i, j] represents how many tokens from chunk i
                       are assigned to bucket j. Zero if no assignment.
    """
    count_tokens_per_chunk_cumsum = torch.cumsum(count_tokens_per_chunk, dim=0)
    capacity_per_bucket_cumsum = torch.cumsum(capacity_per_bucket, dim=0)
    count_tokens_per_chunk_start = count_tokens_per_chunk_cumsum - count_tokens_per_chunk
    capacity_per_bucket_start = capacity_per_bucket_cumsum - capacity_per_bucket
    count_tokens_per_chunk_start = count_tokens_per_chunk_start.unsqueeze(1)  # (num_token_chunks, 1)
    count_tokens_per_chunk_end = count_tokens_per_chunk_cumsum.unsqueeze(1)   # (num_token_chunks, 1)
    capacity_per_bucket_start = capacity_per_bucket_start.unsqueeze(0)  # (1, num_buckets)
    capacity_per_bucket_end = capacity_per_bucket_cumsum.unsqueeze(0)   # (1, num_buckets)
    overlap_start = torch.maximum(count_tokens_per_chunk_start, capacity_per_bucket_start)
    overlap_end = torch.minimum(count_tokens_per_chunk_end, capacity_per_bucket_end)
    count_tokens_from_chunk_to_bucket = (overlap_end - overlap_start).clamp(min=0)
    return count_tokens_from_chunk_to_bucket


def reclaim_spare_experts(
    count_tokens_per_ep_rank: torch.Tensor,
    avg_tokens_per_ep_rank: float,
    count_tokens_from_home_expert_to_spare_expert: torch.Tensor,
    threshold_multiplier: float
) -> torch.Tensor:
    """
    Reclaim spare experts by selectively disabling some spare experts.

    This function implements a threshold-based selection algorithm to reduce the
    number of active spare experts by disabling some offloading assignments based
    on load balancing criteria. It helps prevent over-allocation of spare experts
    when the load is already well-balanced.

    The algorithm works by:
    1. Calculating current token loads per EP rank after offloading
    2. Determining a threshold based on average load and multiplier
    3. Sorting offloading assignments by size for each EP rank
    4. Using binary search to find which assignments can be safely disabled
    5. Creating a mask to disable assignments that exceed capacity thresholds

    Args:
        count_tokens_per_ep_rank: Current token distribution per EP rank
            Shape: [num_ep_ranks]
            Type: torch.Tensor (int)
            Description: Number of tokens currently assigned to each EP rank

        avg_tokens_per_ep_rank: Average tokens per EP rank
            Type: int
            Description: Average token load across all EP ranks, used as baseline
                       for threshold calculations

        count_tokens_from_home_expert_to_spare_expert: Current offloading assignment matrix
            Shape: [num_home_experts, num_spare_experts]
            Type: torch.Tensor (int)
            Description: Matrix where [i, j] indicates tokens offloaded from home
                       expert i to spare expert j

        threshold_multiplier: Multiplier for threshold calculation
            Type: float
            Description: Controls how aggressive the filtering is. Lower values
                       result in more spare experts being disabled. If <= 0,
                       no filtering is applied.

    Returns:
        count_tokens_from_home_expert_to_spare_expert: Updated assignment matrix with some assignments disabled
            Shape: [num_home_experts, num_spare_experts]
            Type: torch.Tensor (int)
            Description: Same shape as input but with some offloading assignments
                       set to zero based on threshold criteria

    """
    # Calculate current total tokens per home expert (without offloading)
    num_ep_ranks = count_tokens_per_ep_rank.shape[0]
    num_home_experts, num_spare_experts = count_tokens_from_home_expert_to_spare_expert.shape
    count_tokens_per_home_rank = count_tokens_per_ep_rank.view(num_ep_ranks, -1).sum(dim=1)
    threshold = threshold_multiplier * avg_tokens_per_ep_rank
    max_allowed_load = avg_tokens_per_ep_rank + threshold

    # Calculate current home expert loads after offloading
    count_tokens_after_offloading = count_tokens_per_home_rank - (
        count_tokens_from_home_expert_to_spare_expert.sum(dim=1).view(num_ep_ranks, -1).sum(dim=1)
    )

    # Sort each row in ascending order (zeros will naturally be at the beginning)
    count_tokens_from_home_expert_to_spare_expert_ep_rank_view = (
        count_tokens_from_home_expert_to_spare_expert
        .view(num_ep_ranks, num_home_experts // num_ep_ranks, -1)
        .sum(dim=1)
    )
    count_tokens_sorted, indices_sorted = torch.sort(count_tokens_from_home_expert_to_spare_expert_ep_rank_view, dim=1)
    count_tokens_cumsum = torch.cumsum(count_tokens_sorted, dim=1)

    # Use searchsorted to find where cumulative tokens exceed capacity for each home expert
    # We need to find where count_tokens_cumsum > (max_allowed_load - count_tokens_after_offloading)
    # idx_safe_steps contains the last spare expert index needed for each EP rank (after sorting)
    capacity_remaining = max_allowed_load - count_tokens_after_offloading
    idx_safe_steps = torch.searchsorted(count_tokens_cumsum, capacity_remaining.unsqueeze(1), side='right').squeeze(1)

    # Create a boolean mask based on idx_safe_steps with the same shape as count_tokens_sorted
    # True indicates positions that need to be enabled (after the boundary)
    indices_steps = (
        torch.arange(count_tokens_sorted.shape[1], device=count_tokens_from_home_expert_to_spare_expert.device)
        .unsqueeze(0)
        .expand(count_tokens_sorted.shape[0], -1)
    )
    idx_safe_steps_expanded = idx_safe_steps.unsqueeze(1).expand(-1, count_tokens_sorted.shape[1])
    mask_sorted = indices_steps >= idx_safe_steps_expanded

    # Recover the mask to the original order using indices_sorted
    # Use scatter to map mask_sorted back to original order
    mask_original_order = torch.zeros_like(mask_sorted)
    mask_original_order.scatter_(1, indices_sorted, mask_sorted)

    # Do OR operation along dim=0 to get a 1D tensor indicating which columns to mask
    mask_column = mask_original_order.any(dim=0)

    # Apply the column mask directly to count_tokens_from_home_expert_to_spare_expert
    count_tokens_from_home_expert_to_spare_expert = torch.where(
        mask_column.unsqueeze(0),
        count_tokens_from_home_expert_to_spare_expert,
        torch.zeros_like(count_tokens_from_home_expert_to_spare_expert),
    )
    return count_tokens_from_home_expert_to_spare_expert



def breadth_first_allocation(
    count_tokens_per_expert_from_ep_rank,
    count_tokens_from_home_expert_to_spare_expert,
    device='cuda',
):
    """
    Ultra-compact loop-free version with over-allocation protection

    Args:
        count_tokens_per_expert_from_ep_rank: Token distribution per expert from all EP ranks
            Shape: [num_ep_ranks, num_experts]
            Type: torch.Tensor (int)

        count_tokens_from_home_expert_to_spare_expert: Token assignment map from home to spare experts
            Shape: [num_home_experts, num_spare_experts]
            Type: torch.Tensor (int)

        device: Device to run computation on
            Type: str
            Default: 'cuda'

    Returns:
        tuple containing:
            - count_tokens_offloaded_from_home_to_spare: Tokens offloaded in first pass
                Shape: [num_home_experts, num_spare_experts]
            - count_tokens_per_expert_from_ep_rank_after_offload: Remaining tokens after offload
                Shape: [num_ep_ranks, num_experts]
            - capacity_spare_remaining: Remaining spare capacity
                Shape: [num_home_experts, num_spare_experts]
    """
    count_tokens_per_expert_from_ep_rank = count_tokens_per_expert_from_ep_rank.to(device).float()
    count_tokens_from_home_expert_to_spare_expert = (
        count_tokens_from_home_expert_to_spare_expert.to(device).float()
    )
    # Initial allocation
    idx_supplier = count_tokens_from_home_expert_to_spare_expert.argmax(0)
    mask_active = (count_tokens_from_home_expert_to_spare_expert > 0).sum(0) > 0
    capacity = count_tokens_from_home_expert_to_spare_expert[
        idx_supplier,
        torch.arange(count_tokens_from_home_expert_to_spare_expert.shape[1], device=device),
    ]

    count_tokens_rel = count_tokens_per_expert_from_ep_rank[:, idx_supplier]
    # Use safe division to avoid NaN and division by zero
    denominator = count_tokens_rel.sum(0, keepdim=True)
    probs_proportional = torch.where(
        denominator > 0,
        count_tokens_rel / denominator,
        torch.zeros_like(count_tokens_rel),
    )
    count_tokens_ideal = probs_proportional * capacity
    count_tokens_floors = torch.floor(count_tokens_ideal).int() * mask_active

    count_tokens_offloaded = torch.zeros_like(count_tokens_per_expert_from_ep_rank, dtype=torch.int32)
    idx_supplier_expanded = idx_supplier.unsqueeze(0).expand(
        count_tokens_per_expert_from_ep_rank.shape[0], -1
    )

    # For each new destination k, add t[:, k] to the reduction for old destination idx_supplier[k]
    count_tokens_offloaded.scatter_add_(1, idx_supplier_expanded, count_tokens_floors)
    count_tokens_per_expert_from_ep_rank_after_offload = count_tokens_per_expert_from_ep_rank - count_tokens_offloaded
    capacity_spare_remaining = count_tokens_from_home_expert_to_spare_expert.clone()
    capacity_spare_remaining[
        idx_supplier,
        torch.arange(count_tokens_from_home_expert_to_spare_expert.shape[1], device=device),
    ] -= count_tokens_floors.sum(dim=0)
    return count_tokens_floors, count_tokens_per_expert_from_ep_rank_after_offload, capacity_spare_remaining



def depth_first_allocation(
    count_tokens_per_expert_from_ep_rank: torch.Tensor,
    capacity_spare_remaining: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized batched version of one_shot_greedy_assignment for token routing composition.

    Args:
        count_tokens_per_expert_from_ep_rank: Token distribution per expert from all EP ranks
            Shape: [num_ep_ranks, num_experts]
            Type: torch.Tensor (int)

        capacity_spare_remaining: Remaining spare capacity
            Shape: [num_experts, num_spare_experts]
            Type: torch.Tensor (int)

    Returns:
        tuple containing:
            - count_tokens_offloaded_from_expert_to_spare: Tokens offloaded in second pass
                Shape: [num_ep_ranks, num_spare_experts]
            - count_tokens_per_expert_from_ep_rank_after_second_offload: Remaining tokens
                Shape: [num_ep_ranks, num_experts]
    """
    num_ep_ranks, num_experts = count_tokens_per_expert_from_ep_rank.shape
    num_experts_2, num_spare_experts = capacity_spare_remaining.shape
    assert num_experts == num_experts_2, "Dimension mismatch"

    # For each old destination j, count_tokens_per_expert_from_ep_rank[:,j] are the token chunks
    # Compute cumulative intervals for token chunks along source dimension
    count_tokens_cumsum = torch.cumsum(count_tokens_per_expert_from_ep_rank, dim=0)  # (num_ep_ranks, num_experts)
    count_tokens_start = count_tokens_cumsum - count_tokens_per_expert_from_ep_rank
    count_tokens_end = count_tokens_cumsum

    # Compute bucket intervals for each old destination
    capacity_cumsum = torch.cumsum(capacity_spare_remaining, dim=1)  # (num_experts, num_spare_experts)
    capacity_start = capacity_cumsum - capacity_spare_remaining
    capacity_end = capacity_cumsum

    # Expand dimensions for broadcasting
    count_tokens_start = count_tokens_start.unsqueeze(2)     # (num_ep_ranks, num_experts, 1)
    count_tokens_end = count_tokens_end.unsqueeze(2)         # (num_ep_ranks, num_experts, 1)
    capacity_start = capacity_start.unsqueeze(0)             # (1, num_experts, num_spare_experts)
    capacity_end = capacity_end.unsqueeze(0)                 # (1, num_experts, num_spare_experts)

    # Compute overlaps for all (source, old_dst, new_dst) combinations
    overlap_start = torch.maximum(count_tokens_start, capacity_start)  # (num_ep_ranks, num_experts, num_spare_experts)
    overlap_end = torch.minimum(count_tokens_end, capacity_end)        # (num_ep_ranks, num_experts, num_spare_experts)
    count_tokens_overlap = (overlap_end - overlap_start).clamp(min=0)  # (num_ep_ranks, num_experts, num_spare_experts)

    # Sum over intermediate destinations to get final source->new_dst flows
    count_tokens_offloaded_from_expert_to_spare = count_tokens_overlap.sum(dim=1)  # (num_ep_ranks, num_spare_experts)
    idx_supplier = capacity_spare_remaining.argmax(0)
    idx_supplier_expanded = idx_supplier.unsqueeze(0).expand(
        count_tokens_per_expert_from_ep_rank.shape[0], -1
    )
    count_tokens_per_expert_from_ep_rank_after_second_offload = (
        count_tokens_per_expert_from_ep_rank.scatter_add(
            1,
            idx_supplier_expanded,
            -count_tokens_offloaded_from_expert_to_spare,
        )
    )
    return count_tokens_offloaded_from_expert_to_spare, count_tokens_per_expert_from_ep_rank_after_second_offload


def auction_allocation(
    count_tokens_per_expert_from_ep_rank: torch.Tensor,  # [R, E] int
    count_tokens_from_home_expert_to_spare_expert: torch.Tensor,  # [E, S] int
    ranks_per_node: int = 8,
    inter_weight: float = 1.0 / 3.0,
    epsilon: float = 1.0,
    max_iter: int = 300,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Auction-based exact integer token allocation.  Replaces BFS + DFS.

    Each spare slot s "bids" for the source rank r that provides the best
    net value  (topology_bonus[r,s] - price[r]).  Winning ranks become more
    expensive after each round, naturally diversifying allocation away from
    the greedily-dominant rank while keeping intra-node routes preferred.

    Guarantees:
      - All integers, no floor truncation, no CPU sync.
      - sum_r alloc[r, s]  <= assignment[e_s, s]  for every spare s.
      - sum_s alloc[r, s]  <= count[r, e_s]        for every (r, e_s).
      - Fills each spare slot to min(total_supply_for_expert, assignment) exactly.

    Args:
        count_tokens_per_expert_from_ep_rank : [R, E]  source token counts
        count_tokens_from_home_expert_to_spare_expert : [E, S]  planned capacity
        ranks_per_node  : number of EP ranks per physical node
        inter_weight    : topology bonus for inter-node routes  (intra = 1.0)
        epsilon         : price increment per auction round
        max_iter        : safety cap on iterations

    Returns:
        alloc  [R, S]  tokens routed from rank r to spare s
        supply_remaining [R, E]  tokens NOT offloaded (stay home)
    """
    R, E = count_tokens_per_expert_from_ep_rank.shape
    _, S = count_tokens_from_home_expert_to_spare_expert.shape
    device = count_tokens_per_expert_from_ep_rank.device

    # ── topology bonus  [R, S] ────────────────────────────────────────────
    num_spare_per_rank = max(1, S // R)
    spare_rank = torch.arange(S, device=device) // num_spare_per_rank          # [S]
    src_node   = torch.arange(R, device=device).unsqueeze(1) // ranks_per_node # [R,1]
    dst_node   = (spare_rank // ranks_per_node).unsqueeze(0)                    # [1,S]
    is_self    = torch.arange(R, device=device).unsqueeze(1) == spare_rank      # [R,S]
    is_intra   = (src_node == dst_node) & ~is_self                              # [R,S]
    topo_bonus = torch.where(is_intra,
                             torch.ones(R, S, device=device),
                             torch.full((R, S), inter_weight, device=device))
    topo_bonus = topo_bonus.masked_fill(is_self, 0.0)

    # ── working state ─────────────────────────────────────────────────────
    supply = count_tokens_per_expert_from_ep_rank.clone().long()   # [R, E]
    demand = count_tokens_from_home_expert_to_spare_expert.clone().long()  # [E, S]
    price  = torch.zeros(R, device=device, dtype=torch.float)      # [R]
    alloc  = torch.zeros(R, S, device=device, dtype=torch.long)    # [R, S]

    # Map each spare slot → its source expert  (argmax along E dim)
    expert_for_spare = demand.argmax(0).long()  # [S]

    R_idx = torch.arange(R, device=device)
    S_idx = torch.arange(S, device=device)

    # ── main auction loop ─────────────────────────────────────────────────
    # No .item()/.any() based breaks — fixed iteration count keeps this
    # free of CPU sync and compatible with torch.compile / CUDA graphs.
    for _ in range(max_iter):
        # Current remaining demand per spare  [S]
        cur_demand = demand[expert_for_spare, S_idx]

        # Available supply: supply_rs[r,s] = supply[r, expert_for_spare[s]]  [R,S]
        supply_rs = supply[:, expert_for_spare]

        # Net score  [R, S]  — higher is better for the spare to pick
        score = topo_bonus - price.unsqueeze(1)
        score = score.masked_fill(supply_rs <= 0,          -1e9)
        score = score.masked_fill(cur_demand.unsqueeze(0) <= 0, -1e9)

        # Step 1 – each spare picks its favourite rank
        best_rank = score.argmax(0)                       # [S]

        # Step 2 – conflict resolution: each rank serves only its top spare
        is_chosen    = best_rank.unsqueeze(0) == R_idx.unsqueeze(1)  # [R, S]
        chosen_score = score.masked_fill(~is_chosen, -1e9)           # [R, S]
        winner_spare = chosen_score.argmax(1)                        # [R]
        has_request  = is_chosen.any(1)                              # [R]

        # Step 3 – compute give = min(supply, demand) for each winning pair
        winner_expert = expert_for_spare[winner_spare]               # [R]
        give = torch.minimum(
            supply[R_idx, winner_expert],
            demand[winner_expert, winner_spare],
        ) * has_request.long()                                       # [R]

        # No .item() / .sum()==0 break — masked no-op when give is all-zero.

        # Step 4 – apply allocation
        alloc[R_idx, winner_spare] += give

        # Update supply  (flat scatter: supply[r, e_r] -= give[r])
        supply.view(-1).scatter_add_(0, R_idx * E + winner_expert, -give)

        # Update demand  (flat scatter: demand[e_r, s_r] -= give[r])
        demand.view(-1).scatter_add_(0, winner_expert * S + winner_spare, -give)

        # Step 5 – price ascent: winning ranks get more expensive
        price += (give > 0).float() * epsilon

    return alloc, supply   # supply = remaining (not offloaded) tokens


@triton.jit
def reroute_tokens_w_permute_map_kernel(
    indices_token_sorted_ptr,
    idx_expert_for_offload_ptr,
    count_tokens_to_route_ptr,
    offset_cumulative_ptr,
    map_rerouted_ptr,
    map_permute_ptr,
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    num_offloading_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Reroute tokens and write the source expert into a permute map."""
    idx_offload_expert = tl.program_id(0)

    idx_source_expert = tl.load(idx_expert_for_offload_ptr + idx_offload_expert).to(tl.int64)
    if idx_source_expert < 0:
        return
    count_tokens_to_route = tl.load(count_tokens_to_route_ptr + idx_offload_expert).to(tl.int64)
    offset = tl.load(offset_cumulative_ptr + idx_offload_expert).to(tl.int64)

    indices_token_position = tl.arange(0, BLOCK_SIZE)
    mask_valid = (indices_token_position < count_tokens_to_route)

    # No conditional check needed - masked operations handle empty cases
    offset_base = idx_source_expert * num_tokens + offset
    indices_token = tl.load(indices_token_sorted_ptr + offset_base + indices_token_position,
                           mask=mask_valid, other=0)

    num_total_experts = num_experts + num_offloading_experts

    # These stores are safe even with all-False masks
    idx_flat = indices_token * num_total_experts + idx_source_expert
    tl.store(map_rerouted_ptr + idx_flat, False, mask=mask_valid)

    # Store idx_source_expert to map_permute for valid tokens
    idx_permute = indices_token * num_offloading_experts + idx_offload_expert
    tl.store(map_permute_ptr + idx_permute, idx_source_expert, mask=mask_valid)

    idx_offload_col = num_experts + idx_offload_expert
    idx_flat_rerouted = indices_token * num_total_experts + idx_offload_col
    tl.store(map_rerouted_ptr + idx_flat_rerouted, True, mask=mask_valid)


@triton.jit
def reroute_tokens_kernel(
    indices_token_sorted_ptr,
    idx_expert_for_offload_ptr,
    count_tokens_to_route_ptr,
    offset_cumulative_ptr,
    map_rerouted_ptr,
    probs_rerouted_ptr,
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    num_offloading_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Reroute tokens and move probabilities to spare expert columns."""
    idx_offload_expert = tl.program_id(0)

    idx_source_expert = tl.load(idx_expert_for_offload_ptr + idx_offload_expert).to(tl.int64)
    if idx_source_expert < 0:
        return
    count_tokens_to_route = tl.load(count_tokens_to_route_ptr + idx_offload_expert).to(tl.int64)
    offset = tl.load(offset_cumulative_ptr + idx_offload_expert).to(tl.int64)

    indices_token_position = tl.arange(0, BLOCK_SIZE)
    mask_valid = (indices_token_position < count_tokens_to_route)

    # No conditional check needed - masked operations handle empty cases
    offset_base = idx_source_expert * num_tokens + offset
    indices_token = tl.load(indices_token_sorted_ptr + offset_base + indices_token_position,
                           mask=mask_valid, other=0)

    num_total_experts = num_experts + num_offloading_experts

    # These stores are safe even with all-False masks
    idx_flat = indices_token * num_total_experts + idx_source_expert
    tl.store(map_rerouted_ptr + idx_flat, False, mask=mask_valid)

    # Handle probs_rerouted: read from idx_flat and write to idx_flat_rerouted, zero out idx_flat
    prob_value = tl.load(probs_rerouted_ptr + idx_flat, mask=mask_valid, other=0.0)
    tl.store(probs_rerouted_ptr + idx_flat, 0.0, mask=mask_valid)

    idx_offload_col = num_experts + idx_offload_expert
    idx_flat_rerouted = indices_token * num_total_experts + idx_offload_col
    tl.store(map_rerouted_ptr + idx_flat_rerouted, True, mask=mask_valid)
    tl.store(probs_rerouted_ptr + idx_flat_rerouted, prob_value, mask=mask_valid)


def reroute_tokens_triton(
    map_token_to_expert,
    probs_routing,
    count_tokens_offloading_from_expert,
    count_tokens_offloading_to_spare,
    map_home_expert_to_spare,
):
    """
    Triton-based token rerouting with static shapes for CUDA graph compatibility.

    Args:
        map_token_to_expert: Binary routing map for current EP rank
            Shape: [num_tokens, num_experts]
            Type: torch.Tensor (bool)

        probs_routing: Routing probabilities for current EP rank
            Shape: [num_tokens, num_experts]
            Type: torch.Tensor (float)

        count_tokens_offloading_from_expert: Number of tokens offloading from each expert from the current EP rank
            Shape: [num_experts]
            Type: torch.Tensor (int)

        count_tokens_offloading_to_spare: Number of tokens offloading to each spare expert from the current EP rank
            Shape: [num_spare_experts]
            Type: torch.Tensor (int)

        map_home_expert_to_spare: Boolean map of expert offloading
            Shape: [num_home_experts, num_spare_experts]
            Type: torch.Tensor (bool)

    Returns:
        tuple containing:
            - map_token_to_all_experts: Updated routing map after offloading
                Shape: [num_tokens, num_experts + num_spare_experts]
            - probs_rerouted: Updated routing probabilities after offloading
                Shape: [num_tokens, num_experts + num_spare_experts]
    """
    device = map_token_to_expert.device
    num_tokens, num_experts = map_token_to_expert.shape
    num_spare_experts = count_tokens_offloading_to_spare.shape[0]
    count_tokens_offloading_to_spare = count_tokens_offloading_to_spare.to(torch.int64)
    # Step 1: Generate map with token counts
    count_tokens_from_home_expert_to_spare_expert = (
        map_home_expert_to_spare * count_tokens_offloading_to_spare.unsqueeze(0)
    )

    # Step 2: Compute cumulative offsets
    count_routes_cumsum = torch.cumsum(count_tokens_from_home_expert_to_spare_expert, dim=1)
    offset_cumulative_starts = count_routes_cumsum - count_tokens_from_home_expert_to_spare_expert

    # Step 3: Extract expert mapping
    has_routing_from_expert = map_home_expert_to_spare.any(dim=0)
    indices_expert = torch.argmax(map_home_expert_to_spare.float(), dim=0)
    idx_expert_for_offload = torch.where(has_routing_from_expert, indices_expert, -1)
    indices_offload_expert = torch.arange(num_spare_experts, device=device)

    # Clamp expert indices to valid range for safe indexing (invalid entries will be ignored anyway)
    indices_expert_safe = torch.clamp(idx_expert_for_offload, 0, num_experts - 1)
    # Extract offsets for ALL offloading experts using static indexing
    offset_all = offset_cumulative_starts[indices_expert_safe, indices_offload_expert]

    # Use torch.where to zero out offsets for invalid offloading experts
    offset_cumulative = torch.where(idx_expert_for_offload >= 0, offset_all, 0)

    # Step 5: Create sorted indices
    indices_token_sorted = map_token_to_expert.argsort(dim=0, descending=True).transpose(0, 1).contiguous()

    # Step 6: Initialize output tensor
    map_token_to_all_experts = torch.zeros(num_tokens, num_experts + num_spare_experts,
                    dtype=torch.bool, device=device)
    map_token_to_all_experts[:, :num_experts] = map_token_to_expert.clone()

    # Step 6b: Initialize map_permute tensor with 0
    map_permute = torch.zeros((num_tokens, num_spare_experts),
                             dtype=torch.int64, device=device)

    # Step 7: Launch Triton kernel with permute map
    max_tokens = num_tokens
    BLOCK_SIZE = triton.next_power_of_2(max_tokens)
    grid = (num_spare_experts,)

    # Outpus of the kernel: map_token_to_all_experts, map_permute
    # Akan: We can move all the indices calculation into this kernel.
    reroute_tokens_w_permute_map_kernel[grid](
        indices_token_sorted, idx_expert_for_offload, count_tokens_offloading_to_spare,
        offset_cumulative, map_token_to_all_experts, map_permute, num_tokens, num_experts, num_spare_experts, BLOCK_SIZE
    )

    # Step 8: Apply permute map to probabilities using gather
    # Gather probabilities from source experts for offloading tokens
    probs_gathered = torch.gather(probs_routing, 1, map_permute)

    # Concatenate original probs with gathered probs
    probs_rerouted = torch.cat([probs_routing, probs_gathered], dim=1)

    # Mask with map_token_to_all_experts to zero out probabilities for tokens that are not routed
    probs_rerouted = probs_rerouted * map_token_to_all_experts

    return map_token_to_all_experts, probs_rerouted


@torch.compile
def gen_offloading_plan(
    map_token_to_expert: torch.Tensor,
    probs_routing: torch.Tensor,
    count_tokens_per_expert_from_ep_rank: torch.Tensor,
    ep_rank: Union[torch.Tensor, int],
    num_ep_ranks: int,
    num_spare_experts_per_ep_rank: int = 1,
    threshold_multiplier: float = 0.0,
    dtype_index: torch.dtype = torch.int32,
    assignment_algorithm: Literal[
        "one_shot_greedy",
        "approx_bin_packing",
        "sinkhorn",
        "auction",
    ] = "approx_bin_packing",
    ranks_per_node: int = 8,
    _force_triton: bool | None = None,  # passed through to gen_assignment_sinkhorn for testing
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate an offloading plan to redistribute tokens from overloaded experts to spare experts.

    This function implements a balanced routing algorithm that redistributes tokens
    from overloaded experts to spare experts to achieve better load balancing.
    The algorithm works in phases:
    1. Calculate spillover tokens that need to be offloaded from each expert
    2. Use assignment algorithm to match spillover tokens to spare expert capacity
    3. Apply threshold-based filtering to disable some offloading assignments
    4. Execute the offloading plan using breadth-first and depth-first allocation
    5. Generate the final rerouting map with token movements

    Args:
        map_token_to_expert: Binary routing map for current EP rank
            Shape: [num_tokens_in_ep_rank, num_experts]
            Type: torch.Tensor (bool)
            Description: Binary tensor indicating which tokens are routed to which
                       experts within the current EP rank

        probs_routing: Routing probabilities for current EP rank
            Shape: [num_tokens_in_ep_rank, num_experts]
            Type: torch.Tensor (float)
            Description: Probability values for each token-expert assignment within
                       the current EP rank

        count_tokens_per_expert_from_ep_rank: Token distribution per expert from all EP ranks
            Shape: [num_ep_ranks, num_experts]
            Type: torch.Tensor (int)
            Description: Number of tokens assigned to each expert across all EP ranks

        ep_rank: Current EP rank index
            Type: Union[torch.Tensor, int]
            Description: Index of the current expert parallel rank (0 to num_ep_ranks-1)

        num_ep_ranks: Number of expert parallel ranks
            Type: int
            Description: Total number of expert parallel ranks

        num_spare_experts_per_ep_rank: Number of spare experts per EP rank
            Type: int
            Default: 1
            Description: How many spare experts are available for offloading per
                       EP rank

        threshold_multiplier: Multiplier for threshold-based expert selection
            Type: float
            Default: 0.0
            Description: If > 0, enables threshold-based selection to disable
                       some spare experts based on average token load. Lower
                       values = more aggressive filtering.

        dtype_index: Data type for index tensors
            Type: torch.dtype
            Default: torch.int32
            Description: Data type used for index tensors in the computation

        assignment_algorithm: Algorithm to use for token assignment
            Type: Literal["one_shot_greedy", "approx_bin_packing"]
            Default: "one_shot_greedy"
            Description:
                - "one_shot_greedy": Uses gen_assignment with one_shot_greedy_assignment
                - "approx_bin_packing": Uses gen_assignment_for_approx_bp

    Returns:
        tuple containing:
            - map_token_to_all_experts: Updated routing map after offloading
                Shape: [num_tokens_in_ep_rank, num_experts + num_spare_experts]
                Type: torch.Tensor (bool)
                Description: Binary tensor showing final token assignments including
                           offloaded tokens to spare experts

            - probs_rerouted: Updated routing probabilities after offloading
                Shape: [num_tokens_in_ep_rank, num_experts + num_spare_experts]
                Type: torch.Tensor (float)
                Description: Probability values for final token assignments including
                           offloaded tokens to spare experts

            - map_home_expert_to_spare: Mapping of home experts to spare experts
                Shape: [num_home_experts, num_spare_experts]
                Type: torch.Tensor (bool)
                Description: Boolean matrix where [i,j] = True indicates home expert
                           i offloads tokens to spare expert j
    """

    if assignment_algorithm == "sinkhorn":
        count_tokens_from_home_expert_to_spare_expert = gen_assignment_sinkhorn(
            count_tokens_per_expert_from_ep_rank,
            ep_rank,
            num_ep_ranks,
            num_spare_experts_per_ep_rank,
            ranks_per_node=ranks_per_node,
            dtype_index=dtype_index,
            _force_triton=_force_triton,
        )
        map_home_expert_to_spare = count_tokens_from_home_expert_to_spare_expert > 0
        (
            count_tokens_offloaded_first_pass,
            count_tokens_per_expert_after_first_offload,
            capacity_spare_remaining,
        ) = breadth_first_allocation(
            count_tokens_per_expert_from_ep_rank,
            count_tokens_from_home_expert_to_spare_expert,
        )
        (
            count_tokens_offloaded_second_pass,
            count_tokens_per_expert_after_second_offload,
        ) = depth_first_allocation(
            count_tokens_per_expert_after_first_offload,
            capacity_spare_remaining,
        )
        count_tokens_offloaded_from_ep_rank_to_spare_expert = (
            count_tokens_offloaded_first_pass + count_tokens_offloaded_second_pass
        )
        count_tokens_offloaded_from_ep_rank_from_home_expert = (
            count_tokens_per_expert_from_ep_rank - count_tokens_per_expert_after_second_offload
        )
    elif assignment_algorithm == "auction":
        # Phase 1: sinkhorn plans expert-level routing (topology-aware)
        count_tokens_from_home_expert_to_spare_expert = gen_assignment_sinkhorn(
            count_tokens_per_expert_from_ep_rank,
            ep_rank,
            num_ep_ranks,
            num_spare_experts_per_ep_rank,
            ranks_per_node=ranks_per_node,
            dtype_index=dtype_index,
            _force_triton=_force_triton,
        )
        map_home_expert_to_spare = count_tokens_from_home_expert_to_spare_expert > 0
        # Phase 2: auction distributes tokens per-rank with price-discovery
        #          (no floor loss, topology-aware, no CPU sync)
        count_tokens_offloaded_from_ep_rank_to_spare_expert, supply_remaining = auction_allocation(
            count_tokens_per_expert_from_ep_rank,
            count_tokens_from_home_expert_to_spare_expert,
            ranks_per_node=ranks_per_node,
        )
        count_tokens_offloaded_from_ep_rank_from_home_expert = (
            count_tokens_per_expert_from_ep_rank - supply_remaining
        )
    else:
        if assignment_algorithm == "one_shot_greedy":
            count_tokens_from_home_expert_to_spare_expert, _, _ = gen_assignment(
                count_tokens_per_expert_from_ep_rank,
                ep_rank,
                num_ep_ranks,
                num_spare_experts_per_ep_rank,
                threshold_multiplier,
                dtype_index,
            )
        elif assignment_algorithm == "approx_bin_packing":
            if num_spare_experts_per_ep_rank != 1:
                raise ValueError(f"approx_bin_packing only supports num_spare_experts_per_ep_rank=1, "
                                f"got {num_spare_experts_per_ep_rank}")
            count_tokens_from_home_expert_to_spare_expert, _, _ = gen_assignment_for_approx_bp(
                count_tokens_per_expert_from_ep_rank,
                ep_rank,
                num_ep_ranks,
                dtype_index,
            )
        # Create map_home_expert_to_spare with shape (num_home_experts, num_spare_experts)
        # map_home_expert_to_spare[i][j] indicates if home expert i is offloaded to spare expert j
        map_home_expert_to_spare = count_tokens_from_home_expert_to_spare_expert > 0
        (
            count_tokens_offloaded_first_pass,
            count_tokens_per_expert_after_first_offload,
            capacity_spare_remaining,
        ) = breadth_first_allocation(
            count_tokens_per_expert_from_ep_rank,
            count_tokens_from_home_expert_to_spare_expert,
        )
        (
            count_tokens_offloaded_second_pass,
            count_tokens_per_expert_after_second_offload,
        ) = depth_first_allocation(
            count_tokens_per_expert_after_first_offload,
            capacity_spare_remaining,
        )
        count_tokens_offloaded_from_ep_rank_to_spare_expert = (
            count_tokens_offloaded_first_pass + count_tokens_offloaded_second_pass
        )
        count_tokens_offloaded_from_ep_rank_from_home_expert = (
            count_tokens_per_expert_from_ep_rank - count_tokens_per_expert_after_second_offload
        )

    map_token_to_all_experts, probs_rerouted = reroute_tokens_triton(map_token_to_expert,
                                                         probs_routing,
                                                         count_tokens_offloaded_from_ep_rank_from_home_expert[
                                                             ep_rank
                                                         ].int(),
                                                         count_tokens_offloaded_from_ep_rank_to_spare_expert[
                                                             ep_rank
                                                         ].int().squeeze(),
                                                         map_home_expert_to_spare)

    # postprocess TODO: fused into above kernels
    num_experts = map_token_to_expert.shape[1]
    ep_size = num_ep_ranks
    num_echo_experts = num_spare_experts_per_ep_rank * ep_size
    home_expert_routing_map = map_token_to_all_experts[:, :num_experts].reshape(
        -1, ep_size, num_experts // ep_size
    )
    spare_expert_routing_map = map_token_to_all_experts[:, num_experts:].reshape(
        -1, ep_size, num_echo_experts // ep_size
    )
    map_token_to_all_experts = torch.cat([home_expert_routing_map, spare_expert_routing_map], dim=-1).reshape(
        -1, num_experts + num_echo_experts
    )

    home_expert_probs = probs_rerouted[:, :num_experts].reshape(-1, ep_size, num_experts // ep_size)
    spare_expert_probs = probs_rerouted[:, num_experts:].reshape(
        -1, ep_size, num_echo_experts // ep_size
    )
    probs_rerouted = torch.cat([home_expert_probs, spare_expert_probs], dim=-1).reshape(
        -1, num_experts + num_echo_experts
    )
    return map_token_to_all_experts, probs_rerouted, map_home_expert_to_spare



def gen_assignment(
    count_tokens_per_expert_from_ep_rank: torch.Tensor,
    ep_rank: Union[torch.Tensor, int],
    num_ep_ranks: int,
    num_spare_experts_per_ep_rank: int = 1,
    threshold_multiplier: float = 0.0,
    dtype_index: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate token assignment from home experts to spare experts using one_shot_greedy algorithm.

    For approx_bin_packing, use gen_assignment_for_approx_bp instead.

    Args:
        count_tokens_per_expert_from_ep_rank: Token distribution per expert from all EP ranks
            Shape: [num_ep_ranks, num_experts]
            Type: torch.Tensor (int)

        ep_rank: Current EP rank index
            Type: Union[torch.Tensor, int]

        num_ep_ranks: Number of expert parallel ranks
            Type: int

        num_spare_experts_per_ep_rank: Number of spare experts per EP rank
            Type: int
            Default: 1

        threshold_multiplier: Multiplier for threshold calculation
            Type: float
            Default: 0.0

        dtype_index: Data type for index tensors
            Type: torch.dtype
            Default: torch.int32

    Returns:
        tuple containing:
            - count_tokens_from_home_expert_to_spare_expert: Assignment matrix
                Shape: [num_home_experts, num_spare_experts]
            - count_spillover_per_home_expert: Spillover tokens per expert
                Shape: [num_home_experts]
            - capacity_spare_per_ep_rank: Spare capacity per EP rank
                Shape: [num_ep_ranks]
    """
    # Phase 1: calculate how many tokens need to be offloaded from home experts
    count_spillover_per_home_expert, capacity_spare_per_ep_rank, avg_tokens_per_ep_rank = gen_intermediate(
        count_tokens_per_expert_from_ep_rank,
        ep_rank,
        num_ep_ranks,
        num_spare_experts_per_ep_rank,
        threshold_multiplier,
        dtype_index,
    )

    device = count_tokens_per_expert_from_ep_rank.device
    count_tokens_per_expert = count_tokens_per_expert_from_ep_rank.sum(dim=0).to(dtype_index)
    count_tokens_per_ep_rank = count_tokens_per_expert.view(num_ep_ranks, -1).sum(dim=1)

    # Sort count_spillover_per_home_expert in descending order and capacity_spare_per_ep_rank in descending order
    count_spillover_sorted, indices_spillover_sort = torch.sort(count_spillover_per_home_expert, descending=True)
    capacity_spare_sorted, indices_spare_sort = torch.sort(capacity_spare_per_ep_rank, descending=True)
    # [num_chunks, num_buckets] == [num_experts, num_ep_ranks]
    # Here tokens from an expert are regarded as a single chunk.
    # Space in each ep rank is regarded as a single bucket.
    count_tokens_from_chunk_to_bucket_sorted = one_shot_greedy_assignment(count_spillover_sorted, capacity_spare_sorted)
    # Find top num_spare_experts_per_ep_rank token chunks for each EP rank (on sorted assignment)
    # Get the topk largest chunks (tokens from an expert) for each bucket (each ep rank is a bucket).
    # idx_spare_bucket_max means the index of the topk largest chunks in the sorted space.
    count_spare_bucket_max, idx_spare_bucket_max = torch.topk(
        count_tokens_from_chunk_to_bucket_sorted,
        k=num_spare_experts_per_ep_rank,
        dim=0,
    )
    num_buckets = count_tokens_from_chunk_to_bucket_sorted.shape[1]

    # Map indices_row back to original spillover order
    indices_row_sorted = idx_spare_bucket_max.transpose(0, 1).flatten()
    indices_row = indices_spillover_sort[indices_row_sorted]

    # Map indices_col back to original capacity_spare_per_ep_rank order
    # Original indices_col in sorted space
    indices_ep_rank = torch.arange(num_buckets, device=device).repeat_interleave(num_spare_experts_per_ep_rank)
    indices_spare_slot = torch.arange(num_spare_experts_per_ep_rank, device=device).repeat(num_buckets)
    # Map num_ep_ranks back to original order
    indices_ep_rank_original = indices_spare_sort[indices_ep_rank]
    indices_col = indices_ep_rank_original * num_spare_experts_per_ep_rank + indices_spare_slot

    count_tokens_values = count_spare_bucket_max.transpose(0, 1).flatten()

    # [num_home_experts, num_spare_experts]
    # Entry [i][j] is the number of tokens offloaded from home expert i to spare expert j.
    count_tokens_from_home_expert_to_spare_expert = torch.zeros(
        count_spillover_per_home_expert.shape[0],
        num_buckets * num_spare_experts_per_ep_rank,
        device=device,
        dtype=count_tokens_from_chunk_to_bucket_sorted.dtype,
    )
    count_tokens_from_home_expert_to_spare_expert[indices_row, indices_col] = count_tokens_values
    # Create map_home_expert_to_spare with shape (num_home_experts, num_spare_experts)
    # map_home_expert_to_spare[i][j] indicates if home expert i is offloaded to spare expert j

    # Apply threshold-based offloading expert selection
    if threshold_multiplier > 0:
        count_tokens_from_home_expert_to_spare_expert = reclaim_spare_experts(
            count_tokens_per_ep_rank,
            avg_tokens_per_ep_rank,
            count_tokens_from_home_expert_to_spare_expert,
            threshold_multiplier,
        )
    return count_tokens_from_home_expert_to_spare_expert, count_spillover_per_home_expert, capacity_spare_per_ep_rank


def gen_assignment_for_approx_bp(
    count_tokens_per_expert_from_ep_rank: torch.Tensor,
    ep_rank: Union[torch.Tensor, int],
    num_ep_ranks: int,
    dtype_index: torch.dtype = torch.int32,
    num_buckets: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate token assignment from home experts to EP ranks using approx_bin_packing.

    This function computes spillover and spare capacity, sorts them, runs the
    approx_bin_packing_triton kernel, and recovers the original order.

    Args:
        count_tokens_per_expert_from_ep_rank: Token distribution per expert from all EP ranks
            Shape: [num_ep_ranks, num_experts]
            Type: torch.Tensor (int)

        ep_rank: Current EP rank index
            Type: Union[torch.Tensor, int]

        num_ep_ranks: Number of expert parallel ranks
            Type: int

        dtype_index: Data type for index tensors
            Type: torch.dtype
            Default: torch.int32

        num_buckets: Number of priority buckets for approximate bin packing
            Type: int
            Default: 8

    Returns:
        tuple containing:
            - count_tokens_from_chunk_to_bucket: Assignment matrix in original order
                Shape: [num_experts, num_ep_ranks]
            - count_spillover_per_home_expert: Spillover tokens per expert
                Shape: [num_experts]
            - capacity_spare_per_ep_rank: Spare capacity per EP rank
                Shape: [num_ep_ranks]
    """
    # Calculate spillover and spare capacity (using 1 spare expert per EP rank, no threshold)
    count_spillover_per_home_expert, capacity_spare_per_ep_rank, avg_tokens_per_ep_rank = gen_intermediate(
        count_tokens_per_expert_from_ep_rank,
        ep_rank,
        num_ep_ranks,
        num_spare_experts_per_ep_rank=1,
        threshold_multiplier=0.0,
        dtype_index=dtype_index,
    )

    # Sort spillover and spare capacity in descending order
    count_spillover_sorted, indices_spillover_sort = torch.sort(count_spillover_per_home_expert, descending=True)
    capacity_spare_sorted, indices_spare_sort = torch.sort(capacity_spare_per_ep_rank, descending=True)

    # Run approx bin packing on sorted tensors
    count_tokens_from_chunk_to_bucket_sorted, _ = approx_bin_packing_triton(
        count_spillover_sorted,
        capacity_spare_sorted,
        avg_tokens_per_ep_rank,
        num_buckets=num_buckets
    )

    # Recover original order: unsort rows (experts) and columns (EP ranks)
    inverse_spillover_perm = torch.argsort(indices_spillover_sort)
    inverse_spare_perm = torch.argsort(indices_spare_sort)

    count_tokens_from_chunk_to_bucket = count_tokens_from_chunk_to_bucket_sorted[
        inverse_spillover_perm
    ][:, inverse_spare_perm]
    return (
        count_tokens_from_chunk_to_bucket.to(dtype_index),
        count_spillover_per_home_expert,
        capacity_spare_per_ep_rank,
    )


def gen_intermediate(
    count_tokens_per_expert_from_ep_rank: torch.Tensor,
    ep_rank: Union[torch.Tensor, int],
    num_ep_ranks: int,
    num_spare_experts_per_ep_rank: int = 1,
    threshold_multiplier: float = 0.0,
    dtype_index: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate spillover tokens and spare capacity.

    Args:
        count_tokens_per_expert_from_ep_rank: Token distribution per expert from all EP ranks
            Shape: [num_ep_ranks, num_experts]
            Type: torch.Tensor (int)

        ep_rank: Current EP rank index
            Type: Union[torch.Tensor, int]

        num_ep_ranks: Number of expert parallel ranks
            Type: int

        num_spare_experts_per_ep_rank: Number of spare experts per EP rank
            Type: int
            Default: 1

        threshold_multiplier: Multiplier for threshold calculation
            Type: float
            Default: 0.0

        dtype_index: Data type for index tensors
            Type: torch.dtype
            Default: torch.int32

    Returns:
        tuple containing:
            - count_spillover_per_home_expert: Spillover tokens per expert
                Shape: [num_home_experts]
            - capacity_spare_per_ep_rank: Spare capacity per EP rank
                Shape: [num_ep_ranks]
            - avg_tokens_per_ep_rank: Average tokens per EP rank
                Type: torch.Tensor (scalar)
    """
    device = count_tokens_per_expert_from_ep_rank.device
    count_tokens_per_expert = count_tokens_per_expert_from_ep_rank.sum(dim=0).to(dtype_index)
    count_tokens_per_ep_rank = count_tokens_per_expert.view(num_ep_ranks, -1).sum(dim=1)
    avg_tokens_per_ep_rank = count_tokens_per_ep_rank.sum() // num_ep_ranks
    deviation = count_tokens_per_ep_rank - avg_tokens_per_ep_rank
    capacity_spare_per_ep_rank = torch.relu(-deviation)

    # sort the local experts by token count and place smaller token chunk first
    # Here we:
    # 1. split the token counts into groups by ep rank;
    # 2. sort the token counts within each group by number of tokens.
    count_tokens_per_local_expert_sorted, indices_local_expert_sorted = count_tokens_per_expert.view(
        num_ep_ranks, -1
    ).sort(dim=1)
    # 3. calculate how many tokens are spilled over in each group.
    # Here we treat average tokens per ep rank as the capacity of the ep rank.
    count_spillover_per_expert_sorted_cumsum = (
        count_tokens_per_local_expert_sorted.cumsum(dim=1) - avg_tokens_per_ep_rank
    ).clamp(min=0)
    # From cumulative sum to spillover counts, the first element is equal to the first cumulative sum itself.
    count_spillover_per_expert_sorted = torch.cat(
        [
            count_spillover_per_expert_sorted_cumsum[:, :1],
            torch.diff(count_spillover_per_expert_sorted_cumsum, dim=1),
        ],
        dim=1,
    )
    # 3. Recover spillover counts in original expert order and concat all EP ranks.
    count_spillover_per_home_expert = torch.scatter(
        torch.empty_like(count_spillover_per_expert_sorted),
        1,
        indices_local_expert_sorted,
        count_spillover_per_expert_sorted,
    ).view(-1)

    return count_spillover_per_home_expert, capacity_spare_per_ep_rank, avg_tokens_per_ep_rank


# ---------------------------------------------------------------------------
# Triton kernels for gen_assignment_sinkhorn
# ---------------------------------------------------------------------------

@triton.jit
def _sinkhorn_knopp_kernel(
    M_ptr,          # [R, R] float32, row-major
    supply_ptr,     # [R] float32
    demand_ptr,     # [R] float32
    u_ptr,          # [R] float32  (output: row scaling)
    v_ptr,          # [R] float32  (output: col scaling)
    R: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
):
    """
    Single-CTA Sinkhorn-Knopp: fuse all ITERS into one kernel launch.
    R ≤ 64, so the R×R matrix fits in registers / SRAM.
    Grid: (1,)  —  one CTA does everything.
    """
    # Load M into registers as a flat array [R*R]
    M = tl.load(M_ptr + tl.arange(0, R * R))     # [R*R]
    supply = tl.load(supply_ptr + tl.arange(0, R))  # [R]
    demand = tl.load(demand_ptr + tl.arange(0, R))  # [R]

    u = tl.full([R], 1.0, dtype=tl.float32)
    v = tl.full([R], 1.0, dtype=tl.float32)

    for _ in tl.static_range(ITERS):
        # u = supply / (M @ v)  : row i ->  sum_j M[i,j]*v[j]
        rows = tl.reshape(M, [R, R])   # view as [R, R]
        Mv   = tl.sum(rows * v[None, :], axis=1)   # [R]
        u    = supply / tl.maximum(Mv, EPS)

        # v = demand / (M^T @ u) : col j -> sum_i M[i,j]*u[i]
        MTu  = tl.sum(rows * u[:, None], axis=0)   # [R]
        v    = demand / tl.maximum(MTu, EPS)

    tl.store(u_ptr + tl.arange(0, R), u)
    tl.store(v_ptr + tl.arange(0, R), v)


@triton.jit
def _sinkhorn_phase2_all_rounds_kernel(
    # Read-only static inputs
    topo_bonus_ptr,     # [E, R] float32
    T_er_norm_ptr,      # [E, R] float32
    is_hot_e_ptr,       # [E]    int8
    is_self_er_ptr,     # [E, R] int8
    alpha_ptr,          # []     float32 scalar
    # Mutable state (initialised by caller, updated in-place)
    rem_spill_ptr,      # [E]    float32
    rem_spare_ptr,      # [R]    float32
    # Output: final assignment matrix [E, K*R]  (layout: slot = r*K + k)
    count_matrix_ptr,   # [E, S] float32  (S = R*K, zeroed by caller)
    # Shapes
    E: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    INF: tl.constexpr,
):
    """
    Single-CTA kernel that fuses ALL K rounds of Phase 2 into one launch.

    Eliminates per-round:
      - torch.zeros(E, R) for sel_tokens_k
      - K kernel launches
      - count_matrix.scatter_add_ (K × R index-adds)
      - (used.bool() | ...).to(int8) for K updates

    The `used` mask lives entirely in registers across K rounds — never
    written back to global memory until the very end.

    Register budget (E=256, R=16, K=2):
      topo_bonus, T_er_norm, is_self_er : 4 KB each (16-bit indices avoided)
      used [E, R] : 1 bit per entry → packed into int32 words
      count_matrix accumulated in registers [E, R] per round, written once
      rem_spill [E], rem_spare [R] : live across K rounds

    Grid: (1,)
    """
    e_idx = tl.arange(0, E)   # [E]
    r_idx = tl.arange(0, R)   # [R]

    # ── Load static read-only arrays once ────────────────────────────────────
    topo_bonus = tl.load(topo_bonus_ptr + e_idx[:, None] * R + r_idx[None, :])     # [E, R]
    T_er_norm  = tl.load(T_er_norm_ptr  + e_idx[:, None] * R + r_idx[None, :])     # [E, R]
    is_hot_e   = tl.load(is_hot_e_ptr   + e_idx).to(tl.int1)                       # [E]
    is_self_er = tl.load(is_self_er_ptr + e_idx[:, None] * R + r_idx[None, :]).to(tl.int1)  # [E, R]
    alpha      = tl.load(alpha_ptr)

    # ── Load mutable residuals ────────────────────────────────────────────────
    rem_spill  = tl.load(rem_spill_ptr + e_idx)   # [E]
    rem_spare  = tl.load(rem_spare_ptr + r_idx)   # [R]

    # used[E, R] — accumulates across K rounds, never leaves registers
    used = tl.zeros([E, R], dtype=tl.int1)

    # ── K rounds ─────────────────────────────────────────────────────────────
    for k in tl.static_range(K):
        # Accumulator for this round
        sel_tokens_k = tl.zeros([E, R], dtype=tl.float32)
        matched_r    = tl.full([R], 0, dtype=tl.int1)

        # R inner iterations — use range(R) (NOT tl.static_range) to avoid
        # R-fold register unrolling.  For R=16/32 the unrolling creates
        # enormous register pressure (10+ live [E,R] copies), cutting SM
        # occupancy severely.  A dynamic loop lets the compiler reuse
        # temporaries across iterations.  K is still compile-time unrolled
        # via tl.static_range(K) above, keeping slot arithmetic as constexpr.
        for _inner in range(R):
            effective = tl.minimum(rem_spill[:, None], rem_spare[None, :])  # [E, R]
            score     = effective + alpha * topo_bonus + (alpha * 0.1) * T_er_norm

            spare_ok  = rem_spare[None, :] > 0.0
            spill_ok  = rem_spill[:, None] > 0.0
            valid_er  = is_hot_e[:, None] & ~is_self_er & ~used & ~matched_r[None, :] \
                        & spare_ok & spill_ok
            score     = tl.where(valid_er, score, -INF)

            score_T   = tl.trans(score)
            top1_r    = tl.argmax(score_T, axis=1)
            max_r     = tl.max(score_T, axis=1)
            valid_col = max_r > (-INF / 2.0)
            top1_r    = tl.where(valid_col, top1_r, tl.zeros([R], dtype=tl.int64))

            prop_mask    = (top1_r[None, :] == e_idx[:, None]) & valid_col[None, :]
            prop_score   = tl.where(prop_mask, score, -INF)
            best_col     = tl.argmax(prop_score, axis=1)
            was_proposed = tl.sum(prop_mask.to(tl.int32), axis=1) > 0
            sel          = (best_col[:, None] == r_idx[None, :]) & was_proposed[:, None]

            give      = tl.where(sel, effective, 0.0)
            col_sum   = tl.sum(give, axis=0)
            row_sum   = tl.sum(give, axis=1)

            sel_tokens_k = sel_tokens_k + give
            matched_r    = matched_r | (col_sum > 0.0)
            rem_spill    = tl.maximum(rem_spill - row_sum, 0.0)
            rem_spare    = tl.maximum(rem_spare - col_sum, 0.0)

        # Update cross-round used mask
        used = used | (sel_tokens_k > 0.0)

        # Write this round's assignment into count_matrix at slot r*K+k
        # slot_cols[r] = r*K + k  →  offset = e*S + r*K + k
        slot_r = r_idx * K + k                                                  # [R]
        tl.store(
            count_matrix_ptr + e_idx[:, None] * (R * K) + slot_r[None, :],
            sel_tokens_k,
        )

    # Write back residuals (optional — caller may not need them)
    tl.store(rem_spill_ptr + e_idx, rem_spill)
    tl.store(rem_spare_ptr + r_idx, rem_spare)


def gen_assignment_sinkhorn(
    count_tokens_per_expert_from_ep_rank: torch.Tensor,
    ep_rank,
    num_ep_ranks: int,
    num_spare_experts_per_ep_rank: int = 1,
    ranks_per_node: int = 8,
    topo_inter_cost: float = 3.0,
    topo_bonus_inter_weight: float = -1.0,
    sinkhorn_iters: int = 50,
    dtype_index: torch.dtype = torch.int32,
    _force_triton: bool | None = None,
) -> torch.Tensor:
    """
    Topology-aware expert offloading via Sinkhorn-Knopp + iterative col-top1.

    Zero .item() / CPU-sync in the assignment hot-path.  All tensor ops run
    natively on the device (GPU cuBLAS / CPU BLAS).

    Parameters
    ----------
    topo_inter_cost : float, default 3.0
        Communication cost ratio inter-node / intra-node used in two places:
          • Phase 1 Sinkhorn cost matrix: cost_hc = 1 (intra) or topo_inter_cost (inter).
            Controls how strongly the OT plan prefers intra-node routing.
          • Phase 2 topo_bonus (when topo_bonus_inter_weight < 0):
            inter_weight = 1 / topo_inter_cost.
    topo_bonus_inter_weight : float, default -1.0
        Weight for inter-node pairs in the Phase-2 tie-breaker score.
        Range [0, 1].  intra-node pairs always have weight 1.0.
          • -1.0  (default) → auto: use 1/topo_inter_cost  (coupled with Phase 1)
          •  0.0  → strongly prefer intra; inter-node ranks are never chosen as
                    tie-breaker (still chosen when balance score is strictly higher)
          •  0.33 → moderate preference: intra 3× more preferred than inter
          •  1.0  → topology-blind: intra and inter rank equally as tie-breaker

    ┌──────────────────────────────────────────────────────────────────────────┐
    │ Phase 1  Sinkhorn-Knopp rank-level transport  [H × C]                    │
    │                                                                          │
    │  Supply[h] = load[h] - avg  (hot-rank excess)                            │
    │  Demand[c] = avg - load[c]  (cold-rank spare capacity)                   │
    │  Cost[h,c] = 1 (intra) / topo_inter_cost (inter)                         │
    │  M[h,c]   = exp(-cost / topo_inter_cost)   Gibbs kernel                  │
    │  GEMV loop:  u ← supply/(M v),  v ← demand/(Mᵀ u)                        │
    │  T_er[e, r] = T_rr[home(e), r]  (expand to expert level)                 │
    │                                                                          │
    │ Phase 2  K rounds × C inner iters of col-top1 + single-best assignment   │
    │                                                                          │
    │  Score[e, r] = min(rem_spill[e], rem_spare[r])          ← balance 1st    │
    │              + alpha * topo_bonus[e, r]                  ← topo 2nd      │
    │              + alpha * 0.1 * T_er_norm[e, r]            ← OT hint 3rd    │
    │                                                                          │
    │    topo_bonus = 1.0 (intra) / topo_bonus_inter_weight (inter) / 0 (self) │
    │    alpha      = avg_load / hpr                                           │
    └──────────────────────────────────────────────────────────────────────────┘
    """
    # Resolve topo_bonus inter weight
    _inter_w = (1.0 / topo_inter_cost) if topo_bonus_inter_weight < 0 else topo_bonus_inter_weight

    device = count_tokens_per_expert_from_ep_rank.device
    R   = num_ep_ranks
    K   = num_spare_experts_per_ep_rank
    S   = R * K
    E   = count_tokens_per_expert_from_ep_rank.shape[1]
    hpr = E // R

    _R_po2 = R > 0 and (R & (R - 1)) == 0 and R <= 64
    _E_po2 = E > 0 and (E & (E - 1)) == 0 and E <= 512
    # K only needs to be a small compile-time constant (tl.constexpr), no power-of-2 required.
    _K_ok  = K > 0 and K <= 8
    _use_triton = (device.type == 'cuda' and _E_po2 and _R_po2 and _K_ok) \
                  if _force_triton is None else _force_triton

    # ── Load statistics ───────────────────────────────────────────────────────
    count_tokens_per_expert = count_tokens_per_expert_from_ep_rank.sum(0).float()  # [E]
    load_per_rank = count_tokens_per_expert.view(R, hpr).sum(1)                    # [R]
    avg_load      = load_per_rank.mean()

    # ── Supply / demand ───────────────────────────────────────────────────────
    hot_excess = (load_per_rank - avg_load).clamp(min=0)   # [R]
    spare_cap  = (avg_load - load_per_rank).clamp(min=0)   # [R]
    is_hot     = hot_excess > 0
    supply     = hot_excess
    demand     = spare_cap

    # ── Static tensors (config-only, cached across calls) ────────────────────
    # M, home_rank_of_e, topo_bonus, is_self_er do NOT depend on count_tokens.
    # Fetching them from the cache saves ~15 GPU kernel launches per call.
    _statics       = _get_sinkhorn_statics(device, R, E, hpr,
                                            ranks_per_node, topo_inter_cost, _inter_w)
    M              = _statics['M']                  # [R, R] Gibbs kernel
    home_rank_of_e = _statics['home_rank_of_e']    # [E]
    is_self_er     = _statics['is_self_er']         # [E, R] bool
    _topo_bonus_f  = _statics['topo_bonus_f']       # [E, R] float32 contiguous
    _is_self_er_i8 = _statics['is_self_er_i8']      # [E, R] int8  contiguous

    # ── Phase 1: Sinkhorn-Knopp on [R, R] ────────────────────────────────────
    # Scale supply so total supply ≤ total demand.
    supply = supply * (demand.sum() / supply.sum().clamp(min=1.0)).clamp(max=1.0)

    EPS = 1e-8
    u = torch.ones(R, device=device)
    v = torch.ones(R, device=device)

    if device.type == 'cuda' and _R_po2:
        _sinkhorn_knopp_kernel[(1,)](
                     M, supply, demand, u, v,
                     R=R, ITERS=sinkhorn_iters, EPS=EPS)
    else:
        for _ in range(sinkhorn_iters):
            u = supply / torch.mv(M, v).clamp(min=EPS)
            v = demand / torch.mv(M.t(), u).clamp(min=EPS)

    T_rr = u.unsqueeze(1) * M * v.unsqueeze(0)               # [R, R]
    # Hard column cap.
    T_rr = T_rr * (demand / T_rr.sum(0).clamp(min=EPS)).clamp(max=1.0).unsqueeze(0)
    T_er = T_rr[home_rank_of_e]                               # [E, R]

    # ── Dynamic scalars (depend on count_tokens) ──────────────────────────────
    row_budget = hot_excess[home_rank_of_e].clamp(min=1.0)    # [E]
    T_er_norm  = T_er / row_budget.unsqueeze(1)               # [E, R]
    alpha      = avg_load / max(hpr, 1)
    is_hot_e   = is_hot[home_rank_of_e]                       # [E] bool

    # ── Per-expert spillover ──────────────────────────────────────────────────
    count_per_local = count_tokens_per_expert.view(R, hpr)
    sorted_local, sort_idx = count_per_local.sort(dim=1)
    cum_excess   = (sorted_local.cumsum(1) - avg_load).clamp(min=0)
    spill_sorted = torch.cat([cum_excess[:, :1], cum_excess.diff(dim=1)], dim=1)
    spill_flat   = torch.zeros_like(count_per_local)
    spill_flat.scatter_(1, sort_idx, spill_sorted)
    spill = spill_flat.view(E)                                 # [E]

    # ── Phase 2: K×R iterative assignment ────────────────────────────────────
    INF = 1e9

    _T_er_norm_f = T_er_norm.float().contiguous()
    _is_hot_e_i8 = is_hot_e.to(torch.int8).contiguous()
    _alpha_t     = alpha.float().contiguous() if isinstance(alpha, torch.Tensor) \
                   else torch.tensor(float(alpha), dtype=torch.float32, device=device)

    rem_spill = spill.float().clone()
    rem_spare = spare_cap.float().clone()

    if _use_triton:
        # ALL K rounds in a SINGLE kernel launch.
        count_matrix = torch.zeros(E, S, dtype=torch.float32, device=device)
        _sinkhorn_phase2_all_rounds_kernel[(1,)](
            _topo_bonus_f, _T_er_norm_f,
            _is_hot_e_i8, _is_self_er_i8,
            _alpha_t,
            rem_spill, rem_spare,
            count_matrix,
            E=E, R=R, K=K, INF=INF,
        )
        return count_matrix.long().to(dtype_index)

    # ── PyTorch fallback ──────────────────────────────────────────────────────
    count_matrix = torch.zeros(E, S, device=device)
    used   = torch.zeros(E, R, dtype=torch.int8, device=device)
    r_idx  = torch.arange(R, device=device)
    is_self_er_bool = is_self_er  # bool tensor, for fallback only

    for k in range(K):
        sel_tokens_k = torch.zeros(E, R, dtype=torch.float32, device=device)
        matched_r_k = torch.zeros(R, dtype=torch.int8, device=device)
        for _inner in range(R):
            effective = torch.minimum(
                rem_spill.unsqueeze(1).expand(E, R),
                rem_spare.unsqueeze(0).expand(E, R))
            score = effective + _alpha_t * _topo_bonus_f + (_alpha_t * 0.1) * _T_er_norm_f

            invalid = (_is_hot_e_i8.unsqueeze(1) == 0) | (is_self_er_bool) \
                      | (used.bool()) | (matched_r_k.bool().unsqueeze(0)) \
                      | (rem_spare.unsqueeze(0) <= 0) \
                      | (rem_spill.unsqueeze(1) <= 0)
            score = score.masked_fill(invalid, -INF)

            _, top1_r  = score.topk(1, dim=0)
            top1_r     = top1_r.squeeze(0)
            valid_col  = score.max(0).values > (-INF / 2)
            valid_top1 = top1_r.masked_fill(~valid_col, 0)

            prop_mask = torch.zeros(E, R, dtype=torch.bool, device=device)
            prop_mask.scatter_(0, valid_top1.unsqueeze(0), valid_col.unsqueeze(0))
            prop_score = score.masked_fill(~prop_mask, -INF)
            best_col   = prop_score.max(dim=1).indices

            was_proposed = prop_mask.any(1)
            sel = torch.zeros(E, R, device=device)
            sel.scatter_(1, best_col.unsqueeze(1), was_proposed.float().unsqueeze(1))

            give = sel * torch.min(
                rem_spill.unsqueeze(1).expand(E, R),
                rem_spare.unsqueeze(0).expand(E, R))

            sel_tokens_k = sel_tokens_k + give
            matched_r_k  = (matched_r_k.bool() | (give.sum(0) > 0)).to(torch.int8)
            rem_spill    = (rem_spill - give.sum(1)).clamp(min=0)
            rem_spare    = (rem_spare - give.sum(0)).clamp(min=0)

        slot_cols = r_idx * K + k
        count_matrix.scatter_add_(1, slot_cols.unsqueeze(0).expand(E, R), sel_tokens_k)
        used = (used.bool() | (sel_tokens_k > 0)).to(torch.int8)

    # ── Return ────────────────────────────────────────────────────────────────
    return count_matrix.long().to(dtype_index)
