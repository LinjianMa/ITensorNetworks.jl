# TODO: test needed
function _constrained_minswap_inds_ordering(
  constraint_tree::NamedDiGraph{Vector}, ref_ordering::Vector, tn::ITensorNetwork
)
  leaves = leaf_vertices(constraint_tree)
  root = _root(constraint_tree)
  v_to_order = Dict{Vector,Vector{Set}}()
  for v in post_order_dfs_vertices(constraint_tree, root)
    if v in leaves
      v_to_order[v] = [v[1]...]
      continue
    end
    child_orders = Vector{Vector{Set}}()
    children = child_vertices(constraint_tree, v)
    for inds_vector in v[1]
      cs = filter(c -> c[1] == inds_vector, children)
      @assert length(cs) == 1
      push!(child_orders, v_to_order[cs[1]])
    end
    input_order = [n for n in ref_ordering if n in vcat(child_orders...)]
    # Optimize the ordering in child_orders
    if v[2] == "ordered"
      perms = [child_orders, reverse(child_orders)]
      nswaps = [length(_bubble_sort(vcat(p...), input_order)) for p in perms]
      perms = [perms[i] for i in 1:length(perms) if nswaps[i] == min(nswaps...)]
      output_order = _mincut_permutation(perms, tn)
    else
      output_order = _best_perm_greedy(child_orders, input_order, tn)
    end
    v_to_order[v] = vcat(output_order...)
  end
  return v_to_order[root]
end

function _constrained_minswap_inds_ordering(
  constraint_tree::NamedDiGraph{Vector},
  input_order_1::Vector,
  input_order_2::Vector,
  tn::ITensorNetwork,
  c1_in_leaves::Bool,
  c2_in_leaves::Bool,
)
  inter_igs = intersect(input_order_1, input_order_2)
  left_1, right_1 = _split_array(input_order_1, inter_igs)
  left_2, right_2 = _split_array(input_order_2, inter_igs)
  # @info "lengths of the input partitions",
  # sort([length(left_1), length(right_2), length(left_2), length(right_2)])

  # TODO: this conditions seems not that useful, except the quantum
  # circuit simulation experiment. We should consider removing this
  # once confirming that this is only useful for special cases.
  if c1_in_leaves && !c2_in_leaves
    inputs = collect(permutations([left_1..., right_1...]))
    inputs = [[left_2..., i..., right_2...] for i in inputs]
  elseif !c1_in_leaves && c2_in_leaves
    inputs = collect(permutations([left_2..., right_2...]))
    inputs = [[left_1..., i..., right_1...] for i in inputs]
  else
    num_swaps_1 =
      min(length(left_1), length(left_2)) + min(length(right_1), length(right_2))
    num_swaps_2 =
      min(length(left_1), length(right_2)) + min(length(right_1), length(left_2))
    if num_swaps_1 == num_swaps_2
      inputs_1 = _low_swap_merge(left_1, right_1, left_2, right_2)
      inputs_2 = _low_swap_merge(left_1, right_1, reverse(right_2), reverse(left_2))
      inputs = [inputs_1..., inputs_2...]
    elseif num_swaps_1 > num_swaps_2
      inputs = _low_swap_merge(left_1, right_1, reverse(right_2), reverse(left_2))
    else
      inputs = _low_swap_merge(left_1, right_1, left_2, right_2)
    end
  end
  # ####
  # mincuts = map(o -> _mps_mincut_partition_cost(tn, o), inputs)
  # inputs = [inputs[i] for i in 1:length(inputs) if mincuts[i] == mincuts[argmin(mincuts)]]
  # ####
  constraint_tree_copies = [copy(constraint_tree) for _ in 1:length(inputs)]
  outputs = []
  nswaps_list = []
  for (t, i) in zip(constraint_tree_copies, inputs)
    output = _constrained_minswap_inds_ordering(t, i, tn)
    push!(outputs, output)
    push!(nswaps_list, length(_bubble_sort(output, i)))
  end
  inputs = [inputs[i] for i in 1:length(inputs) if nswaps_list[i] == min(nswaps_list...)]
  outputs = [outputs[i] for i in 1:length(outputs) if nswaps_list[i] == min(nswaps_list...)]
  if length(inputs) == 1
    return inputs[1], outputs[1]
  end
  mincuts = map(o -> _mps_mincut_partition_cost(tn, o), outputs)
  return inputs[argmin(mincuts)], outputs[argmin(mincuts)]
end

function _mincut_permutation(perms::Vector{<:Vector{<:Vector}}, tn::ITensorNetwork)
  if length(perms) == 1
    return perms[1]
  end
  mincuts_dist = map(p -> _mps_mincut_partition_cost(tn, vcat(p...)), perms)
  return perms[argmin(mincuts_dist)]
end

function _best_perm_greedy(vs::Vector{<:Vector}, order::Vector, tn::ITensorNetwork)
  ordered_vs = [vs[1]]
  for v in vs[2:end]
    perms = [insert!(copy(ordered_vs), i, v) for i in 1:(length(ordered_vs) + 1)]
    suborder = filter(n -> n in vcat(perms[1]...), order)
    nswaps = map(p -> length(_bubble_sort(vcat(p...), suborder)), perms)
    perms = [perms[i] for i in 1:length(perms) if nswaps[i] == min(nswaps...)]
    ordered_vs = _mincut_permutation(perms, tn)
  end
  return ordered_vs
end

function _make_list(left_lists, right_lists)
  out_lists = []
  for l in left_lists
    for r in right_lists
      push!(out_lists, [l..., r...])
    end
  end
  return out_lists
end

function _low_swap_merge(l1_left, l1_right, l2_left, l2_right)
  if length(l1_left) < length(l2_left)
    left_lists = [[l2_left..., l1_left...]]
  elseif length(l1_left) > length(l2_left)
    left_lists = [[l1_left..., l2_left...]]
  else
    left_lists = [[l2_left..., l1_left...], [l1_left..., l2_left...]]
  end
  if length(l1_right) < length(l2_right)
    right_lists = [[l1_right..., l2_right...]]
  elseif length(l1_right) > length(l2_right)
    right_lists = [[l2_right..., l1_right...]]
  else
    right_lists = [[l2_right..., l1_right...], [l1_right..., l2_right...]]
  end
  return _make_list(left_lists, right_lists)
end
