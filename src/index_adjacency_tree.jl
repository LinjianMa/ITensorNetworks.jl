function boundary_state(v::Tuple{Tuple,String}, adj_igs::Set)
  if Set(Leaves(v[1])) == adj_igs
    return "all"
  end
  if v[2] == "unordered"
    filter_children = filter(c -> issubset(adj_igs, Set(Leaves(c))), v[1])
    # length(filter_children) < 1 means adj_igs is distributed in multiple children
    @assert length(filter_children) <= 1
    if length(filter_children) == 1
      return "middle"
    end
    # TODO: if more than 1 children contain adj_igs, currently we don't reorder the
    # leaves. This may need to be optimized later.
    return "invalid"
  end
  @assert length(v[1]) >= 2
  for i in 1:(length(v[1]) - 1)
    leaves = vcat([Set(Leaves(c)) for c in v[1][1:i]]...)
    if Set(leaves) == adj_igs
      return "left"
    end
  end
  for i in 2:length(v[1])
    leaves = vcat([Set(Leaves(c)) for c in v[1][i:end]]...)
    if Set(leaves) == adj_igs
      return "right"
    end
  end
  return "invalid"
end

function reorder_to_boundary!(
  adj_tree::NamedDiGraph{Tuple{Tuple,String}},
  v::Tuple{Tuple,String},
  target_child::Tuple{Tuple,String};
  direction="right",
)
  new_v = v
  children = child_vertices(adj_tree, v)
  remain_children = setdiff(children, [target_child])
  @assert length(remain_children) >= 1
  if length(remain_children) == 1
    remain_child = remain_children[1]
    if direction == "right"
      new_v = ((remain_child[1], target_child[1]), "ordered")
    else
      new_v = ((target_child[1], remain_child[1]), "ordered")
    end
    if new_v != v
      _add_vertex_edges!(
        adj_tree, new_v; children=children, parent=parent_vertex(adj_tree, v)
      )
      rem_vertex!(adj_tree, v)
    end
  else
    new_child = (Tuple([v[1] for v in remain_children]), "unordered")
    _add_vertex_edges!(adj_tree, new_child; children=remain_children, parent=v)
    if direction == "right"
      new_v = ((new_child[1], target_child[1]), "ordered")
    else
      new_v = ((target_child[1], new_child[1]), "ordered")
    end
    _add_vertex_edges!(
      adj_tree, new_v; children=[new_child, target_child], parent=parent_vertex(adj_tree, v)
    )
    rem_vertex!(adj_tree, v)
  end
  return new_v
end

function _add_vertex_edges!(
  adj_tree::NamedDiGraph{Tuple{Tuple,String}}, v; children=[], parent=nothing
)
  add_vertex!(adj_tree, v)
  if parent != nothing
    add_edge!(adj_tree, parent => v)
  end
  for c in children
    add_edge!(adj_tree, v => c)
  end
end

"""
reorder adj_tree based on adj_igs
"""
function reorder!(
  adj_tree::NamedDiGraph{Tuple{Tuple,String}},
  root::Tuple{Tuple,String},
  adj_igs::Set;
  boundary="right",
)
  @assert boundary in ["left", "right"]
  if boundary_state(root, adj_igs) == "all"
    return false, root
  end
  sub_tree = subgraph(v -> issubset(Set(Leaves(v[1])), Set(Leaves(root[1]))), adj_tree)
  traversal = post_order_dfs_vertices(sub_tree, root)
  path = [v for v in traversal if issubset(adj_igs, Set(Leaves(v[1])))]
  # TODO: below is hacky
  new_root = root
  # get the boundary state
  for v in path
    state = boundary_state(v, adj_igs)
    if state == "invalid"
      return false, root
    end
    children = child_vertices(adj_tree, v)
    # reorder
    if state in ["left", "right"] && state != boundary
      @assert v[2] == "ordered"
      new_v = (reverse(v[1]), v[2])
      # TODO: below is hacky
      if v == root
        new_root = new_v
      end
      _add_vertex_edges!(
        adj_tree, new_v; children=children, parent=parent_vertex(adj_tree, v)
      )
      rem_vertex!(adj_tree, v)
    elseif state == "middle"
      @assert v[2] == "unordered"
      target_child = filter(c -> issubset(adj_igs, Set(Leaves(c[1]))), children)
      @assert length(target_child) == 1
      new_v = reorder_to_boundary!(adj_tree, v, target_child[1]; direction=boundary)
      if v == root
        new_root = new_v
      end
    end
  end
  return true, new_root
end

# Update both keys and values in igs_to_adjacency_tree based on adjacent_igs
# NOTE: keys of `igs_to_adjacency_tree` are target igs, not those adjacent to ancestors
function update_adjacency_tree!(
  adjacency_tree::NamedDiGraph{Tuple{Tuple,String}}, adjacent_igs::Set
)
  @timeit_debug ITensors.timer "update_adjacency_tree" begin
    root_v_to_adjacent_igs = Dict{Tuple{Tuple,String},Set}()
    for r in _roots(adjacency_tree)
      root_igs = Set(Leaves(r[1]))
      common_igs = intersect(adjacent_igs, root_igs)
      if common_igs != Set()
        root_v_to_adjacent_igs[r] = common_igs
      end
    end
    if length(root_v_to_adjacent_igs) == 1
      return nothing
    end
    # if at least 3: for now just put everything together
    if length(root_v_to_adjacent_igs) >= 3
      __roots = keys(root_v_to_adjacent_igs)
      new_v = (Tuple([r[1] for r in __roots]), "unordered")
      _add_vertex_edges!(adjacency_tree, new_v; children=__roots)
      return nothing
    end
    # if 2: assign adjacent_igs to boundary of root_igs (if possible), then concatenate
    v1, v2 = collect(keys(root_v_to_adjacent_igs))
    reordered_1, update_v1 = reorder!(
      adjacency_tree, v1, root_v_to_adjacent_igs[v1]; boundary="right"
    )
    reordered_2, update_v2 = reorder!(
      adjacency_tree, v2, root_v_to_adjacent_igs[v2]; boundary="left"
    )
    if (!reordered_1) && (!reordered_2)
      new_v = ((update_v1[1], update_v2[1]), "unordered")
      _add_vertex_edges!(adjacency_tree, new_v; children=[update_v1, update_v2])
    elseif (reordered_2)
      new_v = ((update_v1[1], update_v2[1]...), "ordered")
      _add_vertex_edges!(
        adjacency_tree,
        new_v;
        children=[update_v1, child_vertices(adjacency_tree, update_v2)...],
      )
      rem_vertex!(adjacency_tree, update_v2)
    elseif (reordered_1)
      new_v = ((update_v1[1]..., update_v2[1]), "ordered")
      _add_vertex_edges!(
        adjacency_tree,
        new_v;
        children=[update_v2, child_vertices(adjacency_tree, update_v1)...],
      )
      rem_vertex!(adjacency_tree, update_v1)
    else
      new_v = ((update_v1[1]..., update_v2[1]...), "ordered")
      _add_vertex_edges!(
        adjacency_tree,
        new_v;
        children=[
          child_vertices(adjacency_tree, update_v1)...,
          child_vertices(adjacency_tree, update_v2)...,
        ],
      )
      rem_vertex!(adjacency_tree, update_v1)
      rem_vertex!(adjacency_tree, update_v2)
    end
  end
end

# Generate the adjacency tree of a contraction tree
# Args:
# ==========
# ctree: the input contraction tree
# path: the path containing ancestor ctrees of the input ctree
# ctree_to_igs: mapping each ctree to neighboring index groups 
function _generate_adjacency_tree(ctree, path, ctree_to_open_edges)
  @timeit_debug ITensors.timer "_generate_adjacency_tree" begin
    # mapping each index group to adjacent input igs
    ig_to_input_adj_igs = Dict{Any,Set}()
    # mapping each igs to an adjacency tree
    adjacency_tree = NamedDiGraph{Tuple{Tuple,String}}()
    for ig in ctree_to_open_edges[ctree]
      ig_to_input_adj_igs[ig] = Set([ig])
      v = ((ig,), "unordered")
      add_vertex!(adjacency_tree, v)
    end
    for (i, a) in path
      inter_igs = intersect(ctree_to_open_edges[a[1]], ctree_to_open_edges[a[2]])
      new_igs_index = (i == 1) ? 2 : 1
      new_igs = setdiff(ctree_to_open_edges[a[new_igs_index]], inter_igs)
      adjacent_igs = union([ig_to_input_adj_igs[ig] for ig in inter_igs]...)
      # `inter_igs != []` means it's a tensor product
      if inter_igs != []
        update_adjacency_tree!(adjacency_tree, adjacent_igs)
      end
      for ig in new_igs
        ig_to_input_adj_igs[ig] = adjacent_igs
      end
      # @info "adjacency_tree", adjacency_tree
      if length(_roots(adjacency_tree)) == 1
        return adjacency_tree
      end
    end
    __roots = _roots(adjacency_tree)
    if length(__roots) > 1
      new_v = (Tuple([r[1] for r in __roots]), "unordered")
      _add_vertex_edges!(adjacency_tree, new_v; children=__roots)
    end
    return adjacency_tree
  end
end

function bubble_sort(v::Vector)
  @timeit_debug ITensors.timer "bubble_sort" begin
    permutations = []
    n = length(v)
    for i in 1:n
      for j in 1:(n - i)
        if v[j] > v[j + 1]
          v[j], v[j + 1] = v[j + 1], v[j]
          push!(permutations, j)
        end
      end
    end
    return permutations
  end
end

function bubble_sort(v1::Vector{IndexGroup}, v2::Vector{IndexGroup})
  index_to_number = Dict{IndexGroup,Int}()
  for (i, v) in enumerate(v2)
    index_to_number[v] = i
  end
  v1_num = [index_to_number[v] for v in v1]
  return bubble_sort(v1_num)
end

function num_adj_swaps(v1::Vector{IndexGroup}, v2::Vector{IndexGroup})
  return length(bubble_sort(v1, v2))
end

num_adj_swaps(v::Vector) = length(bubble_sort(v))

function _mincut_permutation(perms::Vector{<:Vector}, tensors::Vector{ITensor})
  if length(perms) == 1
    return perms[1]
  end
  mincuts_dist = []
  for perm in perms
    _dist = _comb_mincuts_dist(ITensorNetwork(tensors), [igs[1].data for igs in perm])
    push!(mincuts_dist, _dist)
  end
  return perms[argmin(mincuts_dist)]
end

"""
Inplace change `adj_tree` so that its children will be an order of its igs.
"""
function mindist_ordering(
  adj_tree::NamedDiGraph{Tuple{Tuple,String}},
  reference_order::Vector,
  tensors::Vector{ITensor},
)
  leaves = leaf_vertices(adj_tree)
  root = _root(adj_tree)
  v_to_order = Dict{Tuple{Tuple,String},Vector{IndexGroup}}()
  for v in post_order_dfs_vertices(adj_tree, root)
    if v in leaves
      v_to_order[v] = [v[1]...]
      continue
    end
    child_orders = Vector{Vector{IndexGroup}}()
    children = child_vertices(adj_tree, v)
    for inds_tuple in v[1]
      cs = filter(c -> c[1] == inds_tuple, children)
      @assert length(cs) == 1
      push!(child_orders, v_to_order[cs[1]])
    end
    input_order = [n for n in reference_order if n in vcat(child_orders...)]
    # Optimize the ordering in child_orders
    if v[2] == "ordered"
      perms = [child_orders, reverse(child_orders)]
      nswaps = [num_adj_swaps(vcat(p...), input_order) for p in perms]
      perms = [perms[i] for i in 1:length(perms) if nswaps[i] == min(nswaps...)]
      output_order = _mincut_permutation(perms, tensors)
    else
      output_order = _best_perm_greedy(child_orders, input_order, tensors)
    end
    v_to_order[v] = vcat(output_order...)
  end
  nswap = num_adj_swaps(v_to_order[root], reference_order)
  return v_to_order[root], nswap
end

function _best_perm_greedy(vs::Vector{<:Vector}, order::Vector, tensors::Vector{ITensor})
  ordered_vs = [vs[1]]
  for v in vs[2:end]
    perms = [insert!(copy(ordered_vs), i, v) for i in 1:(length(ordered_vs) + 1)]
    suborder = [n for n in order if n in vcat(perms[1]...)]
    nswaps = [num_adj_swaps(vcat(p...), suborder) for p in perms]
    perms = [perms[i] for i in 1:length(perms) if nswaps[i] == min(nswaps...)]
    ordered_vs = _mincut_permutation(perms, tensors)
  end
  return ordered_vs
end

function _merge(left_lists, right_lists)
  out_lists = []
  for l in left_lists
    for r in right_lists
      push!(out_lists, [l..., r...])
    end
  end
  return out_lists
end

function _merge(l1_left, l1_right, l2_left, l2_right)
  left_lists = [[l2_left..., l1_left...], [l1_left..., l2_left...]]
  right_lists = [[l2_right..., l1_right...], [l1_right..., l2_right...]]
  return _merge(left_lists, right_lists)
end

function _low_swap_merge(l1_left, l1_right, l2_left, l2_right, l1_in_leaves, l2_in_leaves)
  if l1_in_leaves
    left_lists = [[l2_left..., l1_left...]]
  elseif l2_in_leaves
    left_lists = [[l1_left..., l2_left...]]
  elseif length(l1_left) < length(l2_left)
    left_lists = [[l2_left..., l1_left...]]
  elseif length(l1_left) > length(l2_left)
    left_lists = [[l1_left..., l2_left...]]
  else
    left_lists = [[l2_left..., l1_left...], [l1_left..., l2_left...]]
  end
  if l1_in_leaves
    right_lists = [[l1_right..., l2_right...]]
  elseif l2_in_leaves
    right_lists = [[l2_right..., l1_right...]]
  elseif length(l1_right) < length(l2_right)
    right_lists = [[l1_right..., l2_right...]]
  elseif length(l1_right) > length(l2_right)
    right_lists = [[l2_right..., l1_right...]]
  else
    right_lists = [[l2_right..., l1_right...], [l1_right..., l2_right...]]
  end
  return _merge(left_lists, right_lists)
end

"""
Given an `adj_tree` and two input adj trees, `input_tree1`, `input_tree2`,
return two adj trees, `tree1` and `tree2`, where `tree1` is a simple concatenation of
`input_tree1` and `input_tree2`, and `tree2` satisfy the constraints in `adj_tree`
and has the minimin number of swaps w.r.t. `tree1`.
"""
function mindist_ordering(
  adj_tree::NamedDiGraph{Tuple{Tuple,String}},
  input_order_1::Vector,
  input_order_2::Vector,
  input1_in_leaves::Bool,
  input2_in_leaves::Bool,
  tensors::Vector{ITensor},
)
  @timeit_debug ITensors.timer "minswap_ordering" begin
    inter_igs = intersect(input_order_1, input_order_2)
    leaves_1_left, leaves_1_right = split_igs(input_order_1, inter_igs)
    leaves_2_left, leaves_2_right = split_igs(input_order_2, inter_igs)
    @info "lengths of the input partitions",
    sort([
      length(leaves_1_left),
      length(leaves_1_right),
      length(leaves_2_left),
      length(leaves_2_right),
    ])
    # TODO: the inputs are not optimal. Consider using recursive bisection.
    if input1_in_leaves && !input2_in_leaves
      inputs = collect(permutations([leaves_1_left..., leaves_1_right...]))
      inputs = [[leaves_2_left..., i..., leaves_2_right...] for i in inputs]
    elseif !input1_in_leaves && input2_in_leaves
      inputs = collect(permutations([leaves_2_left..., leaves_2_right...]))
      inputs = [[leaves_1_left..., i..., leaves_1_right...] for i in inputs]
    else
      num_swaps_1 =
        min(length(leaves_1_left), length(leaves_2_left)) +
        min(length(leaves_1_right), length(leaves_2_right))
      num_swaps_2 =
        min(length(leaves_1_left), length(leaves_2_right)) +
        min(length(leaves_1_right), length(leaves_2_left))
      if num_swaps_1 == num_swaps_2
        inputs_1 = _low_swap_merge(
          leaves_1_left,
          leaves_1_right,
          leaves_2_left,
          leaves_2_right,
          input1_in_leaves,
          input2_in_leaves,
        )
        inputs_2 = _low_swap_merge(
          leaves_1_left,
          leaves_1_right,
          reverse(leaves_2_right),
          reverse(leaves_2_left),
          input1_in_leaves,
          input2_in_leaves,
        )
        inputs = [inputs_1..., inputs_2...]
      elseif num_swaps_1 > num_swaps_2
        inputs = _low_swap_merge(
          leaves_1_left,
          leaves_1_right,
          reverse(leaves_2_right),
          reverse(leaves_2_left),
          input1_in_leaves,
          input2_in_leaves,
        )
      else
        inputs = _low_swap_merge(
          leaves_1_left,
          leaves_1_right,
          leaves_2_left,
          leaves_2_right,
          input1_in_leaves,
          input2_in_leaves,
        )
      end
    end
    adj_tree_copies = [copy(adj_tree) for _ in 1:length(inputs)]
    outputs = []
    nswaps_dist_list = []
    for (t, i) in zip(adj_tree_copies, inputs)
      output, nswaps = mindist_ordering(t, i, tensors)
      push!(outputs, output)
      push!(nswaps_dist_list, nswaps)
    end
    inputs = [
      inputs[i] for i in 1:length(inputs) if nswaps_dist_list[i] == min(nswaps_dist_list...)
    ]
    outputs = [
      outputs[i] for
      i in 1:length(outputs) if nswaps_dist_list[i] == min(nswaps_dist_list...)
    ]
    if length(inputs) == 1
      return inputs[1], outputs[1]
    end
    mincuts = [
      _comb_mincuts_dist(ITensorNetwork(tensors), [igs.data for igs in output]) for
      output in outputs
    ]
    return inputs[argmin(mincuts)], outputs[argmin(mincuts)]
  end
end

function _permute(v::Vector, perms)
  v = copy(v)
  for p in perms
    temp = v[p]
    v[p] = v[p + 1]
    v[p + 1] = temp
  end
  return v
end

"""
Given `igs1` and `igs2`, generate a list of igs that evenly split
`igs1` and `igs2` based on the adjacent swap distance.
"""
function _interpolate(igs1::Vector{IndexGroup}, igs2::Vector{IndexGroup}; size)
  if igs1 == igs2
    return [igs2]
  end
  perms_list = collect(Iterators.partition(bubble_sort(igs1, igs2), size))
  @info "perms_list", perms_list
  out_igs_list = [igs1]
  current_igs = igs1
  for perms in perms_list
    igs = _permute(current_igs, perms)
    push!(out_igs_list, igs)
    current_igs = igs
  end
  return out_igs_list
end
