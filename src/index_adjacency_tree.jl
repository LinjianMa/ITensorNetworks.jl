# Note that the children ordering matters here.
mutable struct IndexAdjacencyTree
  children::Union{Vector{IndexAdjacencyTree},Vector{IndexGroup}}
  fixed_order::Bool
end

function Base.copy(tree::IndexAdjacencyTree)
  node_to_copynode = Dict{IndexAdjacencyTree,IndexAdjacencyTree}()
  for node in topo_sort(tree; type=IndexAdjacencyTree)
    if node.children isa Vector{IndexGroup}
      node_to_copynode[node] = IndexAdjacencyTree(node.children, node.fixed_order)
      continue
    end
    copynode = IndexAdjacencyTree(
      [node_to_copynode[n] for n in node.children], node.fixed_order
    )
    node_to_copynode[node] = copynode
  end
  return node_to_copynode[tree]
end

function Base.show(io::IO, tree::IndexAdjacencyTree)
  out_str = "\n"
  stack = [tree]
  node_to_level = Dict{IndexAdjacencyTree,Int}()
  node_to_level[tree] = 0
  # pre-order traversal
  while length(stack) != 0
    node = pop!(stack)
    indent_vec = ["  " for _ in 1:node_to_level[node]]
    indent = string(indent_vec...)
    if node.children isa Vector{IndexGroup}
      for c in node.children
        out_str = out_str * indent * string(c) * "\n"
      end
    else
      out_str =
        out_str * indent * "AdjTree:  [fixed_order]: " * string(node.fixed_order) * "\n"
      for c in node.children
        node_to_level[c] = node_to_level[node] + 1
        push!(stack, c)
      end
    end
  end
  return print(io, out_str)
end

IndexAdjacencyTree(index_group::IndexGroup) = IndexAdjacencyTree([index_group], false)

function get_adj_tree_leaves(tree::IndexAdjacencyTree)
  if tree.children isa Vector{IndexGroup}
    return tree.children
  end
  leaves = [get_adj_tree_leaves(c) for c in tree.children]
  return vcat(leaves...)
end

function Base.contains(adj_tree::IndexAdjacencyTree, adj_igs::Set{IndexGroup})
  leaves = Set(get_adj_tree_leaves(adj_tree))
  return issubset(adj_igs, leaves)
end

function Base.iterate(x::IndexAdjacencyTree)
  return iterate(x, 1)
end

function Base.iterate(x::IndexAdjacencyTree, index)
  if index > length(x.children)
    return nothing
  end
  return x.children[index], index + 1
end

function boundary_state(ancestor::IndexAdjacencyTree, adj_igs::Set{IndexGroup})
  if ancestor.children isa Vector{IndexGroup}
    return "all"
  end
  if !ancestor.fixed_order
    filter_children = filter(a -> contains(a, adj_igs), ancestor.children)
    # length(filter_children) < 1 means adj_igs is distributed in multiple children
    @assert length(filter_children) <= 1
    if length(filter_children) == 1
      return "middle"
    elseif Set(get_adj_tree_leaves(ancestor)) == adj_igs
      return "all"
    else
      return "invalid"
    end
  end
  @assert length(ancestor.children) >= 2
  if Set(get_adj_tree_leaves(ancestor)) == adj_igs
    return "all"
  end
  for i in 1:(length(ancestor.children) - 1)
    leaves = vcat([get_adj_tree_leaves(c) for c in ancestor.children[1:i]]...)
    if Set(leaves) == adj_igs
      return "left"
    end
  end
  for i in 2:length(ancestor.children)
    leaves = vcat([get_adj_tree_leaves(c) for c in ancestor.children[i:end]]...)
    if Set(leaves) == adj_igs
      return "right"
    end
  end
  return "invalid"
end

function reorder_to_right!(
  ancestor::IndexAdjacencyTree, filter_children::Vector{IndexAdjacencyTree}
)
  remain_children = setdiff(ancestor.children, filter_children)
  @assert length(filter_children) >= 1
  @assert length(remain_children) >= 1
  if length(remain_children) == 1
    new_child1 = remain_children[1]
  else
    new_child1 = IndexAdjacencyTree(remain_children, false)
  end
  if length(filter_children) == 1
    new_child2 = filter_children[1]
  else
    new_child2 = IndexAdjacencyTree(filter_children, false)
  end
  ancestor.children = [new_child1, new_child2]
  return ancestor.fixed_order = true
end

"""
reorder adj_tree based on adj_igs
"""
function reorder!(adj_tree::IndexAdjacencyTree, adj_igs::Set{IndexGroup}; boundary="right")
  @assert boundary in ["left", "right"]
  if boundary_state(adj_tree, adj_igs) == "all"
    return false
  end
  adj_trees = topo_sort(adj_tree; type=IndexAdjacencyTree)
  path = [tree for tree in adj_trees if contains(tree, adj_igs)]
  ancestor_to_state = Dict{IndexAdjacencyTree,String}()
  # get the boundary state
  for ancestor in path
    state = boundary_state(ancestor, adj_igs)
    if state == "invalid"
      return false
    end
    ancestor_to_state[ancestor] = state
  end
  # update path
  for ancestor in path
    # reorder
    if ancestor_to_state[ancestor] == "left"
      @assert ancestor.fixed_order == true
      ancestor.children = reverse(ancestor.children)
    elseif ancestor_to_state[ancestor] == "middle"
      @assert ancestor.fixed_order == false
      filter_children = filter(a -> contains(a, adj_igs), ancestor.children)
      reorder_to_right!(ancestor, filter_children)
    end
  end
  # check boundary
  if boundary == "left"
    for ancestor in path
      ancestor.children = reverse(ancestor.children)
    end
  end
  return true
end

# Update both keys and values in igs_to_adjacency_tree based on adjacent_igs
# NOTE: keys of `igs_to_adjacency_tree` are target igs, not those adjacent to ancestors
function update_igs_to_adjacency_tree!(
  adjacent_igs::Set{IndexGroup},
  igs_to_adjacency_tree::Dict{Set{IndexGroup},IndexAdjacencyTree},
)
  @timeit_debug ITensors.timer "update_igs_to_adjacency_tree" begin
    # get each root igs, get the adjacent igs needed. TODO: do we need to consider boundaries here?
    root_igs_to_adjacent_igs = Dict{Set{IndexGroup},Set{IndexGroup}}()
    for root_igs in keys(igs_to_adjacency_tree)
      common_igs = intersect(adjacent_igs, root_igs)
      if common_igs != Set()
        root_igs_to_adjacent_igs[root_igs] = common_igs
      end
    end
    if length(root_igs_to_adjacent_igs) == 1
      return nothing
    end
    # if at least 3: for now just put everything together
    if length(root_igs_to_adjacent_igs) >= 3
      root_igs = keys(root_igs_to_adjacent_igs)
      root = union(root_igs...)
      igs_to_adjacency_tree[root] = IndexAdjacencyTree(
        [igs_to_adjacency_tree[r] for r in root_igs], false
      )
      for r in root_igs
        delete!(igs_to_adjacency_tree, r)
      end
      return nothing
    end
    # if 2: assign adjacent_igs to boundary of root_igs (if possible), then concatenate
    igs1, igs2 = collect(keys(root_igs_to_adjacent_igs))
    reordered_1 = reorder!(
      igs_to_adjacency_tree[igs1], root_igs_to_adjacent_igs[igs1]; boundary="right"
    )
    reordered_2 = reorder!(
      igs_to_adjacency_tree[igs2], root_igs_to_adjacent_igs[igs2]; boundary="left"
    )
    adj_tree_1 = igs_to_adjacency_tree[igs1]
    adj_tree_2 = igs_to_adjacency_tree[igs2]
    if (!reordered_1) && (!reordered_2)
      out_adj_tree = IndexAdjacencyTree([adj_tree_1, adj_tree_2], false)
    elseif (!reordered_1)
      out_adj_tree = IndexAdjacencyTree([adj_tree_1, adj_tree_2.children...], true)
    elseif (!reordered_2)
      out_adj_tree = IndexAdjacencyTree([adj_tree_1.children..., adj_tree_2], true)
    else
      out_adj_tree = IndexAdjacencyTree(
        [adj_tree_1.children..., adj_tree_2.children...], true
      )
    end
    root_igs = keys(root_igs_to_adjacent_igs)
    root = union(root_igs...)
    igs_to_adjacency_tree[root] = out_adj_tree
    for r in root_igs
      delete!(igs_to_adjacency_tree, r)
    end
  end
end

# Generate the adjacency tree of a contraction tree
# Args:
# ==========
# ctree: the input contraction tree
# path: the path containing ancestor ctrees of the input ctree
# ctree_to_igs: mapping each ctree to neighboring index groups 
function generate_adjacency_tree(ctree, path, ctree_to_igs)
  @timeit_debug ITensors.timer "generate_adjacency_tree" begin
    # mapping each index group to adjacent input igs
    ig_to_input_adj_igs = Dict{IndexGroup,Set{IndexGroup}}()
    # mapping each igs to an adjacency tree
    # TODO: better to rewrite igs_to_adjacency_tree based on a disjoint set
    igs_to_adjacency_tree = Dict{Set{IndexGroup},IndexAdjacencyTree}()
    for ig in ctree_to_igs[ctree]
      ig_to_input_adj_igs[ig] = Set([ig])
      igs_to_adjacency_tree[Set([ig])] = IndexAdjacencyTree(ig)
    end
    for (i, a) in path
      inter_igs = intersect(ctree_to_igs[a[1]], ctree_to_igs[a[2]])
      new_igs_index = (i == 1) ? 2 : 1
      new_igs = setdiff(ctree_to_igs[a[new_igs_index]], inter_igs)
      adjacent_igs = union([ig_to_input_adj_igs[ig] for ig in inter_igs]...)
      # `inter_igs != []` means it's a tensor product
      if inter_igs != []
        update_igs_to_adjacency_tree!(adjacent_igs, igs_to_adjacency_tree)
      end
      for ig in new_igs
        ig_to_input_adj_igs[ig] = adjacent_igs
      end
      if length(igs_to_adjacency_tree) == 1
        return collect(values(igs_to_adjacency_tree))[1]
      end
    end
    if length(igs_to_adjacency_tree) >= 1
      return IndexAdjacencyTree([collect(values(igs_to_adjacency_tree))...], false)
    end
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

function minswap_adjacency_tree!(adj_tree::IndexAdjacencyTree)
  # TODO: this seems not correct, need to 1) generate the reference ordering, and 2) minimize the Kendall-Tau distance
  leaves = Vector{IndexGroup}(get_adj_tree_leaves(adj_tree))
  adj_tree.children = leaves
  return adj_tree.fixed_order = true
end

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
function mindist_adjacency_tree!(
  adj_tree::IndexAdjacencyTree, input_tree::IndexAdjacencyTree, tensors::Vector{ITensor}
)
  for node in topo_sort(adj_tree; type=IndexAdjacencyTree)
    if node.children isa Vector{IndexGroup}
      continue
    end
    children_tree = [get_adj_tree_leaves(n) for n in node.children]
    input_order = [n for n in input_tree.children if n in vcat(children_tree...)]
    # Optimize the ordering of children. Note that for each child tree,
    # its ordering is fixed so we don't optimize that.
    if node.fixed_order
      perms = [children_tree, reverse(children_tree)]
      nswaps = [num_adj_swaps(vcat(p...), input_order) for p in perms]
      perms = [perms[i] for i in 1:length(perms) if nswaps[i] == min(nswaps...)]
      children_tree = _mincut_permutation(perms, tensors)
    else
      children_tree = _best_perm_greedy(children_tree, input_order, tensors)
    end
    node.children = vcat(children_tree...)
    node.fixed_order = true
  end
  nswap = num_adj_swaps(adj_tree.children, input_tree.children)
  return nswap
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
      push!(out_lists, IndexAdjacencyTree([l..., r...], true))
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
function minswap_adjacency_tree(
  adj_tree::IndexAdjacencyTree,
  input_tree1::IndexAdjacencyTree,
  input_tree2::IndexAdjacencyTree,
  input1_in_leaves::Bool,
  input2_in_leaves::Bool,
  tensors::Vector{ITensor},
)
  @timeit_debug ITensors.timer "minswap_adjacency_tree" begin
    leaves_1 = get_adj_tree_leaves(input_tree1)
    leaves_2 = get_adj_tree_leaves(input_tree2)
    inter_igs = intersect(leaves_1, leaves_2)
    leaves_1_left, leaves_1_right = split_igs(leaves_1, inter_igs)
    leaves_2_left, leaves_2_right = split_igs(leaves_2, inter_igs)
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
      inputs = [
        IndexAdjacencyTree([leaves_2_left..., i..., leaves_2_right...], true) for
        i in inputs
      ]
    elseif !input1_in_leaves && input2_in_leaves
      inputs = collect(permutations([leaves_2_left..., leaves_2_right...]))
      inputs = [
        IndexAdjacencyTree([leaves_1_left..., i..., leaves_1_right...], true) for
        i in inputs
      ]
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
    nswaps_dist_list = [
      mindist_adjacency_tree!(t, i, tensors) for (t, i) in zip(adj_tree_copies, inputs)
    ]
    inputs = [
      inputs[i] for i in 1:length(inputs) if nswaps_dist_list[i] == min(nswaps_dist_list...)
    ]
    adj_tree_copies = [
      adj_tree_copies[i] for
      i in 1:length(adj_tree_copies) if nswaps_dist_list[i] == min(nswaps_dist_list...)
    ]
    if length(inputs) == 1
      return inputs[1], adj_tree_copies[1]
    end
    mincuts = [
      _comb_mincuts_dist(ITensorNetwork(tensors), [igs.data for igs in adj_tree.children])
      for adj_tree in adj_tree_copies
    ]
    return inputs[argmin(mincuts)], adj_tree_copies[argmin(mincuts)]
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
