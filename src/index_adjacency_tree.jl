# Note that the children ordering matters here.
mutable struct IndexAdjacencyTree
  children::Union{Vector{IndexAdjacencyTree},Vector{IndexGroup}}
  fixed_direction::Bool
  fixed_order::Bool
end

function Base.copy(tree::IndexAdjacencyTree)
  node_to_copynode = Dict{IndexAdjacencyTree,IndexAdjacencyTree}()
  for node in topo_sort(tree; type=IndexAdjacencyTree)
    if node.children isa Vector{IndexGroup}
      node_to_copynode[node] = IndexAdjacencyTree(
        node.children, node.fixed_direction, node.fixed_order
      )
      continue
    end
    copynode = IndexAdjacencyTree(
      [node_to_copynode[n] for n in node.children], node.fixed_direction, node.fixed_order
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
        out_str *
        indent *
        "AdjTree: [fixed_direction]: " *
        string(node.fixed_direction) *
        " [fixed_order]: " *
        string(node.fixed_order) *
        "\n"
      for c in node.children
        node_to_level[c] = node_to_level[node] + 1
        push!(stack, c)
      end
    end
  end
  return print(io, out_str)
end

function IndexAdjacencyTree(index_group::IndexGroup)
  return IndexAdjacencyTree([index_group], false, false)
end

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
  if contains(ancestor.children[1], adj_igs)
    return "left"
  elseif contains(ancestor.children[end], adj_igs)
    return "right"
  elseif Set(get_adj_tree_leaves(ancestor)) == adj_igs
    return "all"
  else
    return "invalid"
  end
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
    new_child1 = IndexAdjacencyTree(remain_children, false, false)
  end
  if length(filter_children) == 1
    new_child2 = filter_children[1]
  else
    new_child2 = IndexAdjacencyTree(filter_children, false, false)
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
  ancestors = [tree for tree in adj_trees if contains(tree, adj_igs)]
  ancestor_to_state = Dict{IndexAdjacencyTree,String}()
  # get the boundary state
  for ancestor in ancestors
    state = boundary_state(ancestor, adj_igs)
    if state == "invalid"
      return false
    end
    ancestor_to_state[ancestor] = state
  end
  # update ancestors
  for ancestor in ancestors
    # reorder
    if ancestor_to_state[ancestor] == "left"
      ancestor.children = reverse(ancestor.children)
    elseif ancestor_to_state[ancestor] == "middle"
      @assert ancestor.fixed_order == false
      filter_children = filter(a -> contains(a, adj_igs), ancestor.children)
      reorder_to_right!(ancestor, filter_children)
    end
    # merge
    if ancestor.fixed_order && ancestor.children isa Vector{IndexAdjacencyTree}
      new_children = Vector{IndexAdjacencyTree}()
      for child in ancestor.children
        if !child.fixed_order
          push!(new_children, child)
        else
          push!(new_children, child.children...)
        end
      end
      ancestor.children = new_children
    end
  end
  # check boundary
  if boundary == "left"
    for ancestor in ancestors
      ancestor.children = reverse(ancestor.children)
    end
  end
  return true
end

# Update both keys and values in igs_to_adjacency_tree based on list_adjacent_igs
# NOTE: keys of `igs_to_adjacency_tree` are target igs, not those adjacent to ancestors
function update_igs_to_adjacency_tree!(
  list_adjacent_igs::Vector, igs_to_adjacency_tree::Dict{Set{IndexGroup},IndexAdjacencyTree}
)
  function update!(root_igs, adjacent_igs)
    if !haskey(root_igs_to_adjacent_igs, root_igs)
      root_igs_to_adjacent_igs[root_igs] = adjacent_igs
    else
      val = root_igs_to_adjacent_igs[root_igs]
      root_igs_to_adjacent_igs[root_igs] = union(val, adjacent_igs)
    end
  end
  @timeit_debug ITensors.timer "update_igs_to_adjacency_tree" begin
    # get each root igs, get the adjacent igs needed. TODO: do we need to consider boundaries here?
    root_igs_to_adjacent_igs = Dict{Set{IndexGroup},Set{IndexGroup}}()
    for adjacent_igs in list_adjacent_igs
      for root_igs in keys(igs_to_adjacency_tree)
        if issubset(adjacent_igs, root_igs)
          update!(root_igs, adjacent_igs)
        end
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
        [igs_to_adjacency_tree[r] for r in root_igs], false, false
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
      out_adj_tree = IndexAdjacencyTree([adj_tree_1, adj_tree_2], false, false)
    elseif (!reordered_1)
      out_adj_tree = IndexAdjacencyTree([adj_tree_1, adj_tree_2.children...], false, true)
    elseif (!reordered_2)
      out_adj_tree = IndexAdjacencyTree([adj_tree_1.children..., adj_tree_2], false, true)
    else
      out_adj_tree = IndexAdjacencyTree(
        [adj_tree_1.children..., adj_tree_2.children...], false, true
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
# ancestors: ancestor ctrees of the input ctree
# ctree_to_igs: mapping each ctree to neighboring index groups 
function generate_adjacency_tree(ctree, ancestors, ctree_to_igs)
  @timeit_debug ITensors.timer "generate_adjacency_tree" begin
    # mapping each index group to adjacent input igs
    ig_to_adjacent_igs = Dict{IndexGroup,Set{IndexGroup}}()
    # mapping each igs to an adjacency tree
    igs_to_adjacency_tree = Dict{Set{IndexGroup},IndexAdjacencyTree}()
    for ig in ctree_to_igs[ctree]
      ig_to_adjacent_igs[ig] = Set([ig])
      igs_to_adjacency_tree[Set([ig])] = IndexAdjacencyTree(ig)
    end
    for (i, a) in ancestors
      inter_igs = intersect(ctree_to_igs[a[1]], ctree_to_igs[a[2]])
      new_igs_index = (i == 1) ? 2 : 1
      new_igs = setdiff(ctree_to_igs[a[new_igs_index]], inter_igs)
      # Tensor product is not considered for now
      @assert length(inter_igs) >= 1
      list_adjacent_igs = [ig_to_adjacent_igs[ig] for ig in inter_igs]
      update_igs_to_adjacency_tree!(list_adjacent_igs, igs_to_adjacency_tree)
      for ig in new_igs
        ig_to_adjacent_igs[ig] = union(list_adjacent_igs...)
      end
      if length(igs_to_adjacency_tree) == 1
        return collect(values(igs_to_adjacency_tree))[1]
      end
    end
    if length(igs_to_adjacency_tree) >= 1
      return IndexAdjacencyTree([collect(values(igs_to_adjacency_tree))...], false, false)
    end
  end
end

# Mutates `v` by sorting elements `x[lo:hi]` using the insertion sort algorithm.
# This method is a copy-paste-edit of sort! in base/sort.jl, amended to return the bubblesort distance.
function _insertion_sort(v::Vector, lo::Int, hi::Int)
  @timeit_debug ITensors.timer "_insertion_sort" begin
    v = copy(v)
    if lo == hi
      return 0
    end
    nswaps = 0
    for i in (lo + 1):hi
      j = i
      x = v[i]
      while j > lo
        if x < v[j - 1]
          nswaps += 1
          v[j] = v[j - 1]
          j -= 1
          continue
        end
        break
      end
      v[j] = x
    end
    return nswaps
  end
end

function insertion_sort(v1::Vector, v2::Vector)
  value_to_index = Dict{Int,Int}()
  for (i, v) in enumerate(v2)
    value_to_index[v] = i
  end
  new_v1 = [value_to_index[v] for v in v1]
  return _insertion_sort(new_v1, 1, length(new_v1))
end

function minswap_adjacency_tree!(adj_tree::IndexAdjacencyTree)
  leaves = Vector{IndexGroup}(get_adj_tree_leaves(adj_tree))
  adj_tree.children = leaves
  adj_tree.fixed_order = true
  return adj_tree.fixed_direction = true
end

function minswap_adjacency_tree!(
  adj_tree::IndexAdjacencyTree, input_tree::IndexAdjacencyTree
)
  nodes = input_tree.children
  node_to_int = Dict{IndexGroup,Int}()
  int_to_node = Dict{Int,IndexGroup}()
  index = 1
  for node in nodes
    node_to_int[node] = index
    int_to_node[index] = node
    index += 1
  end
  for node in topo_sort(adj_tree; type=IndexAdjacencyTree)
    if node.children isa Vector{IndexGroup}
      continue
    end
    children_tree = [get_adj_tree_leaves(n) for n in node.children]
    children_order = vcat(children_tree...)
    input_int_order = [node_to_int[n] for n in nodes if n in children_order]
    if node.fixed_order
      perms = [children_tree, reverse(children_tree)]
    else
      perms = collect(permutations(children_tree))
    end
    nswaps = []
    for perm in perms
      int_order = [node_to_int[n] for n in vcat(perm...)]
      push!(nswaps, insertion_sort(int_order, input_int_order))
    end
    children_tree = perms[argmin(nswaps)]
    node.children = vcat(children_tree...)
    node.fixed_order = true
    node.fixed_direction = true
  end
  int_order = [node_to_int[n] for n in adj_tree.children]
  return _insertion_sort(int_order, 1, length(int_order))
end

function minswap_adjacency_tree(
  adj_tree::IndexAdjacencyTree,
  input_tree1::IndexAdjacencyTree,
  input_tree2::IndexAdjacencyTree,
)
  function merge(l1_left, l1_right, l2_left, l2_right)
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
    out_lists = []
    for l in left_lists
      for r in right_lists
        push!(out_lists, IndexAdjacencyTree([l..., r...], true, true))
      end
    end
    return out_lists
  end
  @timeit_debug ITensors.timer "minswap_adjacency_tree" begin
    leaves_1 = get_adj_tree_leaves(input_tree1)
    leaves_2 = get_adj_tree_leaves(input_tree2)
    inter_igs = intersect(leaves_1, leaves_2)
    leaves_1_left, leaves_1_right = split_igs(leaves_1, inter_igs)
    leaves_2_left, leaves_2_right = split_igs(leaves_2, inter_igs)
    num_swaps_1 =
      min(length(leaves_1_left), length(leaves_2_left)) +
      min(length(leaves_1_right), length(leaves_2_right))
    num_swaps_2 =
      min(length(leaves_1_left), length(leaves_2_right)) +
      min(length(leaves_1_right), length(leaves_2_left))
    if num_swaps_1 == num_swaps_2
      inputs_1 = merge(leaves_1_left, leaves_1_right, leaves_2_left, leaves_2_right)
      inputs_2 = merge(
        leaves_1_left, leaves_1_right, reverse(leaves_2_right), reverse(leaves_2_left)
      )
      inputs = [inputs_1..., inputs_2...]
    elseif num_swaps_1 > num_swaps_2
      inputs = merge(
        leaves_1_left, leaves_1_right, reverse(leaves_2_right), reverse(leaves_2_left)
      )
    else
      inputs = merge(leaves_1_left, leaves_1_right, leaves_2_left, leaves_2_right)
    end
    # TODO: may want to change this back
    # leaves_1 = [i for i in leaves_1 if !(i in inter_igs)]
    # leaves_2 = [i for i in leaves_2 if !(i in inter_igs)]
    # input1 = IndexAdjacencyTree([leaves_1..., leaves_2...], true, true)
    # input2 = IndexAdjacencyTree([leaves_1..., reverse(leaves_2)...], true, true)
    # input3 = IndexAdjacencyTree([reverse(leaves_1)..., leaves_2...], true, true)
    # input4 = IndexAdjacencyTree([reverse(leaves_1)..., reverse(leaves_2)...], true, true)
    # inputs = [input1, input2, input3, input4]
    # ======================================
    adj_tree_copies = [copy(adj_tree) for _ in 1:length(inputs)]
    nswaps = [minswap_adjacency_tree!(t, i) for (t, i) in zip(adj_tree_copies, inputs)]
    return adj_tree_copies[argmin(nswaps)]
  end
end
