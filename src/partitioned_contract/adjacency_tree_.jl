function boundary_state(v_data::Vector{Any}, adj_igs::Set)
  if Set(Leaves(v_data[1])) == adj_igs
    return "all"
  end
  if v_data[2] == "unordered"
    filter_children = filter(c -> issubset(adj_igs, Set(Leaves(c))), v_data[1])
    # length(filter_children) < 1 means adj_igs is distributed in multiple children
    @assert length(filter_children) <= 1
    if length(filter_children) == 1
      return "middle"
    end
    # TODO: if more than 1 children contain adj_igs, currently we don't reorder the
    # leaves. This may need to be optimized later.
    return "invalid"
  end
  @assert length(v_data[1]) >= 2
  for i in 1:(length(v_data[1]) - 1)
    leaves = vcat([Set(Leaves(c)) for c in v_data[1][1:i]]...)
    if Set(leaves) == adj_igs
      return "left"
    end
  end
  for i in 2:length(v_data[1])
    leaves = vcat([Set(Leaves(c)) for c in v_data[1][i:end]]...)
    if Set(leaves) == adj_igs
      return "right"
    end
  end
  return "invalid"
end

# Create a new vertex used to replace `v` that reorders `target_child` to the boundary 
function reorder_to_boundary!(
  adj_tree::DataGraph,
  v::Integer,
  target_child::Integer,
  v_index::Integer;
  direction::String="right",
)
  new_v = v
  children = child_vertices(adj_tree, v)
  remain_children = setdiff(children, [target_child])
  @assert length(remain_children) >= 1
  if length(remain_children) == 1
    remain_child = remain_children[1]
    if direction == "right"
      new_data = [[adj_tree[remain_child][1], adj_tree[target_child][1]], "ordered"]
    else
      new_data = [[adj_tree[target_child][1], adj_tree[remain_child][1]], "ordered"]
    end
    if new_data != adj_tree[v]
      _add_vertex_edges!(
        adj_tree, v_index; children=children, parent=parent_vertex(adj_tree, v)
      )
      adj_tree[v_index] = new_data
      new_v = v_index
      v_index += 1
      rem_vertex!(adj_tree, v)
    end
  else
    new_child_data = [[adj_tree[v][1] for v in remain_children], "unordered"]
    _add_vertex_edges!(adj_tree, v_index; children=remain_children, parent=v)
    adj_tree[v_index] = new_child_data
    v_index += 1
    if direction == "right"
      new_data = [[new_child_data[1], adj_tree[target_child][1]], "ordered"]
    else
      new_data = [[adj_tree[target_child][1], new_child_data[1]], "ordered"]
    end
    _add_vertex_edges!(
      adj_tree,
      v_index;
      children=[v_index - 1, target_child],
      parent=parent_vertex(adj_tree, v),
    )
    adj_tree[v_index] = new_data
    new_v = v_index
    v_index += 1
    rem_vertex!(adj_tree, v)
  end
  return new_v, v_index
end

function _add_vertex_edges!(
  adj_tree::DataGraph,
  v::Integer;
  children::Vector=[],
  parent::Union{Nothing,Integer}=nothing,
)
  @info "typeof(adj_tree)", typeof(adj_tree)
  @timeit_debug ITensors.timer "_add_vertex_edges!" begin
    add_vertex!(adj_tree, v)
    if parent != nothing
      add_edge!(adj_tree, parent => v)
    end
    for c in children
      add_edge!(adj_tree, v => c)
    end
  end
end

"""
reorder adj_tree based on adj_igs
"""
function reorder!(
  adj_tree::DataGraph,
  root::Integer,
  adj_igs::Set,
  v_index::Integer;
  boundary::String="right",
)
  @timeit_debug ITensors.timer "reorder!" begin
    @assert boundary in ["left", "right"]
    @timeit_debug ITensors.timer "precompile boundary_state" begin
      if boundary_state(adj_tree[root], adj_igs) == "all"
        return false, root, v_index
      end
    end
    sub_tree = subgraph(
      v -> issubset(Set(Leaves(adj_tree[v][1])), Set(Leaves(adj_tree[root][1]))), adj_tree
    )
    traversal = post_order_dfs_vertices(sub_tree, root)
    path = [v for v in traversal if issubset(adj_igs, Set(Leaves(adj_tree[v][1])))]
    new_root = root
    # get the boundary state
    v_to_state = Dict{Integer,String}()
    for v in path
      @timeit_debug ITensors.timer "precompile boundary_state" begin
        state = boundary_state(adj_tree[v], adj_igs)
      end
      if state == "invalid"
        return false, root, v_index
      end
      v_to_state[v] = state
    end
    for v in path
      children = child_vertices(adj_tree, v)
      # reorder
      if v_to_state[v] in ["left", "right"] && v_to_state[v] != boundary
        @assert adj_tree[v][2] == "ordered"
        new_data = (reverse(adj_tree[v][1]), adj_tree[v][2])
        new_root = (v == root) ? v_index : new_root
        _add_vertex_edges!(
          adj_tree, v_index; children=children, parent=parent_vertex(adj_tree, v)
        )
        adj_tree[v_index] = new_data
        v_index += 1
        rem_vertex!(adj_tree, v)
      elseif v_to_state[v] == "middle"
        @assert adj_tree[v][2] == "unordered"
        target_child = filter(c -> issubset(adj_igs, Set(Leaves(adj_tree[c][1]))), children)
        @assert length(target_child) == 1
        new_v, v_index = reorder_to_boundary!(
          adj_tree, v, target_child[1], v_index; direction=boundary
        )
        new_root = (v == root) ? new_v : new_root
      end
    end
    return true, new_root, v_index
  end
end

# Update both keys and values in igs_to_adjacency_tree based on adjacent_igs
# NOTE: keys of `igs_to_adjacency_tree` are target igs, not those adjacent to ancestors
function update_adjacency_tree!(
  adjacency_tree::DataGraph, adjacent_igs::Set, v_index::Integer
)
  @timeit_debug ITensors.timer "update_adjacency_tree" begin
    root_v_to_adjacent_igs = Dict{Integer,Set}()
    for r in _roots(adjacency_tree)
      root_igs = Set(Leaves(adjacency_tree[r][1]))
      common_igs = intersect(adjacent_igs, root_igs)
      if common_igs != Set()
        root_v_to_adjacent_igs[r] = common_igs
      end
    end
    if length(root_v_to_adjacent_igs) == 1
      return v_index
    end
    # if at least 3: for now just put everything together
    if length(root_v_to_adjacent_igs) >= 3
      __roots = collect(keys(root_v_to_adjacent_igs))
      new_data = [[adjacency_tree[r][1] for r in __roots], "unordered"]
      _add_vertex_edges!(adjacency_tree, v_index; children=__roots)
      adjacency_tree[v_index] = new_data
      v_index += 1
      return v_index
    end
    # if 2: assign adjacent_igs to boundary of root_igs (if possible), then concatenate
    v1, v2 = collect(keys(root_v_to_adjacent_igs))
    reordered_1, update_v1, v_index = reorder!(
      adjacency_tree, v1, root_v_to_adjacent_igs[v1], v_index; boundary="right"
    )
    reordered_2, update_v2, v_index = reorder!(
      adjacency_tree, v2, root_v_to_adjacent_igs[v2], v_index; boundary="left"
    )
    cs1 = child_vertices(adjacency_tree, update_v1)
    cs2 = child_vertices(adjacency_tree, update_v2)
    if (!reordered_1) && (!reordered_2)
      new_data = [[adjacency_tree[update_v1][1], adjacency_tree[update_v2][1]], "unordered"]
      _add_vertex_edges!(adjacency_tree, v_index; children=[update_v1, update_v2])
      adjacency_tree[v_index] = new_data
      v_index += 1
    elseif (reordered_2)
      new_data = [
        [adjacency_tree[update_v1][1], adjacency_tree[update_v2][1]...], "ordered"
      ]
      _add_vertex_edges!(adjacency_tree, v_index; children=[update_v1, cs2...])
      adjacency_tree[v_index] = new_data
      v_index += 1
      rem_vertex!(adjacency_tree, update_v2)
    elseif (reordered_1)
      new_data = [
        [adjacency_tree[update_v1][1]..., adjacency_tree[update_v2][1]], "ordered"
      ]
      _add_vertex_edges!(adjacency_tree, v_index; children=[update_v2, cs1...])
      adjacency_tree[v_index] = new_data
      v_index += 1
      rem_vertex!(adjacency_tree, update_v1)
    else
      new_data = [
        [adjacency_tree[update_v1][1]..., adjacency_tree[update_v2][1]...], "ordered"
      ]
      _add_vertex_edges!(adjacency_tree, v_index; children=[cs1..., cs2...])
      adjacency_tree[v_index] = new_data
      v_index += 1
      rem_vertex!(adjacency_tree, update_v1)
      rem_vertex!(adjacency_tree, update_v2)
    end
    return v_index
  end
end

# Generate the adjacency tree of a contraction tree
# TODO: add test
function _adjacency_tree(v::Tuple, path::Vector, par::DataGraph, p_edge_to_inds::Dict)
  @timeit_debug ITensors.timer "_adjacency_tree" begin
    # mapping each index group to adjacent input igs
    ig_to_input_adj_igs = Dict{Any,Set}()
    # mapping each igs to an adjacency tree
    adjacency_tree = DataGraph(NamedDiGraph{Integer}(), Vector)
    p_leaves = vcat(v[1:(end - 1)]...)
    p_edges = _neighbor_edges(par, p_leaves)
    v_index = 1
    for ig in map(e -> Set(p_edge_to_inds[e]), p_edges)
      ig_to_input_adj_igs[ig] = Set([ig])
      add_vertex!(adjacency_tree, v_index)
      # Note: we let the data be a nested vector rather than the nested tuple
      # since the former is more efficient for function precompilation
      # (the type will just be Vector{Any})
      adjacency_tree[v_index] = [[ig], "unordered"]
      v_index += 1
    end
    for contraction in path
      ancester_leaves = filter(u -> issubset(p_leaves, u), contraction[1:2])[1]
      sibling_leaves = setdiff(contraction[1:2], [ancester_leaves])[1]
      ancester_igs = map(e -> Set(p_edge_to_inds[e]), _neighbor_edges(par, ancester_leaves))
      sibling_igs = map(e -> Set(p_edge_to_inds[e]), _neighbor_edges(par, sibling_leaves))
      inter_igs = intersect(ancester_igs, sibling_igs)
      new_igs = setdiff(sibling_igs, inter_igs)
      adjacent_igs = union([ig_to_input_adj_igs[ig] for ig in inter_igs]...)
      # `inter_igs != []` means it's a tensor product
      if inter_igs != []
        v_index = update_adjacency_tree!(adjacency_tree, adjacent_igs, v_index)
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
      new_data = [[adjacency_tree[r][1] for r in __roots], "unordered"]
      _add_vertex_edges!(adjacency_tree, v_index; children=__roots)
      adjacency_tree[v_index] = new_data
      v_index += 1
    end
    return adjacency_tree
  end
end
